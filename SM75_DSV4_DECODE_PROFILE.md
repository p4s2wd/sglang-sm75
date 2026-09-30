# DeepSeek-V4-Flash on SM75: decode accounting, roofline, and dead ends

Notes from tuning an 8x RTX 2080 Ti (SM75, 22 GiB, 150 W) node running
DeepSeek-V4-Flash with `TP2 x PP4`, KV in fp8_e4m3, 43 layers, 256 routed
experts with top-6, `index_topk=512`, MLA head_dim 512 (448 nope + 64 rope).
Measured state: **26.1 tok/s at batch 1**, 132 tok/s at batch 8, single-request
prefill 8.9k tok/s (6k prompt in 0.68 s TTFT), GSM8K 200q 95.0%.

Everything below is from kineto traces of the running engine plus in-situ
`perf_counter` instrumentation. Numbers that came from a profiler are called
out as such; two of them turned out to be artifacts (see Pitfalls).

## Where a decode token goes

38.3 ms/token at batch 1. The PP4 chain is serial at this batch size, so the
token time is the sum of the four stages' GPU work, not a pipeline throughput:

| stage | GPU busy / step | of which NCCL wait (profiler-induced) |
|-------|-----------------|--------------------------------------|
| PP0   | 6.8 ms          | ~0                                   |
| PP1   | 12.4 ms         | 2.7 ms                               |
| PP2   | 15.1 ms         | 4.2 ms                               |
| PP3   | 12.9 ms         | 3.8 ms                               |

Subtracting the waits lands at ~36.5 ms of real kernel time, which closes the
budget against the 38.3 ms/token the engine reports without a profiler. **Decode
is GPU-work bound.** The 2-4 ms NCCL waits are skew absorption: only the rank
that arrives first spins, and the pair totals the same.

Per-stage kernel census (TP0/PP3, per decode step, 670 kernels, 12.94 ms):

| kernel family | ms/step | nodes | share of HBM peak |
|---------------|---------|-------|-------------------|
| `w4a16_v3` (MoE experts) | 1.50 | 22 | ~46% (289 GB/s) |
| `w8a16_gemv` narrow | 1.27 | 29 | (dense pair below) |
| `w8a16_gemv` wide | 1.09 | 33 | |
| dense gemv pair | 2.36 | 62 | ~508 GB/s = **85%** |
| `headshared_sparse` (MLA) | 0.96 | 22 | tuned out |
| `all_reduce_1shot_push` | 0.76 | 22 | comm |
| `wo_a_absorb` | 0.70 | 11 | **546 GB/s = 92%** |
| unnamed Triton (`std::enable_if`) | 0.53 | 34 | mixed, see tail |
| indexer `mqa_paged_smallq` | 0.31 | 6 | fused this series |
| `elementwise` | 0.22 | 94 | 2 us each |
| `cublasDot` + `reduce_1Block` | 0.22 | 44 | in hc_pre, see tail |
| everything else | ~3.0 | ~350 | |

## Why there is little left

Weights resident per rank: 17.25 GiB of MXFP4-packed experts (138 GB of the
156 GB checkpoint, TP-sharded) + 1.19 GiB of fp8/bf16 dense. A batch-1 token
touches top-6/256 experts, so the per-rank read is ~1.63 GB, which at the
card's 594 GB/s spec is 2.9 ms; serialised over four stages that is an 11.6 ms
floor (86 tok/s). The engine runs at 38.3 ms, i.e. each stage reaches ~30% of
read bandwidth -- but the families that dominate (`w8a16_gemv` 85%, `wo_a_absorb`
92%) are already near the roof, so the shortfall is spread thin rather than
concentrated in one fixable kernel.

## Dead ends, with the evidence

- **PP communication knobs.** `SGLANG_PP_COMM_OVERLAP=1` (dedicated comm
  stream), `SGLANG_PP_EARLY_PROXY_SEND=1`, `SGLANG_ENABLE_DELAY_SAMPLE=1`
  together: 25.7 tok/s vs 26.1 baseline. The `cudaGraphLaunch` cost they were
  meant to hide is 0.1 ms, not 2 ms.
- **`cudaGraphLaunch` "2 ms".** Under `/start_profile` the decode graph's
  replay shows ~2.0-2.3 ms on every stage; measured in-situ it is **0.08-0.11
  ms** (670 nodes, ~0.15 us/node). Two mistakes fed this: the profiler inflates
  the API, and clustering kernels by >0.5 ms gaps splits a step into ~4
  fragments, which made GPU busy look like 2 ms out of a 60 ms period.
- **Decode graph node count as a lever.** Node census via
  `cudaGraphGetNodes`/`cudaGraphNodeGetType` (torch's `raw_cuda_graph()` needs
  `CUDAGraph(keep_graph=True)`; `cudaGraphDebugDotPrint` returns rc=1 on this
  driver): 592 nodes at PP0, 670 at PP3, 99% kernel nodes, 4 memcpy, 1-2 event
  nodes. No hidden multi-stream structure, and at a real 0.15 us/node there is
  nothing to win by fusing small kernels.
- **Whole-row gather rewrite of the MLA kernel.** One gather of the full
  [16, 512] row plus a single k=512 QK dot, instead of 8 chunked dots: bit
  identical output, **2.25x slower** (48.4 vs 21.5 ms at B=512/TOPK=512). The
  wide dot's register staging reproduces exactly the cost the chunking avoids.
- **DSPark speculative decoding.** The checkpoint ships it
  (`dspark_block_size=5`, `dspark_target_layer_ids=[40,41,42]`,
  `mtp.2.markov_head.*`), but the framework rejects it for anything but
  `pp_size == 1` unless it is a PD-disaggregated prefill instance, and 8 cards
  cannot host both roles (weights are 96.8% of each card at 1/8 sharding).
- **EAGLE/MTP speculative decoding.** Passes validation for this architecture
  (`--speculative-algorithm EAGLE --speculative-eagle-topk 1`, plus
  `SGLANG_ENABLE_PP_SPEC=1` for PP) and then OOMs: under PP the draft keeps its
  own copy of the input embedding on the last stage (129280x4096 fp16 = 1010 MiB
  per call) where 0.79 GiB is free after weights. Lowering `--mem-fraction-static`
  does not help because the KV pool is allocated after the draft. Making that
  embedding shard-local and fp8 would fit, at the cost of ~60% of the KV pool.
- **Kernel-argument count as a graph-launch cost.** Synthetic 600-node graphs:
  1 arg 1.13 us/node, 3 args 1.52, 25 args 1.14. Not a factor.
- **HS kernel knobs.** `SGLANG_SM75_HS_{NCOL,WARPS,STAGES,TRANS}` swept
  interleaved at the prefill shape: the shipped NCOL=64 / W4 / S2 / transposed
  is the optimum, no >10% win anywhere.
- **The interconnect.** Topology is four NVLink pairs, (0,1) (2,3) (4,5) (6,7);
  `TP2 x PP4` puts the TP all-reduce inside a pair and only the PP handoff on
  PCIe. Measured D2D with the link under load:

  | pair | path | P2P | 256 MB | 8 KB latency |
  |------|------|-----|---------|--------------|
  | 0<->1 | NVLink pair (where TP2 lands) | yes | **87.3 GB/s** | 46 us |
  | 0<->2 | PIX, same PCIe switch | no | 10.3 GB/s | 50 us |
  | 0<->4 | PHB, across host bridge (PP path) | no | 10.25 GB/s | 48 us |
  | 0<->6 | PHB, across host bridge (PP path) | no | 10.27 GB/s | 50 us |

  So the high-volume path never touches PCIe, and the PCIe path carries almost
  nothing: 8 KB per stage per token at decode (0.15 ms/token, 0.4% of 38.3 ms)
  and 4 MB per boundary per 512-token prefill chunk (1.2 ms of a 57 ms chunk,
  ~2%). GeForce cards have no P2P across PCIe, so cross-pair traffic is
  host-staged, but at these volumes it does not matter, and no PP placement
  escapes the 10.3 GB/s ceiling (PIX and PHB measure the same). Rejected as a
  bottleneck on measurement, not on estimate.

## Concurrency is the remaining headroom

| concurrent requests | aggregate | per request |
|---------------------|-----------|-------------|
| 1 | 25.2 tok/s | 25.2 |
| 2 | 39.5 | 19.7 |
| 4 | 65.6 | 16.4 |
| 8 | 132.2 | 16.5 |

5.2x from 1 to 8, so the batch-1 point is latency-shaped, not capacity-shaped.
Note `--cuda-graph-max-bs-decode 2` in the shipped launcher: batch >2 runs
eager. Raising the graphed batch range is untested and is the cheapest thing
left to try for multi-request serving.

## What is actually left

Small, in order of effort:

1. **hc_pre cublas chain** (0.22 ms/step, 0.9 ms/token, ~1%). The trace shows
   `dot_kernel` (grid 32x1x24, 8 us) + `reduce_1Block_kernel` (3 us) sitting
   between `_hc_prenorm_smallm_kernel` and `hc_split_sinkhorn_kernel`, i.e.
   inside the MHC pre-mix statistics, one pair per layer. Folding them into the
   prenorm kernel is the same move as the cast+rms fusion.
2. **46 `FillFunctor` zero-fills per step** (0.09 ms/step, ~1%). Inside a
   captured decode graph most of these buffers are fully overwritten; they
   should not need clearing every step.
3. **MoE k-split at decode** (medium effort, 3-5%). `SGLANG_SM75_W4A16_KSPLIT=2`
   is already the measured optimum at bs=1 (NT4+KS2: 0.087/0.048 ms vs NT4 KS1
   0.116/0.060), and the source records why KS4/KS8 do not pay: `sorted_ids` is
   the align buffer's *capacity* (3861 rows at decode) rather than the 6 live
   rows, so each split carries a 63 MB fp32 partial that costs as much as the
   GEMM (KS1 0.115, KS2 0.116, KS4 0.116 ms). Sizing the align/partial buffers
   to live rows for the decode graph -- bs is known at capture time, so this
   does not need a device-side `num_valid` -- would let the wider grids that
   measured 337 GB/s at 1536 blocks and 388 at 6144 actually apply. Then sweep
   the cfgs already present in `W4A16_V3_CFGS` / `_KS_CFG`.
4. **All-reduce kernel efficiency, prefill only** (~6% of a chunk, and the only
   interconnect-adjacent item left). The TP all-reduce runs on NVLink at
   87 GB/s, but an EXTEND event of 4 MB takes 0.216 ms, i.e. 18.5 GB/s -- a
   fifth of the link. Getting each event to link speed would save ~3.7 ms per
   512-token chunk. This is the kernel, not the wire, so it is invisible in any
   bandwidth check of the interconnect.

## Pitfalls that cost real time

- **Synthetic KV cache layout.** A bench that fills a page as
  `page_off + t*584 + 448` is wrong: the page stride is 584 but the token stride
  inside a page is 576 (`_HS_TOKEN_BYTES`). The rope bytes then hold random
  values, bf16 pairs like 0x6464 are ~2^73, the cast to fp16 is `inf`, and
  `0 * inf` in the rope dot makes the kernel emit NaN for *every* input -- it
  looks exactly like a kernel bug. Sanity check that survives: a constant
  payload with rope zeroed at the right offset must give a constant output.
- **e4m3 NaN byte.** Payload byte 127 (and 255) are NaN in the lookup table.
  Bench payloads must be `randint(0, 127)`, not `randint(0, 256)`.
- **`nvidia-smi` PCIe generation at idle.** `pcie.link.gen.current` reports 1 on
  all eight cards while nothing is running (ASPM downclock); the link trains to
  Gen3 under load, which is what the 10.3 GB/s D2D numbers above are. Judge the
  link by a copy benchmark, not by the idle reading.
- **Profiling then loading.** A server that has run `/start_profile` should be
  restarted before it serves real load; twice in one session we lost a run to a
  crash that a restart fixed.
- **Self-measured "free" GPU time.** Kernel durations from kineto are
  trustworthy; API durations are not, and neither are gaps you inferred by
  clustering kernels.
