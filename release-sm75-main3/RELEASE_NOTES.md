# RELEASE NOTES — sm75 on current main (`sm75main3`)

Base: upstream `main` @ `98fce73d5b` (unchanged since `sm75main2`). Wheel:
`sglang-0.5.21.dev803+ga588149ab0.sm75main3-py3-none-any.whl`. Source:
[p4s2wd/sglang-sm75](https://github.com/p4s2wd/sglang-sm75) branch
`sm75-dsv4-flash-main`, tagged `v0.2.1-sm75main3`. Six commits over
`sm75main2`, verified on 2026-10-04 on the 8x RTX 2080 Ti box (SM75, 22 GiB,
150 W), TP2 x PP4, `--context-length 262144`, `--chunked-prefill-size 256`,
decode CUDA graphs on bs 1 and 2, KV pool 264,192 tokens,
`kv-cache-dtype fp8_e4m3`.

## Read this first: the 256K context is now real, not declared

`sm75main2` accepted `--context-length 262144` and quietly sized the KV pool at
**162,304 tokens**. That was an overcount, not a hardware limit:
`_get_bytes_per_swa_token` multiplied the paged SWA cost by `num_layers_total`,
every layer the stage owns, while only layers with compress ratio 0 have a
sliding window at all. In the DeepSeek-V4-Flash split those are layers 0 and 1
and both land on PP1, so PP0, PP2 and PP3 each paid SWA bytes for a pool they
never allocate.

Charging only the layers that own a window takes PP0 from 1944.75 to 1109.25
bytes per full token. That commit measured the pool at 267,776 tokens against a
`--max-total-tokens 270000` request; this box reports
`max_total_num_tokens=264192` today, after the memory calculator has taken the
graph buckets and the rest. Either way it clears the 262,144 that was asked for,
so the declared context length is now the actual one -- the same launch on
`sm75main2` reported 162,304. The c4 state term is left alone on purpose: it is
priced per c4 layer and every stage owns c4 layers.

Two consequences a deployer has to know about:

- The freed memory becomes **pool, not headroom**. PP0 ends at 0.46 GB instead
  of 0.64 GB, which is enough for the pool but not for a 256K prefill, whose
  indexer needs `[Q, max_seqlen_k]` fp32 logits plus the fp16 score tile. So
  this release ships `--chunked-prefill-size 256` in the launcher (was 512,
  halving the logits buffer from 133 MiB to 66 MiB at 258K) and `_CHUNK 256` in
  the score kernel (was 1024, 64 MiB -> 16 MiB). Both are pure tiling: the same
  values are read.
- **PP3 at 0.64 GB is now the binding rank**, not PP0. Anything that widens
  memory use -- more graph buckets, a higher `--mem-fraction-static`, a bigger
  `--chunked-prefill-size` -- should be checked against PP3 first.

Measured on the box: a 258,941-token prefill completes, greedy output is
byte-identical to the previous build on 6/6 prompts, and 200 GSM8K questions at
parallel 8 give accuracy 0.945 with zero crashes and zero unhandled CUDA
errors. The smaller prefill chunk costs about nothing: short prompts +16%, long
prompts -10%, mean ~0.98x.

## Read this too: tool calls survive model drift now

An agent loop against the live server was dying in three ways, all of them the
parser's fault rather than the model's. A parameter written without its
`string` attribute, or with that value unquoted or capitalised, did not match
`parameter_regex` at all; the leftover DSML then failed the malformed check and
**the whole tool call was dropped**. The same call closed with a plain tag
instead of a `｜DSML｜` prefixed one failed the same way. And a name written as
`Bash` where the client declared `bash` was passed through untouched, so the
client answered `Tool Bash not found`, the model retried, and the session spun.

Closers now match with the `｜DSML｜` prefix optional (open tags stay strict, so a
tag quoted in prose is not mistaken for a call), the `string` attribute is
optional and read case-insensitively with a JSON-then-string fallback, and name
repair lives in one helper that both the one-shot and the streaming path use.
That last part is the reason the first attempt at this fix did not work in
production: the streaming path builds its tool call directly and never goes
through `parse_base_json`, so a repair made there only served one-shot clients --
and an agent loop is a streaming client.

Names that match no declared tool are still forwarded rather than dropped,
because a client error is what lets the model correct itself; the streaming path
now logs them, which it previously did not, so this class of drift is no longer
invisible in the server log.

Evidence, and the honest limit of it: the attribute and closer fixes are
confirmed by a live 21-turn agent session that logged no dropped call. The name
repair is not yet field-confirmed -- the session that produced the
`Tool Read not found` loop ran before it shipped, and the session after it
happened to spell every name in lowercase. It is covered by tests at three chunk
widths instead of by a log line. `test_deepseekv4_detector` is 32 tests OK, up
from 25; the bounded cases (a self-closed parameter, a parameter that never
closes) still fail closed.

## Long-context decode: the sparse top-K stops running on one SM

`_topk_transform_paged_triton_kernel` was launching with a grid of `[1, 1, 1]`:
one program, one of 68 SMs, 1.5% occupancy at bs=1, and the step grew with
context. Each row is now split into slabs, one program per slab, merged
afterwards. The merge is a full `tl.sort` of each fan-in group rather than the
in-kernel bitonic recurrence, because sorting the concatenation is exact by
construction while the recurrence only merges raw blocks -- that cost some speed
and bought certainty. Measured on RTX 2080 Ti, bs=1, output byte-identical:

| row width | single program | parallel | speedup |
|---|---|---|---|
| 8,192 | 202 us | 122 us | 1.7x |
| 37,500 | 671 us | 111 us | 6.0x |
| 150,000 | 2,908 us | 141 us | 20.6x |
| 262,144 | 4,525 us | 177 us | 25.6x |

It engages from row width 8,192 and up to 16 rows, and those thresholds are
measured rather than guessed: at width 6,144 the parallel path runs at 0.84x of
the single-program kernel, and at 32 rows at 0.93x, so below/above them it
would be a regression. Dispatch is on allocated row width and row count, never
on `seq_lens`, which inside a captured CUDA graph is a device buffer that
changes per replay and cannot select a launch. Scratch is cached per shape and
never freed, because a captured graph holds the pointers and reallocating at a
new address would turn a replay into a read of unrelated memory; a budget caps
the total and running out falls back to the single-program kernel.

Switch: `SGLANG_OPT_SM75_PARALLEL_TOPK`, on by default, `=0` to disable.

## Diagnostics that ship off

`SGLANG_OPT_PP_HANDOFF_DIAG=1` breaks the pipeline handoff into metadata
round trip, per-tensor receive and all-gather, once every
`SGLANG_OPT_PP_HANDOFF_EVERY` calls (default 200), with the payload it carries.
It exists because a py-spy profile attributed a quarter of the bs=1 decode step
to `recv_tensor_dict` and said nothing about which part.

The answer, so nobody spends that time again: at bs=1 steady state the three
hops cost 0.045 ms against a 37.3 ms step -- 0.1%. The profile's time was in
`_pp_commit_comm_work`, waiting on sends issued earlier, which is the pipeline
stall itself and not overhead of its own. Host-side timing only: this runs inside
captured CUDA graphs, so it synchronises with nothing, and with the flag off the
only added cost is a call that returns False.

## How this release was verified

- **Wheel is byte-reproducible.** Two builds of `a588149ab0` produce the same
  sha256 (`3e8a6942…40bdc`), so `SHA256SUMS` is reproducible, not descriptive.
- **The packaged wheel was tested, not just built.** It was extracted and the
  tool-call drift regression run against the extracted code.
- **Both patches reproduce the released source.** `sm75-optimizations.patch`
  applies cleanly to upstream `98fce73d5b` and the series patch applies cleanly
  to `sm75main2`'s `331faaeaf7`; in both cases the resulting tree is identical
  to `a588149ab0`, checked with `git diff --name-only`.
- **Function-call suites, against a read-only worktree of the base commit.**
  The only difference this release introduces is `test_deepseekv4_detector`
  25 OK -> 32 OK. `test_deepseekv41_detector` (4 errors) and
  `test_function_call_parser` (4 failures, 8 errors) were already red at the
  base commit and are unchanged -- they are not caused by anything here.
- **Production on this box runs this same commit** as an editable install, and
  was restarted from it and health-checked with a tool-call request.

## Inherited constraints (unchanged)

Everything in the `sm75main2` notes below still applies: the sub-80 FP8 loader
opt-in, the CUDA 12.9 PTX W4A16 sources, `ninja` on PATH, 150 W,
`sglang-kernel` 0.4.7. One addition worth carrying forward: PP0 loads Triton
kernels *while serving* -- `free device mem` was seen falling from 0.33 GiB to
0.18 GiB as `get_and_clear_swa_pages_kernel` and others compiled late. Pre-loading
them during engine init would remove the last avoidable OOM candidate on that
rank; nobody has done it yet.

---

# Provenance: the `sm75main2` notes, kept as written
Base: upstream `main` @ `98fce73d5b`. Wheel:
`sglang-0.5.21.dev797+g331faaeaf7.sm75main2-py3-none-any.whl`. Verified on
2026-10-01 on the 8x RTX 2080 Ti box (SM75, 22 GiB, 150 W), TP2 x PP4,
`--context-length 262144`, `--chunked-prefill-size 512`, decode CUDA graphs on
bs 1 and 2, KV pool 162K tokens, `kv-cache-dtype fp8_e4m3`.

## Read this first: sm75main1 crashes on the stock configuration

`sm75main1` and everything before it fault with an illegal memory access within
the first few greedy prompts, on the default DeepSeek-V4-Flash path. This is a
correctness bug, not a tuning regression, and it is why `sm75main2` exists.
Upgrade rather than deploying `sm75main1` anywhere new.

`dsv4/topk.py`'s paged top-k transform took `page_table_width` as a parameter
and never used it. `valid` only says `raw < seq_len`, which does not imply
`raw // PAGE_SIZE < width`, so a stale token produced a page id past the end of
the row (observed: 12673630). The sparse-attention gather downstream masks only
`raw >= 0` and is never handed `num_pages`, so that garbage id was used as a KV
address. Binding the load and the store to `page_ids < page_table_width`, and
writing -1 otherwise, takes the observed out-of-range entries from 4 to 0.

Attribution note, because it cost most of the debugging time: an async CUDA
error surfaces at whichever eager launch runs next, and this one produced four
unrelated frames across four archives. `CUDA_LAUNCH_BLOCKING` does not help --
Triton goes through `cuLaunchKernel`, so blocking only moves the report. The
trustworthy stack came from synchronizing the device after every Triton launch,
which bounds the report to "one launch late". That switch
(`SGLANG_TRITON_SYNC_EVERY_LAUNCH`) ships here, off by default, together with
`SGLANG_VALIDATE_SPARSE_INDICES` which names the offending index on the host.

## What else changed since sm75main1

Ten commits. The first seven are decode-side work done on 2026-09-30 evening,
after the `sm75main1` wheel was cut, so they were never released; the last
three are the correctness fix and its diagnostics.

| commit | what |
|---|---|
| `6b10af24c6` | decode e4m3 weights with integer ops instead of the payload LUT |
| `ea338fce88` | decode-shape indexer MQA logits in one Triton launch |
| `6e93a681aa` | fuse the small-batch `hc_pre` cast and RMS statistic |
| `a93bd25001` | merge the decode attention partials and sink in one launch |
| `ede0224de1` | store the LM head as block-scaled fp8 and run the GEMV |
| `98d1ce1df1` | docs: the DSV4 decode accounting and its dead ends |
| `d096e83b16` | docs: measure the interconnect, rule it out as the bottleneck |
| `d9e9824a1e` | **fix**: bound the page id in the paged top-k transform |
| `f308351f2a` | fix: size the breakable-graph attention output by local heads |
| `331faaeaf7` | feat: sparse-index validation and launch-sync diagnostics |

`sm75-main2-series.patch` has all ten; `sm75-optimizations.patch` has the whole
change as one diff against `98fce73d5b`.

## Launcher default: `SGLANG_PP_LAYER_PARTITION=11,11,11,10`

The launcher now defaults to the F3 layer split, worth **+7.6% prefill**. sglang
otherwise hands the remainder of 43 layers over pp=4 to the later stages, giving
`10,11,11,11`, which leaves the last stage as the prefill bottleneck (measured
274.6 ms/chunk against PP0's 215.2, because it also carries lm_head and
sampling). Measured on this box: 8/8 cells positive, 1.041-1.100, decode
unchanged with a bs=1 control of 0.999, greedy output byte-identical to the
default split, and 200 GSM8K questions at parallel 8 with zero crashes.

**The cost is VRAM on PP0**, which goes from 10 layers to 11 and drops from
0.93 GB to 0.64 GB of headroom -- the tightest rank in the fleet. Anything that
adds VRAM to PP0 (wider graph buckets, larger context, higher
`--mem-fraction-static`) will hit this first.

Revert with `SGLANG_PP_LAYER_PARTITION= ./launch-dsv4-sm75.sh`. Note the
declaration deliberately uses `${VAR-default}` and not `${VAR:-default}`: with
`:-` an explicit empty value counts as unset and the default comes back, so the
revert would silently do nothing.

Three other settings in the launcher must stay as they are, all measured:
`PREFILL_GRAPH=disabled` (a captured prefill graph runs prefill at 0.368x on
this configuration, and the `full` backend is not implemented for the dsv4
attention backend at all), `SGLANG_PP_EARLY_PROXY_SEND=0` (-30% on multi-request
decode), `SPEC_ALGO=` empty (EAGLE's draft weights are 5.34 GiB per card at
TP2, and the checkpoint's weights are already 93.7% of the 8x22 GiB capacity, so
it does not fit).

## The sm75main1 notes below are kept for provenance

Everything under this line describes the original rebase onto current main.
Nothing about the kernels changed in that step: the 51 commits from
`sm75-dsv4-flash` and the opt4 patch are both in, with the conflicts resolved
against current main and three upstream-driven adaptations.

| commit | what |
|---|---|
| `caeb58825b` | the 51 commits, squashed onto main |
| `a622747b92` | the opt4 layer, ported |
| `e0f76063c8` | `model_hook`: declare `moe_runner_backend` instead of assigning it |

## Upstream changes that needed real decisions

1. **`ShapeKey.dsa_variant` became `attention_variant`.** Upstream generalised
   the DSA dense/sparse dual-graph switch to cover the `candidate_*` variants.
   The rebased decode runner drops the separate `dsa_variant` channel and
   carries `seq_len_bucket` next to `attention_variant`; capture, replay and
   `_make_graph_key` all build the same key.
2. **`can_run` now gets the bucket too.** The decode graph is stored under a
   `ShapeKey`, and the backends that look graphs up by key
   (`breakable_cuda_graph_backend`, `tc_piecewise_cuda_graph_backend`) answer
   `can_run` with `shape_key in self._graphs`. A key without the bucket would
   miss every captured graph and silently fall back to eager. The pre-rebase
   code had this bug too; it only did not show because the plain CUDA-graph
   backend ignores the key.
3. **The indexer's nonpaged path was split.** Upstream moved the nonpaged K
   gather into its own method and switched the logits call to row slices, so
   the sub-90 Triton branch reads `kv` as a tuple and indexes `rows`.
   `max_c4_seq_len` is now `max_compressed_seq_len`.
4. **`_pp_send_proxy_to_next_stage` grew a fence.** Upstream now sets
   `send_proxy_requires_forward_fence = result.can_run_cuda_graph`, and
   `_pp_commit_proxy_send_work` reads and clears it at the top of the loop. The
   opt4 early-proxy-send path bypasses that helper, so ported as-is it would
   have left the flag unset: a CUDA-graph decode would replay while NCCL was
   still reading the previous step's proxy tensors, which are views of
   replay-owned static buffers. The port sets it, and passes
   `ready_event=self.launch_event` so the ordering wait lands on the comm stream
   the send runs on rather than the schedule stream.
5. **Unquantized `wo_a` follows the model dtype.** Upstream pinned it to
   bf16; on sub-90 the model runs fp16 and the bf16 weight fails the matmul
   dtype check, so the pin becomes `torch.get_default_dtype()` on the
   unquantized path only.
6. **`dsv4/gemm.py` moved to `kernels/ops/gemm/bf16_fp32.py`**, and the sm120
   entry points are now keyed off `_use_torch_sparse_mla()` (true for sm120 and
   for sub-90) instead of an sm120-only check.

## A/B against the build this replaces

The old build was reconstructed exactly: worktree at `sm75-dsv4-flash`
(`60b7c400e7`) plus `sm75-optimizations.patch` from the opt4 release, which is
byte-identical to what was installed in `/data/nvme/sglang/.venv` before this
work. Both builds ran from the same launcher, on the same machine, back to back.

Prefill, median of 2 rounds per length, cache-defeating prompts:

| prompt | old | new | delta |
|---|---|---|---|
| 8K | 1176.9 tok/s | 1305.9 tok/s | +11.0% |
| 13K | 1191.2 tok/s | 1296.8 tok/s | +8.9% |

Decode, bs=1, 64 generated tokens on a radix-cached prefix (so the number is
decode, not prefill):

| context | old | new | delta |
|---|---|---|---|
| 2.9K | 22.23 tok/s | 22.10 tok/s | -0.6% |
| 44K | 15.89 tok/s | 16.76 tok/s | +5.5% |
| 118K | 11.75 tok/s | 12.91 tok/s | +9.9% |
| 235K | 6.33 tok/s | 6.99 tok/s | +10.4% |

Short-context decode through the OpenAI endpoint, 256 tokens: 20.96 old, 20.48
new.

Two runs of the prefill probe on the *same* build differed by 7% (1398 vs 1229
tok/s at 8K) as the box warmed and clocks drifted, so treat anything under
~7% as noise. The prefill and long-context decode gains clear that; the
short-context decode difference does not and is not claimed as a regression.

Where the gain comes from is not isolated. The plausible candidates are the
decode graphs now being captured per KV-length bucket (7 buckets instead of 1 at
this configuration, so a short request no longer scans the full context in the
indexer) and upstream's own work in the 1235 commits. Do not read the table as
"the port made it faster"; read it as "the port did not make it slower, and
the newer base is not slower than the old one at long context".

## Correctness checks

| check | result |
|---|---|
| `fp16_mqa_logits_triton` (opt4's cuBLAS rewrite) vs a float64 reference | max rel err 6.7e-4, median 1.9e-4; `[ks,ke)` masked, padding tail zeroed, no inf, peak logit 19426 of 65504 |
| `topk_transform_paged_triton` vs the torch vectorized transform | selected score sums identical in every row, and identical to an fp64 top-K oracle (diff 0.00e+00); `page_indices` differ only in order |
| `moe_combine` slot map vs the owner scan | both 4.876e-4 max rel err against an fp64 oracle; they differ from each other by 9.7e-4, i.e. fp16 accumulation order |
| `test_scheduler_pp_mixin.py` (opt4's regression test, re-registered for CPU CI) | 2/2 pass |
| `launch-dsv4-sm75.sh --check` | PASS |
| wheel contents vs the running tree | 4961/4961 files byte-identical |

## Things that are not regressions, but will look like bugs

- **The first request after startup is 2.5x slower than steady state** (8.2 vs
  20.5 tok/s). Triton kernels are still being device-loaded on the first
  requests; the log says so (`Triton kernel ... device-loaded after serving
  started`). Warm up before measuring.
- **A `<|begin_of_sentence|>` in the conversation history hangs the model**: it
  reasons until `max_tokens` and returns empty content with
  `finish_reason=length`. Reproduces identically on opt3+opt4, and a normal
  history in the same test returns normally, so it is the checkpoint's
  behaviour. `bos_e2e_verify.py` reports this as 2 FAILs; that is a true
  positive about the model, not about the port.
- **`.claude/skills` is missing from the wheel.** setuptools' file finder lists
  that symlink as a file and then refuses to copy it. The build works around it
  for the build to succeed, but the agent docs do not end up inside the wheel.
  The earlier releases contain them. Nothing at runtime reads them.

## Constraints this release inherits (unchanged from opt3/opt4)

- Accuracy and performance are not upstream-verified; the loader warns on every
  bypass of the capability gate.
- `--chunked-prefill-size` <= 512 and `--mem-fraction-static` >= 0.969 on a
  22 GiB card.
- `sglang-kernel` must be >= 0.4.7 now (upstream's floor moved up).
- `fp8_e4m3` KV is the default and the suspected contributor to long-context
  degradation on this path; `auto` needs ~2x the pool.
