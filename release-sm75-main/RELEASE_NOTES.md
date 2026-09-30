# RELEASE NOTES — sm75 on current main (`sm75main1`)

Base: upstream `main` @ `98fce73d5b`. Wheel:
`sglang-0.5.21.dev797+ge0f76063c.sm75main1-py3-none-any.whl`. Verified on
2026-09-30 on the 8x RTX 2080 Ti box (SM75, 22 GiB, 150 W), TP2 x PP4,
`--context-length 262144`, `--chunked-prefill-size 512`, decode CUDA graphs on
bs 1 and 2, KV pool 270K tokens, `kv-cache-dtype fp8_e4m3`.

## What this release is

The sub-90 work, rebased. Nothing about the kernels changed: the 51 commits from
`sm75-dsv4-flash` and the opt4 patch are both in, with the conflicts resolved
against current main and three upstream-driven adaptations. See
`sm75-main-series.patch` for the three commits and `sm75-optimizations.patch`
for the whole thing as one diff against `98fce73d5b`.

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
