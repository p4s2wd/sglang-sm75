# sglang SM75 optimizations on current main — DeepSeek-V4-Flash on RTX 2080 Ti

This is the `sm75-dsv4-flash` work (51 commits on 0.5.19.dev332) plus the opt4
layer, rebased onto current `main` (`98fce73d5b`) and verified on the 8x 2080 Ti
box. Same behaviour as the opt3+opt4 build it replaces, on a base that is 1235
commits newer.

- wheel: `sglang-0.5.21.dev797+ge0f76063c.sm75main1-py3-none-any.whl`
- the whole change as one patch against upstream main: `sm75-optimizations.patch`
- the three commits as patches: `sm75-main-series.patch`
- launcher: `launch-dsv4-sm75.sh` (identical to `/data/nvme/sglang/deepseek-v4-flash.sh`)
- environment: `prod-constraints.txt` (a copy of the opt4 one with
  `sglang-kernel` bumped to 0.4.7, which current main requires)
- rebuild: `build-wheel.sh`, which is byte-reproducible -- two runs of the same
  commit produce the same sha256, so the checksum in `SHA256SUMS` is
  reproducible and not just descriptive

中文说明：本目录是 SM75 优化版在**当前 main 基线**上的重新移植版（内容与之前的
opt3+opt4 一致，基线更新了 1235 个 commit），已在 8x 2080 Ti 上做过 A/B 验证。
详见 `RELEASE_NOTES.md`。

## What it is

DeepSeek-V4-Flash is an FP8/FP4 checkpoint. Turing (SM75) and Ampere (SM80) have
neither FP8 nor FP4 tensor cores, so upstream refuses to load it below sm80.
This release is a community-maintained path that makes it run, and fast:

- **Expert GEMMs.** The routed MXFP4 experts are dequantized in registers by a
  hand-written PTX `mma.sync` kernel (6.6x the Triton version), with a
  lane-major repack done at load time (a further 1.35-1.59x, memory-neutral
  because it permutes the same bytes), an n-tiles-per-warp variant that loads
  the `m16n8k8` activation fragments once per k step, and a k-split that
  doubles the block count at bs=1 where the kernel is occupancy-starved.
- **Indexer.** The DSA/DSV4 indexer logits run as a key-chunked cuBLAS fp16
  GEMM plus epilogue. The Triton `tl.dot` version was measured at ~1.5
  TFLOP-equivalent on SM75 because this Triton build lowers every fp16 dot to
  scalar FFMA.
- **Sparse attention.** A head-shared kernel that amortizes each KV gather over
  16 query heads, with the topk loop split across `grid.z`, plus a batch-chunked
  torch gather for long prompts.
- **Everything else that upstream assumes sm80+.** Dense FP8 linears dequantize
  to fp16 at load time (or keep the FP8 payload and dequantize in registers),
  `wo_a` and the compressor's `wkv_gate` follow the model dtype, `mhc_post`,
  the MoE SwiGLU+clamp and the topk combine are fused Triton kernels, decode
  CUDA graphs are captured per KV-length bucket, and the DSA top-k / MoE align
  / sparse-MLA backends fall back to implementations that can load here.

Everything is gated. On sm90+ none of it engages, and the loader gate
(`SGLANG_ALLOW_SUB80_QUANT`) refuses to bypass the capability check unless you
set it.

## Measured on 8x 2080 Ti, TP2 x PP4, 256K context

Same launcher, same machine, old build (opt3+opt4 on 0.5.19) vs this one:

| metric | old (opt3+opt4) | this build | delta |
|---|---|---|---|
| prefill, 8K prompt (tok/s) | 1177 | 1306 | +11.0% |
| prefill, 13K prompt (tok/s) | 1191 | 1297 | +8.9% |
| decode bs=1, 2.9K ctx (tok/s) | 22.2 | 22.1 | -0.6% |
| decode bs=1, 44K ctx (tok/s) | 15.9 | 16.8 | +5.5% |
| decode bs=1, 118K ctx (tok/s) | 11.8 | 12.9 | +9.9% |
| decode bs=1, 235K ctx (tok/s) | 6.3 | 7.0 | +10.4% |

Run-to-run spread on this box is ~5-7% (clocks drift across a session), so the
prefill and long-context decode gains are real and the short-context decode
difference is not. See `RELEASE_NOTES.md` for the method and the raw numbers.

## Install

See `INSTALL.md`. Short version, into a Python 3.12 venv that already runs this
model on sm75:

```bash
pip install --no-deps --force-reinstall sglang-0.5.21.dev797+ge0f76063c.sm75main1-py3-none-any.whl
pip install "sglang-kernel==0.4.7"        # current main requires >= 0.4.7
```

## Known limits (unchanged from opt3/opt4)

- Accuracy and performance are **not** upstream-verified. The loader prints a
  warning saying so every time you bypass the capability gate.
- `--chunked-prefill-size` must stay <= 512 on this box; the sub-90 sparse-MLA
  prefill materialises `[tokens, 128 heads, 512]` fp32 per layer.
- `--mem-fraction-static` must stay >= 0.969: the weights are 96.8% of a 22 GiB
  card.
- `kv-cache-dtype fp8_e4m3` is the default and the suspected contributor to
  long-context degradation on this path; `auto` (bf16) needs roughly 2x the KV
  pool.
- A `<|begin_of_sentence|>` anywhere in the conversation history sends the model
  into unbounded reasoning until `max_tokens` runs out, with empty content. This
  is the checkpoint's behaviour, not this build's: it reproduces identically on
  opt3+opt4. (Which is why the encoder does not prepend BOS to a fresh
  conversation, and why `SGLANG_DEFAULT_SAMPLING_PARAMS` exists.)
- The agent-doc files under `sglang/multimodal_gen/.claude/skills` are missing
  from this wheel (setuptools' file finder and that symlink do not get along);
  the earlier releases contain them. Nothing at runtime reads them.
