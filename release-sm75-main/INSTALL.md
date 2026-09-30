# Install: sglang SM75 wheel on current main

Target: a Python 3.12 venv on 8x RTX 2080 Ti (SM75, 22 GiB each) that already
runs DeepSeek-V4-Flash through the sm75 path. `prod-constraints.txt` is the
`pip freeze` of the venv this release was verified on.

中文说明：本文档面向已经能跑 SM75 版 DeepSeek-V4-Flash 的 Python 3.12 venv。
`prod-constraints.txt` 是验证环境的环境快照。

## 1. Hard requirements

These are not negotiable on this hardware; the launcher checks them and refuses
to start.

| requirement | why |
|---|---|
| 8x GPU, compute capability 7.5 | TP2 x PP4 with NVLink inside the pairs (0,1)(2,3)(4,5)(6,7) |
| `SGLANG_ALLOW_SUB80_QUANT=1` | the loader otherwise refuses an FP8 checkpoint below sm80 |
| `nvcc` on PATH (CUDA 12.9) | the PTX W4A16 expert kernels are JIT-compiled at load; on this box only cuda-12.9 actually ships one |
| `ninja` on PATH | same; without it the JIT helper returns `None` silently and you fall back to the ~6.6x slower Triton kernel |
| `--mem-fraction-static` >= 0.969 | weights are 96.8% of a 22 GiB card |
| `--chunked-prefill-size` <= 512 | the sub-90 sparse-MLA prefill materialises `[tokens, 128 heads, 512]` fp32 per layer; 2048 OOMs a PP3 card during warmup |
| 150 W power limit | higher risks GPU6/7 hangs |

## 2. Install

```bash
VENV=/data/nvme/sglang/.venv
"$VENV/bin/pip" install --no-deps --force-reinstall \
    sglang-0.5.21.dev797+ge0f76063c.sm75main1-py3-none-any.whl
"$VENV/bin/pip" install "sglang-kernel==0.4.7"
```

`--no-deps` on purpose: the wheel's dependency set is upstream's, and changing
it on a box that is already working tends to move torch or triton underneath
you. Install the dependencies separately if you are building the venv from
scratch:

```bash
"$VENV/bin/pip" install -r prod-constraints.txt
```

`sglang-kernel` 0.4.7 is a hard floor on current main: the engine asserts it at
startup (`assert_pkg_version("sglang-kernel", "0.4.7", ...)`). The 0.4.6.post1
that the opt3/opt4 releases used is rejected.

The wheel is pure python (`py3-none-any`), so it does not care about your
Python version. It does not contain the Rust extensions; those ship in
`sglang-kernel` and `sgl-router`.

## 3. Check before serving

```bash
./launch-dsv4-sm75.sh --check
```

It verifies: python and sglang import, the three PTX W4A16 kernel sources are
present under `KERNEL_PATH` (without them no expert kernel would compile),
torch sees 8 GPUs at capability 7.5, nvcc and ninja exist, the checkpoint is
there, no GPU is still held by a previous server, and the power limit. It exits
non-zero on any of those, and the launcher refuses to start if it fails.

## 4. Serve

```bash
./launch-dsv4-sm75.sh              # background, log to $LOG_DIR/serve-prod.log
tail -f logs/serve-prod.log        # ready when it says "fired up and ready to roll"
./launch-dsv4-sm75.sh --stop       # stop and wait for the GPU memory
```

Every setting in the launcher is an environment variable override, so an A/B
does not need the file edited: `MEM_FRACTION=0.97 CHUNK=512 MAXREQ=16
./launch-dsv4-sm75.sh`. The A/B switches for the individual kernels are
`SGLANG_SM75_*` and `SGLANG_DSV4_*`; `SGLANG_DSV4_DECODE_SEQ_LEN_BUCKETS=`
(one graph per batch size) and `SGLANG_OPT_USE_SM75_C4_TOPK=0` are the two with
the largest effect.

## 5. Rebuilding the wheel

```bash
CHECKOUT=/data/nvme/sglang-codex/sglang PYTHON=/data/nvme/sglang/.venv/bin/python ./build-wheel.sh
```

Needs `setuptools`, `setuptools-scm`, `setuptools-rust` and `wheel` in
`$PYTHON`, and it builds with `SGLANG_BUILD_RUST_EXTS=none` (pure python). The
script comments explain both workarounds. It builds in a detached git worktree
rather than from `git archive`, because without git metadata setuptools-scm's
file finder drops 56 package-data entries (dotfiles, the agent docs), and it
normalises the zip timestamps afterwards because `wheel` 0.48 ignores
`SOURCE_DATE_EPOCH`. Verified: two runs produce the same sha256, and the wheel
is byte-identical to the tree that was benchmarked. To rebuild from the patches instead
of a checkout, apply `sm75-main-series.patch` (or
`sm75-optimizations.patch`) to upstream `main` at `98fce73d5b` first; both
apply cleanly to that commit.
