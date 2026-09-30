#!/usr/bin/env bash
#
# DeepSeek-V4-Flash on 8x RTX 2080 Ti (SM75) -- production launcher.
#
# Self-contained: needs only a Python environment with the sm75 sglang wheel
# installed. It does not read a sglang checkout, so it works the same from a
# wheel install and from an editable tree.
#
#   ./launch-dsv4-sm75.sh                 # start in background, log to $LOG_DIR
#   ./launch-dsv4-sm75.sh --foreground    # start in this terminal
#   ./launch-dsv4-sm75.sh --check         # environment check only
#   ./launch-dsv4-sm75.sh --stop          # stop a background instance
#
# Point it at your own environment/checkpoint with, for example:
#   VENV=/opt/venv MODEL=/models/DeepSeek-V4-Flash ./launch-dsv4-sm75.sh
#
# 中文说明:本脚本在 8x RTX 2080 Ti(SM75)上以 TP2xPP4 拉起 DeepSeek-V4-Flash。
#   ./launch-dsv4-sm75.sh            后台启动,日志写入 $LOG_DIR
#   ./launch-dsv4-sm75.sh --foreground  前台启动
#   ./launch-dsv4-sm75.sh --check    仅做环境自检(GPU/nvcc/ninja/权重/显存),不启动
#   ./launch-dsv4-sm75.sh --stop     停止后台实例并等待显存释放
# 所有配置都可用环境变量覆盖,例如:
#   VENV=/opt/venv MODEL=/models/DeepSeek-V4-Flash ./launch-dsv4-sm75.sh
# 注意:--chunked-prefill-size 必须 <= 512(超过会在 warmup 阶段 OOM),
#       MEM_FRACTION 不要低于 0.969(权重已占单卡约 97%)。
# 完整文档见 INSTALL.zh-CN.md / INSTALL.md。
#
# Every setting is an env override; the defaults are the values measured and
# frozen for this machine (see RELEASE-NOTES-sm75-dsv4-flash.md).
#
set -uo pipefail

VENV="${VENV:-/data/nvme/sglang/.venv}"
if [ -f "$VENV/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
fi

# --------------------------------------------------------------------------
# Settings (override by exporting the variable before calling)
# --------------------------------------------------------------------------
MODEL="${MODEL:-/data/nvme/models/DeepSeek/DeepSeek-V4-Flash-Vision-Exp}"
PYTHON="${PYTHON:-$(command -v python3)}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8200}"

# Parallelism. NVLink is only wired inside the pairs (0,1)(2,3)(4,5)(6,7), so
# TP2 lands in one NVLink pair and TP2xPP4 is the communication-optimal split.
TP="${TP:-2}"
PP="${PP:-4}"
PP_ASYNC="${PP_ASYNC:-2}"

# Memory. Weights occupy 96.83% of a 22 GiB card, so the static fraction must
# stay above 0.969 or the KV pool gets nothing. --max-total-tokens caps the pool
# explicitly: at 256K context an uncapped pool leaves no room for prefill
# activations (measured: 51 MiB free, prefill OOMs).
MEM_FRACTION="${MEM_FRACTION:-0.97}"
MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-270000}"
# KV dtype is overridable so the long-context A/B can switch it without editing
# this file. Default is unchanged (fp8_e4m3). NOTE: fp8_e4m3 is the suspected
# contributor to long-context degradation on this sm75 sub-80 path -- see
# PREFILL_PROFILE.md; `auto` means bf16 and needs roughly 2x the KV pool.
KV_DTYPE="${KV_DTYPE:-fp8_e4m3}"
CTX="${CTX:-262144}"
# Do not raise above 512 on this box: the sub-90 sparse-MLA path runs prefill
# through the triton decode kernel, and _merge_partial_attn materializes
# [tokens, 128 heads, 512] fp32 temporaries -- at 2048 tokens that is >1 GiB
# per layer and OOMs a PP3 card during warmup (measured 2026-09-25).
CHUNK="${CHUNK:-512}"

# Throughput. --max-running-requests also sets pp_max_micro_batch_size
# (= MAXREQ / PP), i.e. how full the pipeline can get.
MAXREQ="${MAXREQ:-8}"
GRAPH_MAX_BS="${GRAPH_MAX_BS:-2}"
GRAPH_BS="${GRAPH_BS:-1 2}"

# JIT compilation of the hand-written PTX W4A16 expert kernels needs nvcc.
# On this box only cuda-12.9 actually ships nvcc (cuda-12.6 is a symlink to
# 12.8 with no nvcc), and the wrong nvcc fails with
# "nvcc fatal: Unknown option '--compress-mode=size'".
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.9}"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

# The one mandatory switch: without it the loader refuses an FP8 checkpoint on
# a sub-SM80 GPU.
export SGLANG_ALLOW_SUB80_QUANT=1

# Sparse-MLA decode backend. The fused Triton path measured faster than the
# torch fallback end to end (decode +11%, prefill +47%); the wheel defaults to
# it on SM75, this just makes the choice explicit and overridable.
# Default sampling for clients that send none is read from the checkpoint's
# generation_config.json. DeepSeek-V4-Flash ships the generic placeholder
# do_sample=true, temperature=1.0, top_p=1.0, and unrestricted sampling is
# unusable for agentic tool calls: 2 of 9 long (4-6k token) HTML/JS generations
# at that default had a stray CJK character wedged into a numeric literal
# ("dir += 0.5<CJK>;"), which is both a syntax error and an unparseable tool
# call. temperature: 0 in generation_config.json fixes it. If you would rather
# not edit the checkpoint, set this instead (it wins over generation_config.json
# and still loses to an explicit per-request temperature):
#   export SGLANG_DEFAULT_SAMPLING_PARAMS='{"temperature": 0.6, "top_p": 0.95}'
# See _sampling_params_override_from_env in srt/entrypoints/openai/serving_chat.py.
export SGLANG_SM120_FLASHMLA_BACKEND="${SGLANG_SM120_FLASHMLA_BACKEND:-triton}"
export SGLANG_DEFAULT_THINKING="${SGLANG_DEFAULT_THINKING:-1}"
export SGLANG_OPT_USE_SM75_C4_TOPK="${SGLANG_OPT_USE_SM75_C4_TOPK:-1}"
export SGLANG_PP_EARLY_PROXY_SEND="${SGLANG_PP_EARLY_PROXY_SEND:-0}"
export SGLANG_OPT_W8A16_WIDE_M1_K1024="${SGLANG_OPT_W8A16_WIDE_M1_K1024:-0}"
export SGLANG_DSV4_DECODE_SEQ_LEN_BUCKETS="${SGLANG_DSV4_DECODE_SEQ_LEN_BUCKETS:-4096,8192,16384,32768,65536,131072}"

# Prefill indexer logits: the default gate (8192 query tokens) disables the
# fused Triton fp16 MQA-logits path for every chunked prefill on this box
# (chunk < 8192), leaving the paged torch fallback, which gathers + dequantizes
# every key per chunk through ~8 aten kernels per page-chunk. Lowering the gate
# to the chunk size routes single-request prefill through the Triton kernel.
export SGLANG_OPT_DSV4_NONPAGED_INDEXER="${SGLANG_OPT_DSV4_NONPAGED_INDEXER:-1}"
export SGLANG_OPT_DSV4_NONPAGED_INDEXER_MIN_QUERY_TOKENS="${SGLANG_OPT_DSV4_NONPAGED_INDEXER_MIN_QUERY_TOKENS:-64}"

LOG_DIR="${LOG_DIR:-$PWD/logs}"
LOG_NAME="${LOG_NAME:-serve-prod}"

usage() { grep -E "^# " "$0" | sed 's/^# \{0,1\}//'; exit 0; }

MODE="background"
for a in "$@"; do
  case "$a" in
    --foreground|-f) MODE="foreground" ;;
    --check|-c) MODE="check" ;;
    --stop) MODE="stop" ;;
    --help|-h) usage ;;
    *) echo "unknown argument: $a (try --help)" >&2; exit 2 ;;
  esac
done

# --------------------------------------------------------------------------
# Environment check
# --------------------------------------------------------------------------
check() {
  local fail=0
  say() { printf '%-34s %s\n' "$1" "$2"; }
  bad() { printf '%-34s %s  <== FIX THIS\n' "$1" "$2"; fail=1; }

  echo "== environment =="
  if [ -x "$PYTHON" ]; then say "python" "$PYTHON ($($PYTHON -V 2>&1))"
  else bad "python" "not found at $PYTHON"; fi

  "$PYTHON" -c "import importlib.metadata as m; print(m.version('sglang'))" >/dev/null 2>&1
  if [ $? -eq 0 ]; then
    say "sglang version" "$("$PYTHON" -c "import importlib.metadata as m;print(m.version('sglang'))")"
  else
    bad "sglang version" "not installed -- pip install the sm75 wheel"
  fi

  "$PYTHON" -c "import sglang.srt.layers.quantization.fp8" >/dev/null 2>&1
  if [ $? -eq 0 ]; then say "sub-90 wiring import" "ok"
  else bad "sub-90 wiring import" "sglang.srt.layers.quantization.fp8 failed to import"; fi

  local jit
  jit="$("$PYTHON" - <<'PY' 2>/dev/null
import pathlib, sys
try:
    from sglang.kernels.jit.utils.compile.paths import KERNEL_PATH
except Exception as exc:
    print(f"import-failed: {exc}"); sys.exit(1)
need = ["moe/mxfp4_w4a16_ptx_v3.cuh", "moe/mxfp4_w4a16_ptx.cuh",
        "moe/mxfp4_w4a16_ptx_direct.cuh"]
missing = [n for n in need if not (pathlib.Path(KERNEL_PATH) / "csrc" / n).exists()]
if missing:
    print("missing: " + ", ".join(missing)); sys.exit(1)
print(KERNEL_PATH)
PY
)"
  if [ -n "$jit" ]; then say "JIT kernel sources" "$jit"
  else bad "JIT kernel sources" "not in the install -- no PTX W4A16 kernel would compile"; fi

  "$PYTHON" - <<'PY' 2>/dev/null
import torch
print(f"{torch.__version__} cuda={torch.version.cuda} gpus={torch.cuda.device_count()}")
PY
  if [ $? -eq 0 ]; then
    say "torch" "$("$PYTHON" -c 'import torch;print(f"{torch.__version__} cuda={torch.version.cuda} gpus={torch.cuda.device_count()}")' 2>/dev/null)"
  else
    bad "torch" "not importable"
  fi

  local ngpu
  ngpu="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l)"
  if [ "${ngpu:-0}" -ge 8 ]; then say "GPUs visible" "$ngpu"
  else bad "GPUs visible" "${ngpu:-0} (need 8)"; fi

  local cap
  cap="$("$PYTHON" -c 'import torch;print(".".join(map(str,torch.cuda.get_device_capability(0))))' 2>/dev/null)"
  if [ "$cap" = "7.5" ]; then say "compute capability" "$cap (SM75, supported)"
  else bad "compute capability" "${cap:-?} (this build targets 7.5)"; fi

  if [ -x "$CUDA_HOME/bin/nvcc" ]; then
    say "nvcc" "$("$CUDA_HOME/bin/nvcc" --version | tail -1 | sed 's/^.*Build /Build /')"
  else
    bad "nvcc" "no $CUDA_HOME/bin/nvcc -- the PTX W4A16 kernels cannot compile"
  fi

  if command -v ninja >/dev/null 2>&1; then say "ninja" "$(command -v ninja)"
  else bad "ninja" "missing -- JIT returns None silently without it"; fi

  if [ -d "$MODEL" ]; then say "checkpoint" "$MODEL"
  else bad "checkpoint" "not a directory: $MODEL"; fi

  local free
  free="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)"
  if [ "${free:-9999}" -lt 1000 ]; then say "GPU memory free" "max used ${free} MiB"
  else bad "GPU memory free" "a GPU holds ${free} MiB -- another server is running (--stop it)"; fi

  local lim
  lim="$(nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits | head -1)"
  say "power limit" "${lim} W (150 W is the validated setting; higher risks GPU6/7 hangs)"

  echo
  [ "$fail" = 0 ] && echo "check: PASS" || echo "check: FAIL"
  return "$fail"
}

stop() {
  pkill -9 -f "launch_server" 2>/dev/null || true
  pkill -9 -f "sglang serve" 2>/dev/null || true
  echo "signalled; waiting for GPU memory to be released"
  for _ in $(seq 1 15); do
    local used
    used="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)"
    [ "${used:-1}" = 0 ] && { echo "all GPU processes gone"; return 0; }
    sleep 10
  done
  echo "WARNING: GPU processes still present; memory may not have been released."
  echo "If nvidia-smi shows leaked memory and dmesg says NVRM nvAssertFailedNoLog,"
  echo "the only fix is a machine reboot."
  return 1
}

# --------------------------------------------------------------------------
# Launch
# --------------------------------------------------------------------------
build_args() {
  ARGS=(
    --model "$MODEL"
    --served-model-name "deepseek-v4-flash"
    --tp-size "$TP" --pp-size "$PP"
    --pp-async-batch-depth "$PP_ASYNC"
    --mem-fraction-static "$MEM_FRACTION"
    --kv-cache-dtype "$KV_DTYPE"
    --context-length "$CTX"
    --chunked-prefill-size "$CHUNK"
    --max-total-tokens "$MAX_TOTAL_TOKENS"
    --max-running-requests "$MAXREQ"
    --sleep-on-idle
    --cuda-graph-backend-decode full
    --cuda-graph-max-bs-decode "$GRAPH_MAX_BS"
    --cuda-graph-bs-decode $GRAPH_BS
    --cuda-graph-backend-prefill disabled
    --reasoning-parser deepseek-v4
    --tool-call-parser deepseekv4
    --default-chat-template-kwargs '{"thinking": true}'
    --host "$HOST" --port "$PORT"
  )
}

main() {
  case "$MODE" in
    check) check; exit $? ;;
    stop) stop; exit $? ;;
  esac

  echo "== pre-flight =="
  if ! check; then
    echo
    echo "refusing to start; fix the items above (or run: $0 --check)" >&2
    exit 1
  fi

  build_args
  mkdir -p "$LOG_DIR"
  local log="$LOG_DIR/$LOG_NAME.log"

  echo
  echo "== starting =="
  echo "model   : $MODEL"
  echo "parallel: TP$TP x PP$PP, ctx $CTX, kv pool $MAX_TOTAL_TOKENS, kv dtype $KV_DTYPE"
  echo "http    : http://$HOST:$PORT"
  echo "log     : $log"
  echo "ready   : grep 'fired up and ready to roll' $log"

  if [ "$MODE" = "foreground" ]; then
    exec "$PYTHON" -m sglang.launch_server "${ARGS[@]}" 2>&1 | tee "$log"
  fi

  # setsid is required: without it the server is in the ssh session's process
  # group and dies from SIGHUP when the connection drops -- silently, with
  # nothing in the log.
  setsid nohup sglang serve "${ARGS[@]}" >"$log" 2>&1 </dev/null &
  echo "pid     : $!"
}

main "$@"
