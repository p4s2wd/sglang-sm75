"""FP8 MQA logits for the DSA indexer on GPUs without FP8 tensor cores.

Replaces ``deep_gemm.fp8_mqa_logits`` on sub-90 (SM75/SM80) with the DeepGEMM
semantics:

    logits[q, j] = sum_h weights[q, h] * relu(dot(q[q, h], k[j])) * k_scale[j]
                   for ks[q] <= j < ke[q], else 0.

The Triton ``tl.dot`` path was measured at ~1.5 TFLOP-equivalent on SM75:
this Triton build lowers every fp16 dot (even a vanilla matmul) to scalar
FFMA instead of mma.sync, and a cross-warp head reduce inside the key loop
adds a serialized shared-memory round-trip on top. cuBLAS fp16 tensor-core
GEMMs are ~30x faster, so the fallback is a key-chunked GEMM + epilogue:

  1. dequantize q/k to fp16 once (fp8->fp16 elementwise, negligible),
  2. per key chunk: s = q2d @ k_chunk^T on tensor cores ([Q*H, C]),
  3. relu, then a strided-batched fp16 GEMM contracts the head axis against
     weights ([Q, 1, H] x [Q, H, C]),
  4. scale by k_scale and mask to [ks, ke) (outside the range reads as 0,
     matching the reference the topk transform consumes).

Chunking bounds the [Q*H, C] score tile; at C=1024 it is Q*2 KB fp16.

fp16 range: both GEMMs above land in fp16, so the operands have to be
pre-scaled. The indexer quantizes every q and k row with absmax scaling
(``scale = absmax / FP8_E4M3_MAX`` in the fused rope+hadamard+quant kernel),
so a dequantized row reaches 448 and a D=128 dot of two such rows reaches
``D * 448 * 448 = 2.6e7`` -- fp16 tops out at 65504, and an unscaled GEMM
saturates every real logit tile to inf, which turns the top-k into an
arbitrary prefix of the page range. ``_plan_scales`` divides a power of two
out first (exact in fp16, and relu is positively homogeneous so it commutes
with the scale) and the inverse goes back in fp32. Accuracy left over is the
fp16 pipeline's ~5e-4 relative, which is better than the bf16 torch fallback
upstream uses on the same hardware.
"""

import math

import torch

_CHUNK = 1024

# Comfortably below fp16 max (65504). The scales are exact powers of two, so
# this only has to cover the slack left by the log2 rounding in _pow2_floor.
_FP16_HEADROOM = 30000.0


def _pow2_floor(values: float) -> float:
    """Largest power of two <= ``values`` (positive)."""
    return math.ldexp(1.0, math.floor(math.log2(values)) - 1)


def _plan_scales(
    q_dtype: torch.dtype,
    D: int,
    H: int,
    weights: torch.Tensor,
) -> tuple[float, torch.Tensor, torch.Tensor]:
    """Pick the fp16 range plan for the two GEMMs.

    Returns ``(q_scale, w_scale, inv)``. ``q_scale`` is a Python float applied
    to q before the dot GEMM; ``w_scale`` is a 0-dim tensor folded into the
    weights before the head contraction; ``inv`` undoes both in fp32.
    """
    # The dot bound only needs the fp8 dynamic range, not the data: both
    # operands are absmax-quantized, so |q| , |k| <= FP8_E4M3_MAX always. A
    # compile-time bound is both cheaper than a reduction and strictly safer
    # than a data-dependent one.
    fp8_max = float(torch.finfo(q_dtype).max)
    q_scale = _pow2_floor(_FP16_HEADROOM / (D * fp8_max * fp8_max))

    # The head contraction is a sum of H weighted terms, so it needs its own
    # budget: H * w_absmax * w_scale * dot must fit, and dot is bounded by the
    # headroom above. Capped at 1 -- with small weights the headroom is
    # already there and scaling up could overflow w * w_scale on its own.
    w_absmax = weights.abs().amax().float().clamp(min=torch.finfo(torch.float32).tiny)
    w_scale = torch.exp2(
        torch.clamp(torch.floor(torch.log2(1.0 / (H * w_absmax))) - 1.0, max=0.0)
    )

    # The fp16 result carries one factor of q_scale (scaling q scales the dot
    # linearly) and one of w_scale.
    inv = 1.0 / (q_scale * w_scale)
    return q_scale, w_scale, inv


def fp16_mqa_logits_triton(
    q_fp8: torch.Tensor,  # [Q, H, D] float8_e4m3fn
    k_fp8: torch.Tensor,  # [K, D] float8_e4m3fn
    k_scale: torch.Tensor,  # [K] float32
    weights: torch.Tensor,  # [Q, H] float32
    ks: torch.Tensor,  # [Q] int32
    ke: torch.Tensor,  # [Q] int32
    out: torch.Tensor,  # [Q, N] float32
    chunk: int = _CHUNK,
) -> torch.Tensor:
    Q, H, D = q_fp8.shape
    N = out.shape[1]
    K = k_fp8.shape[0]
    k16 = k_fp8.to(torch.float16)
    q16 = q_fp8.reshape(Q, H, D).to(torch.float16)

    q_scale, w_scale, inv = _plan_scales(q_fp8.dtype, D, H, weights)
    if q_scale != 1.0:
        q16 = (q16 * q_scale).reshape(Q * H, D)
    w16 = (weights * w_scale).to(torch.float16).unsqueeze(1)  # [Q, 1, H]
    inv = inv.to(torch.float32)
    ks_i = ks.to(torch.int64).unsqueeze(1)
    ke_i = ke.to(torch.int64).unsqueeze(1)

    # k_fp8 holds the gathered compressed keys and can be shorter than N,
    # which is page-rounded (max_seqlen_k); only the real keys carry logits.
    n_scored = min(N, K)
    for c0 in range(0, n_scored, chunk):
        c1 = min(c0 + chunk, n_scored)
        s = torch.mm(q16, k16[c0:c1].t())  # [Q*H, C] fp16, tensor cores
        s.relu_()
        acc = torch.bmm(w16, s.view(Q, H, c1 - c0)).float()  # [Q, 1, C]
        acc = acc.squeeze(1) * inv * k_scale[c0:c1].unsqueeze(0)
        cols = torch.arange(c0, c1, device=out.device, dtype=torch.int64)
        acc.masked_fill_((cols < ks_i) | (cols >= ke_i), 0.0)
        out[:, c0:c1] = acc
    if n_scored < N:
        # The tail is page padding with no key behind it. Callers allocate the
        # logits with new_empty, so leaving it undefined would hand the top-k
        # whatever the allocator last left in that block.
        out[:, n_scored:].zero_()
    return out
