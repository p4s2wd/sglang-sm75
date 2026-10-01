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

# Tile width for the [Q*H, C] score tile in fp16_mqa_logits_triton. At 1024
# the tile is 64 MiB for a 512-token prefill chunk, and that allocation is what
# OOMs PP0 when prefilling a 256K context -- the stage has under 0.5 GB free
# and came up ~57 MiB short. The mm reads the same values at any tile width, so
# this is purely a memory/speed knob with no numerical effect.
_CHUNK = 256

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
    if Q <= 16:
        from sglang.srt.environ import envs

        if envs.SGLANG_SM75_FUSE_INDEXER_LOGITS.get():
            return mqa_logits_smallq(q_fp8, k_fp8, k_scale, weights, ks, ke, out)
    N = out.shape[1]
    K = k_fp8.shape[0]
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
        # Convert per chunk rather than promoting the whole gathered key set
        # once. k_fp8 is [K, D] and K grows with the context, so the up-front
        # fp16 copy is O(K*D) of extra device memory -- at a 260K context it
        # competes with the KV pool for the same card and this call is what
        # OOMs on PP0. The loop already walked the keys in chunks for the mm,
        # so promoting inside the loop costs one extra pass over each chunk and
        # drops the peak from O(K*D) to O(chunk*D). Elementwise, so the values
        # fed to torch.mm are bit-identical to the whole-tensor version.
        k16 = k_fp8[c0:c1].to(torch.float16)
        s = torch.mm(q16, k16.t())  # [Q*H, C] fp16, tensor cores
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


import triton
import triton.language as tl

from sglang.kernels.ops.quantization.fp8_w8a16 import _e4m3_to_f16, fp8_payload_lut






@triton.jit
def _mqa_logits_smallq_kernel(
    q_ptr,  # [Q, H, D] uint8-viewed float8_e4m3fn
    k_ptr,  # [K, D] uint8-viewed float8_e4m3fn
    ks_ptr,  # [K] float32 per-key scales
    w_ptr,  # [Q, H] float32
    lo_ptr,  # [Q] int32 first scored column
    hi_ptr,  # [Q] int32 end of scored range
    out_ptr,  # [Q, N] float32
    k_n,
    n_cols,
    stride_q0,
    stride_qh,
    stride_wq,
    stride_on,
    H: tl.constexpr,
    D: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    """logits[q, t] = k_scale[t] * sum_h w[q,h] * relu(q[q,h] . k[t]) on [lo, hi).

    The sub-90 stand-in for deep_gemm's fp8_paged_mqa_logits: the torch chain
    it replaces (fp8->fp16 copies of q and k, an fp16 mm, an fp16 head bmm and
    a masked_fill -- ~25 launches and ~1 ms of eager wall per indexer layer at
    decode) collapses to this one launch. One q row per program: at decode
    q_n is 1, and owning a single [D] row of q per head turns every dot into a
    broadcasted [BLOCK_T, D] product, which keeps the whole kernel off the
    tl.dot operand-staging path that costs more than the math on sm75.
    Products are fp16 (the precision of the torch mm this replaces), the head
    sum is fp32.
    """
    pid_t = tl.program_id(0)
    q = tl.program_id(1)
    offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    offs_d = tl.arange(0, D)
    t_ld = tl.minimum(offs_t, k_n - 1)

    kt = _e4m3_to_f16(tl.load(k_ptr + t_ld[:, None] * D + offs_d[None, :]))
    acc = tl.zeros((BLOCK_T,), dtype=tl.float32)
    for h in range(H):
        qh = _e4m3_to_f16(tl.load(q_ptr + q * stride_q0 + h * stride_qh + offs_d))
        s = tl.sum((kt * qh[None, :]).to(tl.float32), axis=1)
        wh = tl.load(w_ptr + q * stride_wq + h)
        acc += wh * tl.maximum(s, 0.0)

    lo = tl.load(lo_ptr + q)
    hi = tl.load(hi_ptr + q)
    keep = (offs_t >= lo) & (offs_t < hi) & (offs_t < k_n)
    scale = tl.load(ks_ptr + t_ld)
    out = tl.where(keep, acc * scale, 0.0)
    tl.store(out_ptr + q * stride_on + offs_t, out, mask=offs_t < n_cols)


def mqa_logits_smallq(
    q_fp8: torch.Tensor,
    k_fp8: torch.Tensor,
    k_scale: torch.Tensor,
    weights: torch.Tensor,
    ks: torch.Tensor,
    ke: torch.Tensor,
    out: torch.Tensor,
    block_t: int = 32,
) -> torch.Tensor:
    """Decode-shape (Q <= 16) path; see _mqa_logits_smallq_kernel.

    Same arguments as fp16_mqa_logits_triton.
    """
    Q, H, D = q_fp8.shape
    if Q > 16:
        raise ValueError("mqa_logits_smallq serves Q <= 16")
    K = k_fp8.shape[0]
    N = out.shape[1]
    q_b = q_fp8.view(torch.uint8) if q_fp8.dtype != torch.uint8 else q_fp8
    k_b = k_fp8.view(torch.uint8) if k_fp8.dtype != torch.uint8 else k_fp8
    _mqa_logits_smallq_kernel[(triton.cdiv(N, block_t), Q)](
        q_b,
        k_b,
        k_scale,
        weights,
        ks,
        ke,
        out,
        K,
        N,
        q_fp8.stride(0),
        q_fp8.stride(1),
        weights.stride(0),
        out.stride(0),
        H=H,
        D=D,
        BLOCK_T=block_t,
        num_warps=4,
        num_stages=1,
    )
    return out


@triton.jit
def _mqa_paged_smallq_kernel(
    q_ptr,  # [B, H, D] uint8-viewed float8_e4m3fn
    kvc_ptr,  # [num_pages, 64 * (D + 4)] uint8 page rows: D payload + D/32 fp32... see wrapper
    pt_ptr,  # [B, max_pages] int64 page table, -1 = hole
    seq_ptr,  # [B] sequence lengths
    w_ptr,  # [B, H] float32
    out_ptr,  # [B, n_out] float32
    max_pages,
    n_out,
    stride_qb,
    stride_qh,
    stride_pb,
    stride_wb,
    stride_ob,
    PAGE: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    """Paged variant of _mqa_logits_smallq_kernel for the indexer caches.

    Page rows hold [PAGE * D] payload bytes followed by PAGE fp32 per-key
    scales, so a strip's gather is page_table-driven. Holes (page -1) and
    columns past seq_lens store zero, matching the torch chain's clamp-then-
    mask. One q row per program (batch axis of the grid); the same
    dot-free formulation as the non-paged kernel.
    """
    pid_t = tl.program_id(0)
    b = tl.program_id(1)
    offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    offs_d = tl.arange(0, D)
    page = offs_t // PAGE
    row = offs_t % PAGE
    in_t = offs_t < n_out
    pidx = tl.minimum(page, max_pages - 1)
    phys = tl.load(pt_ptr + b * stride_pb + pidx)
    valid = (page < max_pages) & (phys >= 0)
    phys_s = tl.maximum(phys, 0)

    byte_base = kvc_ptr + phys_s[:, None] * (PAGE * (D + 4)) + row[:, None] * D
    kt = _e4m3_to_f16(tl.load(byte_base + offs_d[None, :]))
    acc = tl.zeros((BLOCK_T,), dtype=tl.float32)
    for h in range(H):
        qh = _e4m3_to_f16(tl.load(q_ptr + b * stride_qb + h * stride_qh + offs_d))
        s = tl.sum((kt * qh[None, :]).to(tl.float32), axis=1)
        wh = tl.load(w_ptr + b * stride_wb + h)
        acc += wh * tl.maximum(s, 0.0)

    scale = tl.load(
        kvc_ptr.to(tl.pointer_type(tl.float32))
        + phys_s * (PAGE * (D + 4)) // 4
        + PAGE * D // 4
        + row,
    )
    seq = tl.load(seq_ptr + b)
    keep = valid & (offs_t < seq)
    tl.store(
        out_ptr + b * stride_ob + offs_t,
        tl.where(keep, acc * scale, 0.0),
        mask=in_t,
    )


def mqa_paged_smallq(
    q_fp8: torch.Tensor,  # [B, 1, H, D] fp8
    kvcache_fp8: torch.Tensor,  # [pages, PAGE, 1, D+4] uint8-viewed
    weight: torch.Tensor,  # [B, H] f32
    seq_lens: torch.Tensor,  # [B]
    page_table: torch.Tensor,  # [B, max_pages] int64
    max_seq_len: int,
    out: torch.Tensor,  # [B, max_seq_len]
) -> torch.Tensor:
    """Paged decode-shape indexer logits; see _mqa_paged_smallq_kernel."""
    bsz, _, H, D = q_fp8.shape
    if bsz > 16:
        raise ValueError("mqa_paged_smallq serves B <= 16")
    page = kvcache_fp8.shape[1]
    kvc = kvcache_fp8.view(-1, page * (D + 4))
    kvc_b = kvc if kvc.dtype == torch.uint8 else kvc.view(torch.uint8)
    q_b = (
        q_fp8[:, 0].view(torch.uint8)
        if q_fp8.dtype != torch.uint8
        else q_fp8[:, 0]
    )
    _mqa_paged_smallq_kernel[(triton.cdiv(max_seq_len, 32), bsz)](
        q_b,
        kvc_b,
        page_table,
        seq_lens,
        weight,
        out,
        page_table.shape[1],
        max_seq_len,
        q_fp8.stride(0),
        q_fp8.stride(2),
        page_table.stride(0),
        weight.stride(0),
        out.stride(0),
        PAGE=page,
        H=H,
        D=D,
        BLOCK_T=32,
        num_warps=4,
        num_stages=1,
    )
    return out
