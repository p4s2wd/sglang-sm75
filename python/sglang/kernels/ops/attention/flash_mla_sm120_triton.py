"""SM120-optimized Triton FlashMLA sparse decode kernel — Tiled V2.

Replaces V1's serial token loop with a tiled vectorized approach:
  1. BLOCK_T tokens loaded simultaneously via 2D gather (vs 1-at-a-time)
  2. All BLOCK_T QK scores computed at once via vectorized mul-reduce
  3. V accumulation via vectorized weighted sum across BLOCK_T tokens
  4. Online softmax operates on tile-level maxima (fewer rescales)

Three typed views of the same paged buffer handle FP8/uint8/BF16 regions:
- float8_e4m3fn view -> nope FP8 values (direct load + dequant)
- uint8 view -> UE8M0 scale bytes (raw integer -> exp2 conversion)
- bfloat16 view -> rope BF16 values (direct load)

DSv4 page layout (per token, 576 bytes data + 8 bytes scales):
  Data section: [0:448] FP8 nope | [448:576] BF16 rope (64 values = 128 bytes)
  Scale section: [page_size*576 + offset*8 : +7] UE8M0 scales (7 groups of 64)

Target: RTX PRO 6000 (SM120, 188 SMs, 99KB SMEM, ~1.5 TB/s GDDR7, 96MB L2)
"""

import logging
import os
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.quantization.fp8_w8a16 import fp8_payload_lut
from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

LOG2E = tl.constexpr(1.4426950408889634)

# DSv4 KV cache layout constants
_NOPE_DIM = 448
_ROPE_DIM = 64
_D = _NOPE_DIM + _ROPE_DIM  # 512
_TOKEN_DATA_STRIDE = 576  # bytes per token in data section
_SCALE_STRIDE = 8  # bytes per token in scale section


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_T": 16}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_T": 16}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_T": 32}, num_warps=8, num_stages=2),
    ],
    key=["topk_rounded"],
)
@triton.jit
def _tiled_sparse_decode_kernel(
    # Q: [B, H, D] bf16
    Q_ptr,
    # Paged KV cache — typed views of the same underlying memory
    cache_nope_ptr,  # uint8 flat (1 byte/elem) — e4m3 payload for nope
    # 256-entry e4m3 -> fp32 decode table, indexed by the raw payload byte.
    # SM75/SM80 Triton refuses to *load* an fp8 dtype at all
    # ("type fp8e4nv not supported in this architecture"), so the bytes travel
    # as uint8 and the conversion is a gather from this L1-resident table. Same
    # trick as the dense W8A16 GEMV (fp8_w8a16.py); an integer-shift widening is
    # ~30x slower because it makes SM integer throughput the bottleneck.
    lut_ptr,
    cache_uint8_ptr,  # uint8 flat (1 byte/elem) — for scales
    cache_bf16_ptr,  # bfloat16 flat (2 bytes/elem) — for rope
    # Indices: [B, topk] int32
    indices_ptr,
    # Valid lengths: [B] int32
    topk_len_ptr,
    # Output: [B, H, D] bf16 and LSE: [B, H] float32
    O_ptr,
    LSE_ptr,
    # Scalars
    sm_scale: tl.float32,
    page_size: tl.int32,
    page_bytes: tl.int64,
    scale_section_off: tl.int64,  # page_size * 576
    H: tl.int32,
    topk: tl.int32,
    topk_rounded: tl.int32,  # for autotune key
    has_topk_len: tl.constexpr,
    # Strides
    stride_qb: tl.int32,
    stride_qh: tl.int32,
    stride_ob: tl.int32,
    stride_oh: tl.int32,
    stride_ib: tl.int32,  # indices batch stride
    # Constexprs
    NOPE_PAD: tl.constexpr,  # 512 (padded from 448)
    ROPE_DIM: tl.constexpr,  # 64
    NOPE_DIM_RT: tl.int32,  # 448 (runtime, for masking)
    BLOCK_T: tl.constexpr,  # tokens per tile (16 or 32)
):
    """Tiled sparse decode: vectorized gather + QK + softmax + V accumulation.

    Grid: (B, H) — one block per (batch, head) pair.
    Each block processes all topk tokens in tiles of BLOCK_T.
    """
    bid = tl.program_id(0)
    hid = tl.program_id(1)

    # ---- Load Q for this (batch, head) ----
    q_base = bid * stride_qb + hid * stride_qh
    nope_offs = tl.arange(0, NOPE_PAD)  # [512]
    nope_mask = nope_offs < NOPE_DIM_RT  # [512], True for [0:448]
    rope_offs = tl.arange(0, ROPE_DIM)  # [64]

    q_nope = tl.load(Q_ptr + q_base + nope_offs, mask=nope_mask, other=0.0)
    q_nope = q_nope.to(tl.float32) * sm_scale
    q_rope = tl.load(Q_ptr + q_base + NOPE_DIM_RT + rope_offs)
    q_rope = q_rope.to(tl.float32) * sm_scale

    # ---- Valid token count ----
    valid_topk = topk
    if has_topk_len:
        valid_topk = tl.load(topk_len_ptr + bid).to(tl.int32)
        valid_topk = tl.minimum(valid_topk, topk)

    # ---- Online softmax state (base-2 math for SM120 efficiency) ----
    m_i: tl.float32 = -1e30
    l_i: tl.float32 = 0.0
    acc_nope = tl.zeros([NOPE_PAD], dtype=tl.float32)
    acc_rope = tl.zeros([ROPE_DIM], dtype=tl.float32)

    # ---- Precompute constant index vectors ----
    group_ids = (nope_offs // 64).to(tl.int64)  # [NOPE_PAD], scale group for each dim
    t_offs = tl.arange(0, BLOCK_T)  # [BLOCK_T], token offsets within tile

    # ---- Process tokens in tiles of BLOCK_T ----
    for tile_start in range(0, topk, BLOCK_T):
        t_idx = tile_start + t_offs  # [BLOCK_T], global token indices
        t_in_bounds = t_idx < topk  # bounds for index load
        t_valid = t_idx < valid_topk  # bounds for actual processing

        # Load indices for this tile: [BLOCK_T]
        raw_indices = tl.load(
            indices_ptr + bid * stride_ib + t_idx,
            mask=t_in_bounds,
            other=-1,
        )
        idx_valid = t_valid & (raw_indices >= 0)  # [BLOCK_T] mask

        # Page addressing: [BLOCK_T] (clamp for safe addressing of invalid tokens)
        safe_indices = tl.where(idx_valid, raw_indices, tl.zeros_like(raw_indices))
        page_ids = (safe_indices // page_size).to(tl.int64)
        page_offs_t = (safe_indices % page_size).to(tl.int64)
        token_data_bases = page_ids * page_bytes + page_offs_t * 576  # [BLOCK_T] int64

        # ---- Vectorized NOPE FP8 gather: [BLOCK_T, NOPE_PAD] ----
        nope_addrs = token_data_bases[:, None] + nope_offs[None, :].to(tl.int64)
        nope_2d_mask = idx_valid[:, None] & nope_mask[None, :]
        kv_nope_byte = tl.load(
            cache_nope_ptr + nope_addrs,
            mask=nope_2d_mask,
            other=0,
        )

        # ---- Vectorized scale gather + dequant: [BLOCK_T, NOPE_PAD] ----
        scale_bases = page_ids * page_bytes + scale_section_off + page_offs_t * 8
        scale_addrs = scale_bases[:, None] + group_ids[None, :]
        scale_raw = tl.load(
            cache_uint8_ptr + scale_addrs,
            mask=nope_2d_mask,
            other=127,
        )
        scale_f32 = tl.math.exp2(scale_raw.to(tl.float32) - 127.0)
        kv_nope_f32 = tl.load(lut_ptr + kv_nope_byte)
        kv_nope = tl.where(nope_2d_mask, kv_nope_f32 * scale_f32, 0.0)

        # ---- Vectorized ROPE BF16 gather: [BLOCK_T, ROPE_DIM] ----
        rope_byte_bases = token_data_bases + 448
        rope_elem_bases = (rope_byte_bases // 2).to(tl.int64)
        rope_addrs = rope_elem_bases[:, None] + rope_offs[None, :].to(tl.int64)
        kv_rope = tl.load(
            cache_bf16_ptr + rope_addrs,
            mask=idx_valid[:, None],
            other=0.0,
        ).to(tl.float32)

        # ---- Vectorized QK scores: [BLOCK_T] ----
        # scores[t] = dot(q_nope, kv_nope[t]) + dot(q_rope, kv_rope[t])
        scores = tl.sum(q_nope[None, :] * kv_nope, axis=1) + tl.sum(
            q_rope[None, :] * kv_rope, axis=1
        )
        scores = tl.where(idx_valid, scores, -1e30)

        # ---- Online softmax update (base-2, tile-level) ----
        scores_log2 = scores * LOG2E  # [BLOCK_T]
        tile_max = tl.max(scores_log2)  # scalar
        m_new = tl.maximum(m_i, tile_max)

        alpha = tl.math.exp2(m_i - m_new)  # rescale factor
        p = tl.math.exp2(scores_log2 - m_new)  # [BLOCK_T] attention weights
        p = tl.where(idx_valid, p, 0.0)  # zero out invalid

        l_i = l_i * alpha + tl.sum(p)

        # ---- Vectorized V accumulation (K=V in MLA) ----
        # acc += sum_t(p[t] * kv[t, :]) for both nope and rope
        acc_nope = acc_nope * alpha + tl.sum(p[:, None] * kv_nope, axis=0)
        acc_rope = acc_rope * alpha + tl.sum(p[:, None] * kv_rope, axis=0)
        m_i = m_new

    # ---- Normalize output ----
    safe_l = tl.where(l_i > 0.0, l_i, 1.0)
    acc_nope = acc_nope / safe_l
    acc_rope = acc_rope / safe_l

    # LSE: convert from log2 back to natural log
    lse = tl.where(l_i > 0.0, m_i / LOG2E + tl.math.log(safe_l), float("-inf"))

    # ---- Store output ----
    o_base = bid * stride_ob + hid * stride_oh
    # Store in the output buffer's dtype rather than hardcoding bfloat16:
    # sub-90 (SM75/SM80) runs the model in float16, where a bf16 round-trip both
    # loses mantissa bits (8 vs 10, ~4e-3 relative -- 14x the torch fallback's
    # 3.7e-4) and can overflow (fp16 max 65504 < bf16 max 3.4e38). The PyTorch
    # fallback returns out.to(q.dtype) for the same reason.
    tl.store(
        O_ptr + o_base + nope_offs, acc_nope.to(O_ptr.dtype.element_ty), mask=nope_mask
    )
    tl.store(
        O_ptr + o_base + NOPE_DIM_RT + rope_offs, acc_rope.to(O_ptr.dtype.element_ty)
    )
    tl.store(LSE_ptr + bid * H + hid, lse)


def _run_triton_sparse_decode(
    q: torch.Tensor,  # [B, 1, H, D] bf16
    k_cache: torch.Tensor,  # [num_pages, page_size, 1, bpt] float8
    indices: torch.Tensor,  # [B, ...] int32
    topk_length: Optional[torch.Tensor],
    softmax_scale: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run the tiled Triton sparse decode kernel on one paged KV cache."""
    B, _, H, D = q.shape
    num_pages = k_cache.shape[0]
    page_size = k_cache.shape[1]
    page_bytes = k_cache.stride(0)  # elements = bytes for float8

    # Flatten indices to [B, topk]
    flat_indices = indices.reshape(B, -1).contiguous()
    topk = flat_indices.shape[1]

    # Create three typed views of the flat cache memory.
    # The KV cache may arrive as uint8 or float8_e4m3fn depending on the
    # sglang version.  Ensure each view has the correct dtype so Triton
    # interprets the loaded values correctly (FP8 dequant vs raw integer).
    total_elems = num_pages * page_bytes
    raw_flat = k_cache.as_strided((total_elems,), (1,))
    raw_uint8 = raw_flat.view(torch.uint8)
    raw_bf16 = raw_uint8.view(torch.bfloat16)
    # e4m3 payload travels as uint8 and is decoded through this table: Triton on
    # SM75/SM80 cannot load an fp8 tensor at all.
    lut = fp8_payload_lut(q.device, torch.float32)

    # Squeeze Q: [B, H, D]
    q3 = q.squeeze(1)
    if not q3.is_contiguous():
        q3 = q3.contiguous()

    # Follow the query dtype (see the store above): sub-90 runs in float16.
    out = torch.zeros(B, H, D, dtype=q.dtype, device=q.device)
    lse = torch.full((B, H), float("-inf"), dtype=torch.float32, device=q.device)

    # Round topk for autotune key stability
    topk_rounded = triton.next_power_of_2(topk)

    grid = (B, H)
    _tiled_sparse_decode_kernel[grid](
        q3,
        raw_uint8,
        lut,
        raw_uint8,
        raw_bf16,
        flat_indices,
        (
            topk_length
            if topk_length is not None
            else torch.empty(0, device=q.device, dtype=torch.int32)
        ),
        out,
        lse,
        softmax_scale,
        page_size,
        int(page_bytes),  # page_bytes (int64)
        int(page_size * _TOKEN_DATA_STRIDE),  # scale_section_off (int64)
        H,
        topk,
        topk_rounded,
        topk_length is not None,
        q3.stride(0),
        q3.stride(1),
        out.stride(0),
        out.stride(1),
        flat_indices.stride(0),
        NOPE_PAD=512,
        ROPE_DIM=_ROPE_DIM,
        NOPE_DIM_RT=_NOPE_DIM,
    )

    # Return [B, 1, H, D] and [B, 1, H]
    return out.unsqueeze(1), lse.unsqueeze(1)


def _merge_partial_attn(
    out1: torch.Tensor,
    lse1: torch.Tensor,
    out2: torch.Tensor,
    lse2: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Merge two attention outputs using LSE-weighted combination.

    out: [B, 1, H, D] bf16,  lse: [B, 1, H] float32
    """
    max_lse = torch.maximum(lse1, lse2)
    w1 = torch.where(lse1 > -1e20, torch.exp(lse1 - max_lse), torch.zeros_like(lse1))
    w2 = torch.where(lse2 > -1e20, torch.exp(lse2 - max_lse), torch.zeros_like(lse2))
    total = (w1 + w2).clamp(min=1e-20)
    merged = (
        w1.unsqueeze(-1) * out1.float() + w2.unsqueeze(-1) * out2.float()
    ) / total.unsqueeze(-1)
    merged_lse = max_lse + torch.log(total)
    # Keep the caller's dtype (fp16 on sub-90); a bf16 round-trip here would
    # re-introduce the mantissa loss the kernel store just avoided.
    return merged.to(out1.dtype), merged_lse


def _apply_attn_sink(
    out: torch.Tensor,
    lse: torch.Tensor,
    attn_sink: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply attention sink normalization.

    The sink adds to the softmax denominator without contributing output,
    effectively down-weighting all attention scores.

    out: [B, 1, H, D] bf16,  lse: [B, 1, H] f32,  attn_sink: [H] f32
    """
    sink_lse = attn_sink.view(1, 1, -1).expand_as(lse)
    combined_lse = torch.logaddexp(lse, sink_lse)
    w = torch.where(
        lse > -1e20,
        torch.exp(lse - combined_lse),
        torch.zeros_like(lse),
    )
    return (out.float() * w.unsqueeze(-1)).to(out.dtype), combined_lse


def flash_mla_sparse_decode_triton(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    indices: torch.Tensor,
    topk_length: Optional[torch.Tensor],
    attn_sink: Optional[torch.Tensor],
    head_dim_v: int,
    softmax_scale: float,
    extra_k_cache: Optional[torch.Tensor] = None,
    extra_indices: Optional[torch.Tensor] = None,
    extra_topk_length: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """SM120-optimized sparse MLA decode using tiled Triton kernel.

    Processes SWA and extra (c4/c128) caches separately via the same
    Triton kernel, then merges results using LSE-weighted combination.
    """
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)

    # Process main cache (SWA)
    out, lse = _run_sparse_attention(q, k_cache, indices, topk_length, softmax_scale)

    # Process extra cache (c4 / c128) if present
    if extra_k_cache is not None and extra_indices is not None:
        out_extra, lse_extra = _run_sparse_attention(
            q, extra_k_cache, extra_indices, extra_topk_length, softmax_scale
        )
        out, lse = _merge_partial_attn(out, lse, out_extra, lse_extra)

    # Apply attention sink
    if attn_sink is not None:
        out, lse = _apply_attn_sink(out, lse, attn_sink)

    # Return format matching PyTorch fallback: (out, lse.permute(0,2,1))
    return out, lse.permute(0, 2, 1)


# ---------------------------------------------------------------------------
# Head-shared sparse attention for many-token batches (prefill).
#
# MLA has num_key_value_heads=1: every query head of a token attends to the same
# latent KV. The kernel above launches grid=(B, H), so each of the H heads
# re-gathers and re-dequantizes identical bytes. Measured on a 2080 Ti at the
# prefill shape (B=512, topk=512): H=1 -> 2.34ms, H=64 -> 144.6ms -- runtime is
# linear in H, so L2 absorbs none of the repeat. That kernel is 51.5% of all
# device time in an EXTEND trace (923.8 of 1792.6 ms).
#
# The kernel below gives one program BLOCK_H heads and one KV tile, gathered and
# decoded once for all of them. It needs tl.dot (the per-head broadcast form
# would hold [BLOCK_H, BLOCK_T, D] in registers), which imposes BLOCK_H >= 16 and
# BLOCK_T >= 16, and SM75's 64KB shared-memory cap forces the 448-wide nope half
# into two 256-wide halves. At BLOCK_H=16 it is 2.7x faster than the kernel above
# at B=512; below B=16 it loses (fewer programs than SMs), so dispatch by batch.
# ---------------------------------------------------------------------------

# Crossover with the per-head kernel, re-measured at production cache size
# (236800 tokens = 138 MB, so the gather is DRAM-resident rather than L2-resident)
# with the transposed QK gather in place. per-head / head-shared:
#   bs    per-head  head-shared  blocks (ph vs hs)
#    1      0.2883       0.5139   64 vs 4     0.56x  per-head wins
#    2      0.5568       0.5169  128 vs 8     1.08x
#    4      1.0938       0.5104  256 vs 16    2.14x
#    8      2.1844       0.5209  512 vs 32    4.19x
#   16      4.3253       0.6460 1024 vs 64    6.70x
# The head-shared kernel is flat to bs=16 because MLA shares one KV entry across all
# 64 heads, so it amortizes each gather over BLOCK_H=16 heads while the per-head
# kernel re-gathers the same 512x576 B block once per head. It only loses at bs=1,
# where it has B*H/16 = 4 blocks against the per-head kernel's B*H = 64. Measured
# device-only with the profiler (host wall time is useless here: the wrapper's
# torch.zeros / torch.full / contiguous / as_strided cost ~0.155 ms per call on the
# host, which CUDA graphs capture once in production but which swamps any host-side
# timing of these kernels):
#   topk    per-head  head-shared
#    512     0.2794      0.5054
#    256     0.1442      0.2563
#    128     0.0745      0.1298
#     64     0.0389      0.0665
# Head-shared is linear in topk with no device-side fixed cost, i.e. its 4 blocks each
# walk all 32 tiles serially with nothing overlapping. An in-kernel topk split (grid
# B x H/16 x n_splits, merged with _merge_partial_attn) should therefore approach the
# 8-tile cost and beat the per-head kernel at bs=1; it is untested. Note these probe
# numbers use a single topk=512 cache, while production splits attention into a small
# SWA window plus a c4/c128 compressed cache, so its per-call topk is smaller (the
# production DECODE trace shows 0.159 ms/call for this kernel, not 0.279).
# Simulating the split with n sequential launches is NOT a valid proxy: that
# serializes the blocks and pays the host wrapper n times, which measures ~linear
# slowdown regardless of what a single split launch would do.
# Splitting the topk loop is what moved this threshold from 2 to 1: unsplit, the kernel
# loses at B=1 (8.63 tok/s vs per-head 11.00 at 86K context); with 16 splits it wins
# (13.55). See SGLANG_SM75_HEADSHARED_MIN_BATCH in srt/environ.py, which is the value
# actually read -- this constant is documentation only.
_HEADSHARED_MIN_BATCH = 1
# Tile shape for the head-shared attention kernel, measured on a 2080 Ti.
#
# The kernel is neither bandwidth- nor compute-bound: at the shape the server
# actually runs (grid (113, 2), topk=512) it costs 2.524 ms against 2.368 ms for a
# bare gather of the same bytes, i.e. 1.07x the gather, and it moves data at 26
# GB/s -- 4% of the card's 616 GB/s. It is latency-bound, so what matters is how
# much independent gather work is in flight, and the dot width controls that by
# deciding how early a partial tile can feed an mma while the next one is still
# loading. Narrowing the dot keeps paying all the way down to 64 columns:
#
#   n256 w8 st1 (original)   1.000x
#   n128 w4 st1              1.217x
#   n64  w4 st1              1.431x
#   n64  w4 st2              1.49x   <- shipped, best or tied at every B from 16
#                                      to 512 and at both H=32 and H=64
#   n64  w8 st1              0.63x   <- more warps loses once the dot is narrow
#   n128 w2 st1              0.667x
#
# All of these are numerically equivalent to the original two-half form; the only
# difference is fp16 accumulation order (5e-4 against the old kernel).
_HS_NCOL = int(os.environ.get("SGLANG_SM75_HS_NCOL", "64"))
_HS_NUM_WARPS = int(os.environ.get("SGLANG_SM75_HS_WARPS", "4"))
_HS_NUM_STAGES = int(os.environ.get("SGLANG_SM75_HS_STAGES", "2"))
# Programs to split the topk loop across. 1 keeps the original single-program form.
_HS_TOPK_SPLIT = int(os.environ.get("SGLANG_SM75_HS_TOPK_SPLIT", "16"))
# Total programs at which this kernel stops getting faster from more blocks. Measured on
# a 2080 Ti (68 SMs): 4/8/16/32 blocks all cost 0.710 ms, 64 cost 0.869 (1.2x), 128 cost
# 1.671 (2.4x) -- so the machine fills near 64 programs and splitting past that only
# adds merge work.
_HS_SPLIT_TARGET_BLOCKS = int(os.environ.get("SGLANG_SM75_HS_SPLIT_BLOCKS", "64"))
# QK consumes a transposed gather instead of tl.trans'ing every KV chunk, and PV
# re-gathers naturally. Measured 1.68x on the prefill shape at identical accuracy
# (rel 3.882e-04 against the fp32 reference either way).
_HS_QK_TRANSPOSED = tl.constexpr(os.environ.get("SGLANG_SM75_HS_QK_TRANS", "1") == "1")
_HS_NOPE_HALF = tl.constexpr(256)
_HS_NOPE = tl.constexpr(448)
# Column span the chunked nope dots cover: 448 rounded up to a power of two.
_HS_NOPE_PAD = tl.constexpr(512)
_HS_LOG2E = tl.constexpr(1.4426950408889634)
_HS_ROPE = tl.constexpr(64)
_HS_TOKEN_BYTES = tl.constexpr(576)
_HS_SCALE_BYTES = tl.constexpr(8)
_HS_GROUP = tl.constexpr(64)


@triton.jit
def _hs_load_nope(
    cache_u8_ptr,
    lut_ptr,
    tok_base,
    cols,
    pg,
    po,
    kv_valid,
    page_bytes,
    scale_section_off,
):
    """Gather and decode one [BLOCK_T, 256] slice of the nope half.

    Columns past 448 are padding: they load byte 0 and LUT[0] == 0.0, so they
    contribute nothing to either dot product.
    """
    mask = kv_valid[:, None] & (cols < _HS_NOPE)[None, :]
    byte = tl.load(cache_u8_ptr + tok_base[:, None] + cols[None, :], mask=mask, other=0)
    scale_addr = (pg * page_bytes + scale_section_off + po * _HS_SCALE_BYTES)[
        :, None
    ] + (cols // _HS_GROUP)[None, :]
    scale = tl.math.exp2(
        tl.load(cache_u8_ptr + scale_addr, mask=mask, other=127).to(tl.float32) - 127.0
    )
    return (tl.load(lut_ptr + byte) * scale).to(tl.float16)


@triton.jit
def _hs_load_nope_t(
    cache_u8_ptr,
    lut_ptr,
    tok_base,
    cols,
    pg,
    po,
    kv_valid,
    page_bytes,
    scale_section_off,
):
    """Gather one [NCOL, BLOCK_T] nope slice, already transposed.

    The QK dot wants kv on the right of q, i.e. [NCOL, BLOCK_T], and the natural
    [BLOCK_T, NCOL] gather forces tl.trans on every chunk. Triton stages each
    register-computed tl.trans through shared memory before ldmatrix/mma, and
    with eight chunks per tile that dominates the kernel: a QK-only probe
    measured 13.47 ms with tl.trans against 3.45 ms gathering transposed.

    This reads exactly the same bytes at the same addresses, indexed transposed.
    The PV dot wants the natural layout, so the PV pass gathers again with
    _hs_load_nope; that second gather is free (a tile is 16 tokens x 576 B = 9 KB,
    still L1/L2 resident from the QK pass) and it is what keeps shared memory
    under the 64 KB SM75 cap, since only one tile is live per dot instead of
    eight.
    """
    mask = kv_valid[None, :] & (cols < _HS_NOPE)[:, None]
    byte = tl.load(cache_u8_ptr + tok_base[None, :] + cols[:, None], mask=mask, other=0)
    scale_addr = (pg * page_bytes + scale_section_off + po * _HS_SCALE_BYTES)[
        None, :
    ] + (cols // _HS_GROUP)[:, None]
    scale = tl.math.exp2(
        tl.load(cache_u8_ptr + scale_addr, mask=mask, other=127).to(tl.float32) - 127.0
    )
    return (tl.load(lut_ptr + byte) * scale).to(tl.float16)


@triton.jit
def _headshared_sparse_kernel(
    Q_ptr,  # [B, H, D] fp16
    cache_u8_ptr,  # uint8 flat: e4m3 nope payload + UE8M0 scales
    cache_bf16_ptr,  # bfloat16 flat: rope
    lut_ptr,  # fp32 [256]: e4m3 payload byte -> value
    indices_ptr,  # [B, topk] int32
    topk_len_ptr,  # [B] int32, or empty when HAS_TOPK_LEN is False
    O_ptr,  # [B, H, D]
    LSE_ptr,  # [B, H] fp32
    softmax_scale,
    page_size,
    page_bytes,
    scale_section_off,
    H: tl.constexpr,
    topk,
    HAS_TOPK_LEN: tl.constexpr,
    stride_qb: tl.int64,
    stride_qh: tl.int64,
    stride_os: tl.int64,
    stride_ls: tl.int64,
    stride_ob: tl.int64,
    stride_oh: tl.int64,
    BLOCK_H: tl.constexpr,
    BLOCK_T: tl.constexpr,
    NCOL: tl.constexpr,
    N_SPLIT: tl.constexpr,
):
    bid = tl.program_id(0)
    hblk = tl.program_id(1)
    # Flash-decoding style split over the topk axis. Each program owns a contiguous
    # slice of the tile loop and writes an unmerged partial plus its LSE, which the
    # caller combines. N_SPLIT=1 makes grid.z 1, so sid is 0, the range below is
    # [0, ceil(topk/BLOCK_T)) -- exactly the loop this kernel always ran -- and the
    # sid*stride terms vanish because the caller passes stride 0.
    #
    # This exists because at B=1 the grid is only (1, H/16) = 4 programs, so 4 of 68
    # SMs do work and every tile's gather latency is exposed serially. Measured on a
    # 2080 Ti, this kernel is FLAT in block count -- B=1 (4 blocks) and B=8 (32 blocks)
    # both cost 0.710 ms for 8x the work -- so extra blocks are free and splitting the
    # tile loop is what converts idle SMs into a shorter critical path.
    sid = tl.program_id(2)

    offs_h = hblk * BLOCK_H + tl.arange(0, BLOCK_H)
    h_valid = offs_h < H
    offs_t = tl.arange(0, BLOCK_T)
    offs_c = tl.arange(0, NCOL)
    rope_offs = tl.arange(0, _HS_ROPE)

    q_base = bid * stride_qb + offs_h * stride_qh
    # The nope half is 448 wide, which no power of two divides, so the columns are
    # covered in NCOL-wide chunks out to 512 and the tail is dropped by the
    # cols < 448 mask -- the same padding the original two-half form used, which
    # is why NCOL=256 reproduces that structure exactly.
    #
    # The chunk accumulators are written out for all 8 possible chunks (NCOL down
    # to 64) because Triton allows neither a list comprehension nor the tuple
    # builtin inside @jit; the NCHUNK guards make the unused ones dead code.
    NCHUNK: tl.constexpr = _HS_NOPE_PAD // NCOL
    q0 = tl.load(
        Q_ptr + q_base[:, None] + (0 * NCOL + offs_c)[None, :],
        mask=h_valid[:, None] & (0 * NCOL + offs_c < _HS_NOPE)[None, :],
        other=0.0,
    )
    q1 = tl.load(
        Q_ptr + q_base[:, None] + (1 * NCOL + offs_c)[None, :],
        mask=h_valid[:, None] & (1 * NCOL + offs_c < _HS_NOPE)[None, :],
        other=0.0,
    )
    q2 = tl.load(
        Q_ptr + q_base[:, None] + (2 * NCOL + offs_c)[None, :],
        mask=h_valid[:, None] & (2 * NCOL + offs_c < _HS_NOPE)[None, :],
        other=0.0,
    )
    q3 = tl.load(
        Q_ptr + q_base[:, None] + (3 * NCOL + offs_c)[None, :],
        mask=h_valid[:, None] & (3 * NCOL + offs_c < _HS_NOPE)[None, :],
        other=0.0,
    )
    q4 = tl.load(
        Q_ptr + q_base[:, None] + (4 * NCOL + offs_c)[None, :],
        mask=h_valid[:, None] & (4 * NCOL + offs_c < _HS_NOPE)[None, :],
        other=0.0,
    )
    q5 = tl.load(
        Q_ptr + q_base[:, None] + (5 * NCOL + offs_c)[None, :],
        mask=h_valid[:, None] & (5 * NCOL + offs_c < _HS_NOPE)[None, :],
        other=0.0,
    )
    q6 = tl.load(
        Q_ptr + q_base[:, None] + (6 * NCOL + offs_c)[None, :],
        mask=h_valid[:, None] & (6 * NCOL + offs_c < _HS_NOPE)[None, :],
        other=0.0,
    )
    q7 = tl.load(
        Q_ptr + q_base[:, None] + (7 * NCOL + offs_c)[None, :],
        mask=h_valid[:, None] & (7 * NCOL + offs_c < _HS_NOPE)[None, :],
        other=0.0,
    )
    q_r = tl.load(
        Q_ptr + q_base[:, None] + _HS_NOPE + rope_offs[None, :],
        mask=h_valid[:, None],
        other=0.0,
    )

    valid_len = topk
    if HAS_TOPK_LEN:
        valid_len = tl.load(topk_len_ptr + bid).to(tl.int32)

    m_i = tl.full([BLOCK_H], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_H], tl.float32)
    acc0 = tl.zeros([BLOCK_H, NCOL], tl.float32)
    acc1 = tl.zeros([BLOCK_H, NCOL], tl.float32)
    acc2 = tl.zeros([BLOCK_H, NCOL], tl.float32)
    acc3 = tl.zeros([BLOCK_H, NCOL], tl.float32)
    acc4 = tl.zeros([BLOCK_H, NCOL], tl.float32)
    acc5 = tl.zeros([BLOCK_H, NCOL], tl.float32)
    acc6 = tl.zeros([BLOCK_H, NCOL], tl.float32)
    acc7 = tl.zeros([BLOCK_H, NCOL], tl.float32)
    acc_r = tl.zeros([BLOCK_H, _HS_ROPE], tl.float32)

    # Split the tile loop across grid.z. With N_SPLIT=1 this is per_split=n_tiles,
    # t_lo=0 and t_hi=n_tiles, so the loop below is exactly the one this kernel always
    # ran -- the split path is a strict generalisation, not a second implementation.
    n_tiles = (topk + BLOCK_T - 1) // BLOCK_T
    per_split = (n_tiles + N_SPLIT - 1) // N_SPLIT
    t_lo = sid * per_split
    t_hi = tl.minimum(t_lo + per_split, n_tiles)

    for tile_start in range(t_lo * BLOCK_T, t_hi * BLOCK_T, BLOCK_T):
        t_idx = tile_start + offs_t
        raw = tl.load(indices_ptr + bid * topk + t_idx, mask=t_idx < topk, other=-1)
        idx_valid = (t_idx < valid_len) & (raw >= 0)
        safe = tl.where(idx_valid, raw, 0).to(tl.int64)
        page_ids = safe // page_size
        page_offs = safe % page_size
        tok_base = page_ids * page_bytes + page_offs * _HS_TOKEN_BYTES

        scores = tl.zeros([BLOCK_H, BLOCK_T], tl.float32)
        if NCHUNK >= 1:
            if _HS_QK_TRANSPOSED:
                scores += tl.dot(
                    q0,
                    _hs_load_nope_t(
                        cache_u8_ptr,
                        lut_ptr,
                        tok_base,
                        0 * NCOL + offs_c,
                        page_ids,
                        page_offs,
                        idx_valid,
                        page_bytes,
                        scale_section_off,
                    ),
                )
            else:
                kv0 = _hs_load_nope(
                    cache_u8_ptr,
                    lut_ptr,
                    tok_base,
                    0 * NCOL + offs_c,
                    page_ids,
                    page_offs,
                    idx_valid,
                    page_bytes,
                    scale_section_off,
                )
                scores += tl.dot(q0, tl.trans(kv0))
        if NCHUNK >= 2:
            if _HS_QK_TRANSPOSED:
                scores += tl.dot(
                    q1,
                    _hs_load_nope_t(
                        cache_u8_ptr,
                        lut_ptr,
                        tok_base,
                        1 * NCOL + offs_c,
                        page_ids,
                        page_offs,
                        idx_valid,
                        page_bytes,
                        scale_section_off,
                    ),
                )
            else:
                kv1 = _hs_load_nope(
                    cache_u8_ptr,
                    lut_ptr,
                    tok_base,
                    1 * NCOL + offs_c,
                    page_ids,
                    page_offs,
                    idx_valid,
                    page_bytes,
                    scale_section_off,
                )
                scores += tl.dot(q1, tl.trans(kv1))
        if NCHUNK >= 3:
            if _HS_QK_TRANSPOSED:
                scores += tl.dot(
                    q2,
                    _hs_load_nope_t(
                        cache_u8_ptr,
                        lut_ptr,
                        tok_base,
                        2 * NCOL + offs_c,
                        page_ids,
                        page_offs,
                        idx_valid,
                        page_bytes,
                        scale_section_off,
                    ),
                )
            else:
                kv2 = _hs_load_nope(
                    cache_u8_ptr,
                    lut_ptr,
                    tok_base,
                    2 * NCOL + offs_c,
                    page_ids,
                    page_offs,
                    idx_valid,
                    page_bytes,
                    scale_section_off,
                )
                scores += tl.dot(q2, tl.trans(kv2))
        if NCHUNK >= 4:
            if _HS_QK_TRANSPOSED:
                scores += tl.dot(
                    q3,
                    _hs_load_nope_t(
                        cache_u8_ptr,
                        lut_ptr,
                        tok_base,
                        3 * NCOL + offs_c,
                        page_ids,
                        page_offs,
                        idx_valid,
                        page_bytes,
                        scale_section_off,
                    ),
                )
            else:
                kv3 = _hs_load_nope(
                    cache_u8_ptr,
                    lut_ptr,
                    tok_base,
                    3 * NCOL + offs_c,
                    page_ids,
                    page_offs,
                    idx_valid,
                    page_bytes,
                    scale_section_off,
                )
                scores += tl.dot(q3, tl.trans(kv3))
        if NCHUNK >= 5:
            if _HS_QK_TRANSPOSED:
                scores += tl.dot(
                    q4,
                    _hs_load_nope_t(
                        cache_u8_ptr,
                        lut_ptr,
                        tok_base,
                        4 * NCOL + offs_c,
                        page_ids,
                        page_offs,
                        idx_valid,
                        page_bytes,
                        scale_section_off,
                    ),
                )
            else:
                kv4 = _hs_load_nope(
                    cache_u8_ptr,
                    lut_ptr,
                    tok_base,
                    4 * NCOL + offs_c,
                    page_ids,
                    page_offs,
                    idx_valid,
                    page_bytes,
                    scale_section_off,
                )
                scores += tl.dot(q4, tl.trans(kv4))
        if NCHUNK >= 6:
            if _HS_QK_TRANSPOSED:
                scores += tl.dot(
                    q5,
                    _hs_load_nope_t(
                        cache_u8_ptr,
                        lut_ptr,
                        tok_base,
                        5 * NCOL + offs_c,
                        page_ids,
                        page_offs,
                        idx_valid,
                        page_bytes,
                        scale_section_off,
                    ),
                )
            else:
                kv5 = _hs_load_nope(
                    cache_u8_ptr,
                    lut_ptr,
                    tok_base,
                    5 * NCOL + offs_c,
                    page_ids,
                    page_offs,
                    idx_valid,
                    page_bytes,
                    scale_section_off,
                )
                scores += tl.dot(q5, tl.trans(kv5))
        if NCHUNK >= 7:
            if _HS_QK_TRANSPOSED:
                scores += tl.dot(
                    q6,
                    _hs_load_nope_t(
                        cache_u8_ptr,
                        lut_ptr,
                        tok_base,
                        6 * NCOL + offs_c,
                        page_ids,
                        page_offs,
                        idx_valid,
                        page_bytes,
                        scale_section_off,
                    ),
                )
            else:
                kv6 = _hs_load_nope(
                    cache_u8_ptr,
                    lut_ptr,
                    tok_base,
                    6 * NCOL + offs_c,
                    page_ids,
                    page_offs,
                    idx_valid,
                    page_bytes,
                    scale_section_off,
                )
                scores += tl.dot(q6, tl.trans(kv6))
        if NCHUNK >= 8:
            if _HS_QK_TRANSPOSED:
                scores += tl.dot(
                    q7,
                    _hs_load_nope_t(
                        cache_u8_ptr,
                        lut_ptr,
                        tok_base,
                        7 * NCOL + offs_c,
                        page_ids,
                        page_offs,
                        idx_valid,
                        page_bytes,
                        scale_section_off,
                    ),
                )
            else:
                kv7 = _hs_load_nope(
                    cache_u8_ptr,
                    lut_ptr,
                    tok_base,
                    7 * NCOL + offs_c,
                    page_ids,
                    page_offs,
                    idx_valid,
                    page_bytes,
                    scale_section_off,
                )
                scores += tl.dot(q7, tl.trans(kv7))
        rope_base = ((tok_base + _HS_NOPE) // 2).to(tl.int64)
        if _HS_QK_TRANSPOSED:
            kv_r_t = tl.load(
                cache_bf16_ptr + rope_base[None, :] + rope_offs[:, None],
                mask=idx_valid[None, :],
                other=0.0,
            ).to(tl.float16)
            scores += tl.dot(q_r, kv_r_t)
        else:
            kv_r = tl.load(
                cache_bf16_ptr + rope_base[:, None] + rope_offs[None, :],
                mask=idx_valid[:, None],
                other=0.0,
            ).to(tl.float16)
            scores += tl.dot(q_r, tl.trans(kv_r))

        s = tl.where(
            idx_valid[None, :], scores * (softmax_scale * _HS_LOG2E), float("-inf")
        )
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.where(m_i == float("-inf"), 0.0, tl.math.exp2(m_i - m_safe))
        p = tl.where(idx_valid[None, :], tl.math.exp2(s - m_safe[:, None]), 0.0)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        pf = p.to(tl.float16)
        if _HS_QK_TRANSPOSED:
            kv_r_n = tl.load(
                cache_bf16_ptr + rope_base[:, None] + rope_offs[None, :],
                mask=idx_valid[:, None],
                other=0.0,
            ).to(tl.float16)
            acc_r = acc_r * alpha[:, None] + tl.dot(pf, kv_r_n)
        else:
            acc_r = acc_r * alpha[:, None] + tl.dot(pf, kv_r)
        m_i = m_new

        if NCHUNK >= 1:
            if _HS_QK_TRANSPOSED:
                acc0 = acc0 * alpha[:, None] + tl.dot(
                    pf,
                    _hs_load_nope(
                        cache_u8_ptr,
                        lut_ptr,
                        tok_base,
                        0 * NCOL + offs_c,
                        page_ids,
                        page_offs,
                        idx_valid,
                        page_bytes,
                        scale_section_off,
                    ),
                )
            else:
                acc0 = acc0 * alpha[:, None] + tl.dot(pf, kv0)
        if NCHUNK >= 2:
            if _HS_QK_TRANSPOSED:
                acc1 = acc1 * alpha[:, None] + tl.dot(
                    pf,
                    _hs_load_nope(
                        cache_u8_ptr,
                        lut_ptr,
                        tok_base,
                        1 * NCOL + offs_c,
                        page_ids,
                        page_offs,
                        idx_valid,
                        page_bytes,
                        scale_section_off,
                    ),
                )
            else:
                acc1 = acc1 * alpha[:, None] + tl.dot(pf, kv1)
        if NCHUNK >= 3:
            if _HS_QK_TRANSPOSED:
                acc2 = acc2 * alpha[:, None] + tl.dot(
                    pf,
                    _hs_load_nope(
                        cache_u8_ptr,
                        lut_ptr,
                        tok_base,
                        2 * NCOL + offs_c,
                        page_ids,
                        page_offs,
                        idx_valid,
                        page_bytes,
                        scale_section_off,
                    ),
                )
            else:
                acc2 = acc2 * alpha[:, None] + tl.dot(pf, kv2)
        if NCHUNK >= 4:
            if _HS_QK_TRANSPOSED:
                acc3 = acc3 * alpha[:, None] + tl.dot(
                    pf,
                    _hs_load_nope(
                        cache_u8_ptr,
                        lut_ptr,
                        tok_base,
                        3 * NCOL + offs_c,
                        page_ids,
                        page_offs,
                        idx_valid,
                        page_bytes,
                        scale_section_off,
                    ),
                )
            else:
                acc3 = acc3 * alpha[:, None] + tl.dot(pf, kv3)
        if NCHUNK >= 5:
            if _HS_QK_TRANSPOSED:
                acc4 = acc4 * alpha[:, None] + tl.dot(
                    pf,
                    _hs_load_nope(
                        cache_u8_ptr,
                        lut_ptr,
                        tok_base,
                        4 * NCOL + offs_c,
                        page_ids,
                        page_offs,
                        idx_valid,
                        page_bytes,
                        scale_section_off,
                    ),
                )
            else:
                acc4 = acc4 * alpha[:, None] + tl.dot(pf, kv4)
        if NCHUNK >= 6:
            if _HS_QK_TRANSPOSED:
                acc5 = acc5 * alpha[:, None] + tl.dot(
                    pf,
                    _hs_load_nope(
                        cache_u8_ptr,
                        lut_ptr,
                        tok_base,
                        5 * NCOL + offs_c,
                        page_ids,
                        page_offs,
                        idx_valid,
                        page_bytes,
                        scale_section_off,
                    ),
                )
            else:
                acc5 = acc5 * alpha[:, None] + tl.dot(pf, kv5)
        if NCHUNK >= 7:
            if _HS_QK_TRANSPOSED:
                acc6 = acc6 * alpha[:, None] + tl.dot(
                    pf,
                    _hs_load_nope(
                        cache_u8_ptr,
                        lut_ptr,
                        tok_base,
                        6 * NCOL + offs_c,
                        page_ids,
                        page_offs,
                        idx_valid,
                        page_bytes,
                        scale_section_off,
                    ),
                )
            else:
                acc6 = acc6 * alpha[:, None] + tl.dot(pf, kv6)
        if NCHUNK >= 8:
            if _HS_QK_TRANSPOSED:
                acc7 = acc7 * alpha[:, None] + tl.dot(
                    pf,
                    _hs_load_nope(
                        cache_u8_ptr,
                        lut_ptr,
                        tok_base,
                        7 * NCOL + offs_c,
                        page_ids,
                        page_offs,
                        idx_valid,
                        page_bytes,
                        scale_section_off,
                    ),
                )
            else:
                acc7 = acc7 * alpha[:, None] + tl.dot(pf, kv7)
    safe_l = tl.where(l_i > 0.0, l_i, 1.0)
    # Partials are written normalized (acc/l_i) alongside their LSE, which is what the
    # LSE-weighted combine expects. stride_os/stride_ls index the split axis and are 0
    # when N_SPLIT=1, so the unsplit path stores to exactly the same addresses as before.
    o_base = sid * stride_os + bid * stride_ob + offs_h * stride_oh
    if NCHUNK >= 1:
        ci = 0 * NCOL + offs_c
        tl.store(
            O_ptr + o_base[:, None] + ci[None, :],
            (acc0 / safe_l[:, None]).to(O_ptr.dtype.element_ty),
            mask=h_valid[:, None] & (ci < _HS_NOPE)[None, :],
        )
    if NCHUNK >= 2:
        ci = 1 * NCOL + offs_c
        tl.store(
            O_ptr + o_base[:, None] + ci[None, :],
            (acc1 / safe_l[:, None]).to(O_ptr.dtype.element_ty),
            mask=h_valid[:, None] & (ci < _HS_NOPE)[None, :],
        )
    if NCHUNK >= 3:
        ci = 2 * NCOL + offs_c
        tl.store(
            O_ptr + o_base[:, None] + ci[None, :],
            (acc2 / safe_l[:, None]).to(O_ptr.dtype.element_ty),
            mask=h_valid[:, None] & (ci < _HS_NOPE)[None, :],
        )
    if NCHUNK >= 4:
        ci = 3 * NCOL + offs_c
        tl.store(
            O_ptr + o_base[:, None] + ci[None, :],
            (acc3 / safe_l[:, None]).to(O_ptr.dtype.element_ty),
            mask=h_valid[:, None] & (ci < _HS_NOPE)[None, :],
        )
    if NCHUNK >= 5:
        ci = 4 * NCOL + offs_c
        tl.store(
            O_ptr + o_base[:, None] + ci[None, :],
            (acc4 / safe_l[:, None]).to(O_ptr.dtype.element_ty),
            mask=h_valid[:, None] & (ci < _HS_NOPE)[None, :],
        )
    if NCHUNK >= 6:
        ci = 5 * NCOL + offs_c
        tl.store(
            O_ptr + o_base[:, None] + ci[None, :],
            (acc5 / safe_l[:, None]).to(O_ptr.dtype.element_ty),
            mask=h_valid[:, None] & (ci < _HS_NOPE)[None, :],
        )
    if NCHUNK >= 7:
        ci = 6 * NCOL + offs_c
        tl.store(
            O_ptr + o_base[:, None] + ci[None, :],
            (acc6 / safe_l[:, None]).to(O_ptr.dtype.element_ty),
            mask=h_valid[:, None] & (ci < _HS_NOPE)[None, :],
        )
    if NCHUNK >= 8:
        ci = 7 * NCOL + offs_c
        tl.store(
            O_ptr + o_base[:, None] + ci[None, :],
            (acc7 / safe_l[:, None]).to(O_ptr.dtype.element_ty),
            mask=h_valid[:, None] & (ci < _HS_NOPE)[None, :],
        )
    tl.store(
        O_ptr + o_base[:, None] + _HS_NOPE + rope_offs[None, :],
        (acc_r / safe_l[:, None]).to(O_ptr.dtype.element_ty),
        mask=h_valid[:, None],
    )
    tl.store(
        LSE_ptr + sid * stride_ls + bid * H + offs_h,
        tl.where(l_i > 0.0, m_i / _HS_LOG2E + tl.math.log(safe_l), float("-inf")),
        mask=h_valid,
    )


@triton.jit
def _merge_splits_kernel(
    part_ptr,  # [S, B, H, D] fp16, each split's output already normalized
    part_lse_ptr,  # [S, B, H] f32
    out_ptr,  # [B, H, D]
    lse_ptr,  # [B, H] f32
    n_split,
    H: tl.constexpr,
    D: tl.constexpr,
    stride_ps: tl.int64,
    stride_pb: tl.int64,
    stride_ph: tl.int64,
    stride_ls: tl.int64,
    stride_lb: tl.int64,
    stride_ob: tl.int64,
    stride_oh: tl.int64,
    BLOCK_D: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    """Combine topk-split partials in one launch: one program per (batch, head).

    Same LSE-weighted combination as _merge_partial_attn, generalised over S splits:
    each partial carries its own softmax max and denominator, so exp(lse_s - max_lse)
    is split s's relative weight and the weighted mean of the normalized partials is the
    exact full-softmax result.

    This replaces a plain-torch version that cost 0.058 ms across 17 kernels per call.
    That was 48% of the whole split attention path and, because those kernels are
    elementwise, it showed up in the profile as elementwise work rather than attention:
    1.28 ms per stage per step at 22 attention calls per step. One launch per call
    removes 16 of the 17.
    """
    pid = tl.program_id(0)
    bid = pid // H
    hid = pid % H

    offs_s = tl.arange(0, BLOCK_S)
    s_valid = offs_s < n_split
    l = tl.load(
        part_lse_ptr + offs_s * stride_ls + bid * stride_lb + hid,
        mask=s_valid,
        other=float("-inf"),
    )
    m = tl.max(l, axis=0)
    total = tl.sum(tl.where(s_valid & (l > -1e20), tl.exp(l - m), 0.0), axis=0)

    offs_d = tl.arange(0, BLOCK_D)
    d_valid = offs_d < D
    base = bid * stride_ob + hid * stride_oh
    acc = tl.zeros([BLOCK_D], tl.float32)
    for s in range(n_split):
        ls = tl.load(part_lse_ptr + s * stride_ls + bid * stride_lb + hid)
        ws = tl.where(ls > -1e20, tl.exp(ls - m), 0.0)
        v = tl.load(
            part_ptr + s * stride_ps + bid * stride_pb + hid * stride_ph + offs_d,
            mask=d_valid,
            other=0.0,
        )
        acc += ws * v.to(tl.float32)

    safe = tl.where(total > 0.0, total, 1.0)
    tl.store(
        out_ptr + base + offs_d, (acc / safe).to(out_ptr.dtype.element_ty), mask=d_valid
    )
    tl.store(
        lse_ptr + bid * stride_lb + hid,
        tl.where(total > 0.0, m + tl.log(safe), float("-inf")),
    )


def _merge_splits(
    part: torch.Tensor,  # [S, B, H, D] fp16, each row normalized
    part_lse: torch.Tensor,  # [S, B, H] f32
    out: torch.Tensor,  # [B, H, D]
    lse: torch.Tensor,  # [B, H] f32
) -> None:
    B, H, D = out.shape
    _merge_splits_kernel[(B * H,)](
        part,
        part_lse,
        out,
        lse,
        part.shape[0],
        H,
        D,
        part.stride(0),
        part.stride(1),
        part.stride(2),
        part_lse.stride(0),
        part_lse.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_D=triton.next_power_of_2(D),
        BLOCK_S=32,
        num_warps=4,
    )


def _run_headshared_sparse_decode(
    q: torch.Tensor,  # [B, 1, H, D]
    k_cache: torch.Tensor,
    indices: torch.Tensor,
    topk_length: Optional[torch.Tensor],
    softmax_scale: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Head-shared sparse attention for many-token batches. See the kernel."""
    B, _, H, D = q.shape
    num_pages = k_cache.shape[0]
    page_size = k_cache.shape[1]
    page_bytes = k_cache.stride(0)

    flat_indices = indices.reshape(B, -1).contiguous()
    topk = flat_indices.shape[1]

    total_elems = num_pages * page_bytes
    raw_flat = k_cache.as_strided((total_elems,), (1,))
    raw_uint8 = raw_flat.view(torch.uint8)
    raw_bf16 = raw_uint8.view(torch.bfloat16)
    lut = fp8_payload_lut(q.device, torch.float32)

    q3 = q.squeeze(1)
    if not q3.is_contiguous():
        q3 = q3.contiguous()

    out = torch.zeros(B, H, D, dtype=q.dtype, device=q.device)
    lse = torch.full((B, H), float("-inf"), dtype=torch.float32, device=q.device)

    block_h = min(16, triton.next_power_of_2(H))
    block_t = 16
    # Split the topk loop across grid.z. This kernel's cost is flat in block count
    # (measured: 4 blocks and 32 blocks both cost 0.710 ms for 8x the work, 1.2x at 64,
    # 2.4x at 128), so on 68 SMs the machine is filled at roughly 64 programs and
    # splitting below that converts idle SMs into a shorter serial gather chain.
    #
    # Deriving n_split from the block budget rather than fixing it does two things. It
    # reproduces the measured optimum at B=1 (64 // 4 = 16 splits, and 16 was fastest),
    # and it turns the split off by itself once B*H/16 already fills the machine --
    # which matters because this same path runs during EXTEND at B in the hundreds,
    # where the partial buffers would otherwise cost S*B*H*D*2 bytes (a 512 MiB fp32
    # temporary at B=216, S=16, which is what OOMed the first end-to-end attempt).
    n_tiles = (topk + block_t - 1) // block_t
    blocks_per_batch = triton.cdiv(H, block_h)
    n_split = max(
        1,
        min(_HS_TOPK_SPLIT, n_tiles, _HS_SPLIT_TARGET_BLOCKS // (B * blocks_per_batch)),
    )

    if n_split == 1:
        grid = (B, triton.cdiv(H, block_h))
        _headshared_sparse_kernel[grid](
            q3,
            raw_uint8,
            raw_bf16,
            lut,
            flat_indices,
            (
                topk_length
                if topk_length is not None
                else torch.empty(0, device=q.device, dtype=torch.int32)
            ),
            out,
            lse,
            softmax_scale,
            page_size,
            int(page_bytes),
            int(page_size * _HS_TOKEN_BYTES),
            H,
            topk,
            topk_length is not None,
            q3.stride(0),
            q3.stride(1),
            0,
            0,
            out.stride(0),
            out.stride(1),
            BLOCK_H=block_h,
            BLOCK_T=block_t,
            NCOL=_HS_NCOL,
            N_SPLIT=1,
            num_warps=_HS_NUM_WARPS,
            num_stages=_HS_NUM_STAGES,
        )
        return out.unsqueeze(1), lse.unsqueeze(1)

    part = torch.empty((n_split, B, H, D), dtype=q.dtype, device=q.device)
    part_lse = torch.empty((n_split, B, H), dtype=torch.float32, device=q.device)
    grid = (B, triton.cdiv(H, block_h), n_split)
    _headshared_sparse_kernel[grid](
        q3,
        raw_uint8,
        raw_bf16,
        lut,
        flat_indices,
        (
            topk_length
            if topk_length is not None
            else torch.empty(0, device=q.device, dtype=torch.int32)
        ),
        part,
        part_lse,
        softmax_scale,
        page_size,
        int(page_bytes),
        int(page_size * _HS_TOKEN_BYTES),
        H,
        topk,
        topk_length is not None,
        q3.stride(0),
        q3.stride(1),
        part.stride(0),
        part_lse.stride(0),
        out.stride(0),
        out.stride(1),
        BLOCK_H=block_h,
        BLOCK_T=block_t,
        NCOL=_HS_NCOL,
        N_SPLIT=n_split,
        num_warps=_HS_NUM_WARPS,
        num_stages=_HS_NUM_STAGES,
    )
    _merge_splits(part, part_lse, out, lse)
    return out.unsqueeze(1), lse.unsqueeze(1)


def _run_sparse_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    indices: torch.Tensor,
    topk_length: Optional[torch.Tensor],
    softmax_scale: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pick the sparse-attention kernel that fits the batch size.

    The head-shared kernel amortizes each KV tile over BLOCK_H=16 heads, and MLA shares
    one KV entry across all 64 heads, so it is flat in batch while the per-head kernel
    re-gathers the same block once per head. It used to lose at B=1, where it launches
    only B*H/16 = 4 blocks against the per-head kernel's B*H = 64 and each block walks
    all ceil(topk/16) tiles serially; that is why the threshold was 2, and unsplit the
    numbers were right (86K context, bs=1: 8.63 tok/s vs per-head's 11.00).

    Splitting the topk loop across grid.z removes the serial tile walk, and the split
    count is derived from the block budget so B=1 gets 16-way parallelism while B>=16
    self-disables it. Measured end to end at 86K context, TP2/PP4, 150 W: head-shared
    with 16 splits gives 13.55 tok/s against per-head's 11.00 (+23%), so the kernel now
    wins at every batch size and the threshold is 1.

    SGLANG_SM75_HEADSHARED_MIN_BATCH=0 forces the per-head kernel everywhere,
    which is the A/B switch for measuring the head-shared kernel end to end.
    """
    if _headshared_min_batch() > 0 and q.shape[0] >= _headshared_min_batch():
        return _run_headshared_sparse_decode(
            q, k_cache, indices, topk_length, softmax_scale
        )
    return _run_triton_sparse_decode(q, k_cache, indices, topk_length, softmax_scale)


def _headshared_min_batch() -> int:
    return int(envs.SGLANG_SM75_HEADSHARED_MIN_BATCH.get())
