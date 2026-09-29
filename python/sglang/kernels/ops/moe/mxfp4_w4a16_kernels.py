"""MXFP4 (e2m1 nibble-packed in int8, UE8M0 scale per 32) W4A16 grouped GEMM.

Target: GPUs without FP8/FP4 tensor cores (SM75/Turing, SM80/Ampere) where the
DeepSeek-V4 mixed checkpoint's routed experts (I8-packed MXFP4) cannot use
Marlin / FlashInfer / DeepGEMM kernels. Weights stay packed on disk and in HBM
(no dequant-at-load: 138 GiB packed -> 276 GiB fp16 would not fit); the
dequantization happens in registers inside the GEMM.

Layout contract (matches the DeepSeek-V4-Flash-0731 checkpoint and the
sglang-internal cast_e2m1fn_to_e4m3fn convention, fp8.py:190-209):
  - weight:  [E, N, K // 2] int8 (viewed as uint8). Each byte holds two e2m1
             codes: low nibble = element 2*i, high nibble = element 2*i + 1.
  - scale:   [E, N, K // 32] float32 (the loaded UE8M0 scale values), or
             uint8 UE8M0 bytes when SCALE_IS_E8M0=True.

Grouped-GEMM contract (matches moe_align_block_size output):
  - sorted_token_ids: [num_padded] int32; pad slots carry the sentinel M_total.
  - expert_ids: [num_blocks] int32, one expert per BLOCK_M-row block.
  - C is SLOT-indexed: row = sorted slot (the caller scatters/combines).

Numerics: activation in fp16, e2m1 dequantized arithmetically (no LUT gather:
the e2m1 value is 2^(e-1) * (1 + m/2) for e>0, 0.5*m for e==0), fp32
accumulate via tl.dot. The packed byte is loaded once and split into the two
element rows with tl.interleave, so each weight byte costs one 8-bit load.
"""

from typing import Optional

import torch
import triton
import triton.language as tl

# e2m1: s e e m -> value table for nibble index 0..15.
_E2M1_LUT = [
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
]

_lut_cache: dict = {}


def get_e2m1_lut(device) -> torch.Tensor:
    lut = _lut_cache.get(device)
    if lut is None:
        lut = torch.tensor(_E2M1_LUT, dtype=torch.float16, device=device)
        _lut_cache[device] = lut
    return lut


def dequant_mxfp4_reference(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Reference dequant: [E, N, K//2] int8 + [E, N, K//32] scale -> [E, N, K] fp32.

    `scale` may be uint8 UE8M0 bytes or float32 values.
    """
    lut = torch.tensor(_E2M1_LUT, dtype=torch.float32, device=weight.device)
    w = weight.view(torch.uint8).to(torch.int64)
    lo = lut[w & 0x0F]
    hi = lut[(w >> 4) & 0x0F]
    vals = torch.stack([lo, hi], dim=-1).flatten(-2)  # [E, N, K]
    if scale.dtype == torch.uint8:
        scales = torch.exp2(scale.to(torch.float32) - 127.0)
    else:
        scales = scale.to(torch.float32)
    scales = scales.repeat_interleave(32, dim=-1)
    return vals * scales


@triton.jit
def _e2m1_decode(code):
    """Arithmetic e2m1 decode: code uint8 [0,15] -> fp16 value.

    e2m1: value = 2^(e-1) * (1 + m/2) for e>0; 0.5*m for e==0; sign bit 3.
    """
    c = code.to(tl.int32)
    sign = tl.where((c & 0x08) != 0, -1.0, 1.0).to(tl.float32)
    e = (c >> 1) & 0x03
    m = c & 0x01
    val = tl.where(
        e == 0,
        0.5 * m.to(tl.float32),
        tl.exp2((e - 1).to(tl.float32)) * (1.0 + 0.5 * m.to(tl.float32)),
    )
    return (sign * val).to(tl.float16)


@triton.jit
def _mxfp4_w4a16_gemm_kernel(
    a_ptr,  # [M_total, K] fp16 (token rows; gathered via sorted ids)
    w_ptr,  # [E, N, K // 2] uint8-packed e2m1
    s_ptr,  # [E, N, K // 32] float32 or uint8 scales
    c_ptr,  # [num_padded, N] fp16, SLOT-indexed output
    sorted_token_ids_ptr,  # [num_padded] int32 (sentinel M_total for pad)
    expert_ids_ptr,  # [num_blocks] int32
    num_valid_ptr,  # [1] int32: entries of sorted/expert ids actually written
    M_total,
    N,
    K,
    stride_am,
    stride_we,
    stride_wn,
    stride_wk,
    stride_se,
    stride_sn,
    stride_sk,
    stride_cm,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SCALE_IS_E8M0: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    # Past num_valid the sorted/expert id buffers hold uninitialized memory;
    # loading a garbage expert would index the weights out of bounds.
    if pid_m * BLOCK_M >= tl.load(num_valid_ptr):
        return

    expert = tl.load(expert_ids_ptr + pid_m).to(tl.int64)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    token_ids = tl.load(sorted_token_ids_ptr + offs_m)
    valid = token_ids < M_total
    token_ids64 = tl.where(valid, token_ids, 0).to(tl.int64)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N
    # one packed byte covers two k elements
    BLOCK_KH: tl.constexpr = BLOCK_K // 2
    offs_kh = tl.arange(0, BLOCK_KH)  # packed-k offsets

    a_ptrs = a_ptr + token_ids64[:, None] * stride_am + tl.arange(0, BLOCK_K)[None, :]
    # NOTE: load with the contiguous (k) axis LAST so Triton vectorizes the
    # byte loads; [N, KH] orientation.
    w_ptrs = (
        w_ptr
        + expert * stride_we
        + offs_n[:, None] * stride_wn
        + offs_kh[None, :] * stride_wk
    )
    # scale: one value per 32 elements = per 16 packed bytes
    s_ptrs = s_ptr + expert * stride_se + offs_n[:, None] * stride_sn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for kt in range(0, tl.cdiv(K, BLOCK_K)):
        k_rem = K - kt * BLOCK_K
        kh_rem = k_rem // 2
        a = tl.load(
            a_ptrs,
            mask=valid[:, None] & (tl.arange(0, BLOCK_K)[None, :] < k_rem),
            other=0.0,
        )
        b_u8 = tl.load(
            w_ptrs, mask=n_mask[:, None] & (offs_kh[None, :] < kh_rem), other=0
        )

        lo = _e2m1_decode(b_u8 & 0x0F)
        hi = _e2m1_decode((b_u8 >> 4) & 0x0F)

        # scale index per packed byte: element 2*kh -> group (2*kh)//32 = kh//16
        s_raw = tl.load(
            s_ptrs + (kt * (BLOCK_K // 32) + offs_kh[None, :] // 16) * stride_sk,
            mask=n_mask[:, None] & (offs_kh[None, :] < kh_rem),
            other=127,
        )  # [BLOCK_N, BLOCK_KH]
        if SCALE_IS_E8M0:
            s = tl.exp2((s_raw.to(tl.int32) - 127).to(tl.float32)).to(tl.float16)
        else:
            s = s_raw.to(tl.float16)

        # interleave along the last (k) axis: [N, KH] -> [N, K], then one
        # transpose for the dot operand.
        b = tl.trans(tl.interleave(lo * s, hi * s))  # [BLOCK_K, BLOCK_N]

        acc += tl.dot(a, b, out_dtype=tl.float32)

        a_ptrs += BLOCK_K
        w_ptrs += BLOCK_KH * stride_wk

    c_ptrs = c_ptr + offs_m[:, None].to(tl.int64) * stride_cm + offs_n[None, :]
    tl.store(c_ptrs, acc.to(tl.float16), mask=n_mask[None, :])


def mxfp4_w4a16_gemm(
    a: torch.Tensor,  # [M_total, K] fp16 (token-indexed activations)
    w: torch.Tensor,  # [E, N, K//2] int8 (packed)
    s: torch.Tensor,  # [E, N, K//32] float32 scale values (or uint8 e8m0)
    sorted_token_ids: torch.Tensor,  # [num_padded] int32 (sentinel M_total)
    expert_ids: torch.Tensor,  # [num_blocks] int32
    out: torch.Tensor,  # [num_padded, N] slot-indexed output
    block_m: int = 16,
    block_n: int = 64,
    block_k: int = 128,
    sentinel: int = None,  # invalid-row id (default: a.shape[0])
    num_warps: int = 8,
    num_valid: Optional[torch.Tensor] = None,  # [1] int32 device scalar
) -> torch.Tensor:
    """Grouped W4A16 GEMM over MXFP4-packed experts. Writes slot-indexed `out`.

    Rows of `a` are gathered by the slot id (value in sorted_token_ids); rows
    whose id >= `sentinel` are padding and store zeros. `sentinel` defaults to
    a.shape[0] (token-id layout); pass the flattened-slot count when sorted
    ids are token*topk+k slot indices.
    """
    E, N, K_half = w.shape
    K = K_half * 2
    M_total = a.shape[0] if sentinel is None else sentinel
    w_u8 = w.view(torch.uint8)
    # A non-float32 scale is a raw UE8M0 byte (torch.uint8, or the checkpoint's
    # native float8_e8m0fnu viewed as bytes); the kernel rebuilds 2^(byte-127).
    s_is_e8m0 = s.dtype != torch.float32
    s_arg = s.view(torch.uint8) if s_is_e8m0 else s
    if num_valid is None:
        num_valid = torch.full(
            (1,),
            sorted_token_ids.shape[0],
            dtype=torch.int32,
            device=sorted_token_ids.device,
        )
    grid = (expert_ids.shape[0] * triton.cdiv(N, block_n),)
    _mxfp4_w4a16_gemm_kernel[grid](
        a,
        w_u8,
        s_arg,
        out,
        sorted_token_ids,
        expert_ids,
        num_valid,
        M_total,
        N,
        K,
        a.stride(0),
        w_u8.stride(0),
        w_u8.stride(1),
        w_u8.stride(2),
        s_arg.stride(0),
        s_arg.stride(1),
        s_arg.stride(2),
        out.stride(0),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        SCALE_IS_E8M0=s_is_e8m0,
        num_warps=num_warps,
        num_stages=2,
    )
    return out


# ---------------------------------------------------------------------------
# SM75 hand-written PTX path (mma.sync.m16n8k8, register-side e2m1 dequant).
#
# Triton on Turing must round-trip register-computed operands through shared
# memory before mma (its hard limit; the Triton kernel above tops out at
# ~12 GB/s of weight traffic). The PTX kernel in jit/csrc/moe/
# mxfp4_w4a16_ptx.cuh feeds the tensor cores directly from registers:
# ~75 GB/s measured on a 2080 Ti (9x). It consumes a lane-major repacked
# weight layout (repack_mxfp4_for_ptx); repack once at load time.
# ---------------------------------------------------------------------------


def repack_mxfp4_for_ptx(w: torch.Tensor) -> torch.Tensor:
    """[E, N, K/2] int8-packed e2m1 -> uint32 [E, N/8, K/32, 32] lane-major.

    Byte j (j=0..3) of lane L's u32 = packed byte W[e][nt*8+gid][ks*16+j*4+c4]
    with gid = L>>2, c4 = L&3 — exactly the B-fragment byte for mma k-step j
    of warp tile (BM=16, BN=8, BK=32). Pure reshape/permute/view: no advanced
    indexing, safe on multi-GB expert stacks.
    """
    E, N, Kh = w.shape
    nt, ks = N // 8, Kh // 16
    wv = w.view(E, nt, 8, ks, 4, 4)  # [E, nt, gid, ks, j, c4]
    wv = wv.permute(0, 1, 3, 2, 5, 4).contiguous()  # [E, nt, ks, gid, c4, j]
    return wv.view(torch.uint8).view(torch.uint32).reshape(E, nt, ks, 32).contiguous()


def w4a16_repack_supported(w: torch.Tensor) -> bool:
    """Whether repack_mxfp4_inplace_ would accept `w` [E, N, K/2].

    Exposed so a caller can test every tensor it plans to repack *before*
    repacking any of them: repacking one and declining another leaves a layer
    half in each layout, which the kernels cannot tell apart.
    """
    if w.dim() != 3:
        return False
    _, n, kh = w.shape
    return n % 8 == 0 and kh % 16 == 0


def repack_mxfp4_inplace_(w: torch.Tensor) -> bool:
    """Repack [E, N, K/2] packed e2m1 into the PTX lane-major layout in place.

    Returns False (leaving `w` untouched) if the shape does not fit the warp
    tile, so the caller can keep the direct-layout kernel.

    The repacked layout is a permutation of the same E*N*K/2 bytes, and the
    permutation is expert-local, so each expert can be repacked through a
    one-expert scratch buffer and written back over itself. Repacking the whole
    stack at once would need the packed and repacked buffers alive at the same
    time -- 17.25 GiB + 17.25 GiB on a 22 GiB card -- which is exactly what made
    repacking look unaffordable. This costs one expert (~67 MiB) instead, and the
    steady-state footprint is unchanged because the byte count is unchanged.
    """
    if w.dim() != 3:
        return False
    e, n, kh = w.shape
    if n % 8 != 0 or kh % 16 != 0:
        return False

    flat = w.view(torch.uint8)
    for i in range(e):
        # One expert at a time: read the packed bytes, permute, write back over
        # the same expert. Safe because the permutation never crosses the expert
        # boundary, and the scratch holds a full copy of expert i's bytes before
        # any of them are overwritten.
        scratch = repack_mxfp4_for_ptx(w[i : i + 1])
        dst = flat[i]
        dst.view(-1).copy_(scratch.view(torch.uint8).view(-1))
    return True


_ptx_module = None
_ptx_failed = False


def get_ptx_module():
    """Load (once) the JIT-compiled PTX W4A16 module; None if unavailable."""
    global _ptx_module, _ptx_failed
    if _ptx_module is not None or _ptx_failed:
        return _ptx_module
    try:
        from sglang.kernels.jit.utils import load_jit

        _ptx_module = load_jit(
            "mxfp4_w4a16_ptx",
            cuda_files=["moe/mxfp4_w4a16_ptx.cuh"],
            cuda_wrappers=[("run", "W4A16PtxKernel::run")],
        )
    except Exception:
        _ptx_failed = True
        _ptx_module = None
    return _ptx_module


def mxfp4_w4a16_gemm_ptx(
    a: torch.Tensor,  # [M_total, K] fp16 (token-indexed activations)
    w_rep: torch.Tensor,  # repack_mxfp4_for_ptx output
    s: torch.Tensor,  # [E, N, K//32] 1-byte UE8M0 scales (torch.uint8 or
    # float8_e8m0fnu, both viewed as bytes); the kernel rebuilds 2^(byte-127)
    # exactly, so no fp32 copy of the scales has to be kept in HBM.
    sorted_token_ids: torch.Tensor,  # [num_padded] int32 (sentinel = invalid id)
    expert_ids: torch.Tensor,  # [num_m_blocks] int32 (one per 16-row block)
    out: torch.Tensor,  # [num_padded, N] fp16 slot-indexed
    sentinel: int,  # rows with id >= sentinel read zeros (pad slots)
    num_valid: Optional[torch.Tensor] = None,  # [1] int32 device scalar
) -> torch.Tensor:
    """Repacked-layout grouped GEMM.

    `num_valid` is moe_align_block_size's num_tokens_post_padded: only that many
    leading entries of sorted_token_ids/expert_ids were written, and both buffers
    are capacity-sized torch.empty allocations, so slots past it hold garbage that
    can pass the sentinel test and index the weights out of bounds.
    """
    if num_valid is None:
        num_valid = torch.full(
            (1,),
            sorted_token_ids.shape[0],
            dtype=torch.int32,
            device=sorted_token_ids.device,
        )
    mod = get_ptx_module()
    # The kernel indexes both w_rep and s with computed linear offsets, so a
    # non-contiguous tensor would silently produce wrong results.
    if not w_rep.is_contiguous():
        w_rep = w_rep.contiguous()
    if not s.is_contiguous():
        s = s.contiguous()
    if s.dtype == torch.uint8:
        s_bytes = s
    elif s.dtype == torch.float8_e8m0fnu:
        s_bytes = s.view(torch.uint8)
    else:
        raise ValueError(
            "mxfp4_w4a16_gemm_ptx needs a 1-byte UE8M0 scale (torch.uint8 or "
            f"float8_e8m0fnu); got {s.dtype}."
        )
    mod.run(
        a,
        w_rep,
        s_bytes,
        sorted_token_ids,
        expert_ids,
        out,
        num_valid,
        float(sentinel),
    )
    return out


# ---------------------------------------------------------------------------
# Direct-layout PTX path: consumes the raw packed e2m1 bytes in place (no
# repack buffer). The repacked kernel above is ~40% faster but needs a second
# full-size weight buffer, which does not fit alongside the packed experts on
# a 22 GB card (138 GiB / 8 = 17.25 GiB packed + 17.25 GiB repacked > 22 GiB).
# This variant trades that speed for zero extra VRAM and is the sub-90 default.
# ---------------------------------------------------------------------------


_ptx_direct_module = None
_ptx_direct_failed = False


def get_ptx_direct_module():
    global _ptx_direct_module, _ptx_direct_failed
    if _ptx_direct_module is not None or _ptx_direct_failed:
        return _ptx_direct_module
    try:
        from sglang.kernels.jit.utils import load_jit

        _ptx_direct_module = load_jit(
            "mxfp4_w4a16_ptx_direct",
            cuda_files=["moe/mxfp4_w4a16_ptx_direct.cuh"],
            cuda_wrappers=[("run", "W4A16PtxDirectKernel::run")],
        )
    except Exception:
        _ptx_direct_failed = True
        _ptx_direct_module = None
    return _ptx_direct_module


def mxfp4_w4a16_gemm_ptx_direct(
    a: torch.Tensor,  # [M_total, K] fp16 (token-indexed activations)
    w: torch.Tensor,  # [E, N, K//2] int8 packed e2m1 (raw checkpoint layout)
    s: torch.Tensor,  # [E, N, K//32] float32 scales
    sorted_token_ids: torch.Tensor,  # [num_padded] int32 (sentinel = invalid id)
    expert_ids: torch.Tensor,  # [num_m_blocks] int32 (one per 16-row block)
    out: torch.Tensor,  # [num_padded, N] fp16 slot-indexed
    sentinel: int,  # rows with id >= sentinel read zeros (pad slots)
    num_valid: Optional[torch.Tensor] = None,  # [1] int32 device scalar
) -> torch.Tensor:
    """`num_valid` is moe_align_block_size's num_tokens_post_padded: only that
    many leading entries of `sorted_token_ids`/`expert_ids` were written. The
    buffers are capacity-sized torch.empty allocations, so blocks past it would
    read a garbage expert id out of bounds. Defaults to "all" for tests."""
    if num_valid is None:
        num_valid = torch.full(
            (1,),
            sorted_token_ids.shape[0],
            dtype=torch.int32,
            device=sorted_token_ids.device,
        )
    mod = get_ptx_direct_module()
    # The kernel walks rows with unit inner stride and expert/n strides taken
    # from the tensor; guard full contiguity instead of silently miscomputing
    # on a transposed or sliced view.
    if not w.is_contiguous():
        w = w.contiguous()
    if not s.is_contiguous():
        s = s.contiguous()
    # The kernel reads the scale as raw UE8M0 bytes (value 2^(byte-127)).
    # Accept torch.uint8 or the checkpoint's native float8_e8m0fnu, both viewed
    # as bytes with no copy. A float32 scale is rejected rather than converted:
    # converting here would allocate inside every forward, which cannot be
    # captured in a CUDA graph. Convert once at load time instead.
    if s.dtype == torch.uint8:
        s_bytes = s
    elif s.dtype == torch.float8_e8m0fnu:
        s_bytes = s.view(torch.uint8)
    else:
        raise ValueError(
            "mxfp4_w4a16_gemm_ptx_direct needs a 1-byte UE8M0 scale "
            f"(torch.uint8 or float8_e8m0fnu); got {s.dtype}. Convert once at "
            "weight-load time; do not convert per forward."
        )
    mod.run(
        a, w, s_bytes, sorted_token_ids, expert_ids, out, num_valid, float(sentinel)
    )
    return out


# ---------------------------------------------------------------------------
# v3: activation reuse across n tiles + PRMT nibble dequant. See
# jit/csrc/moe/mxfp4_w4a16_ptx_v3.cuh for the measurements that motivated it.
# ---------------------------------------------------------------------------


_ptx_v3_module = None
_ptx_v3_failed = False


def get_ptx_v3_module():
    """Load (once) the v3 W4A16 module; None if the JIT build is unavailable."""
    global _ptx_v3_module, _ptx_v3_failed
    if _ptx_v3_module is not None or _ptx_v3_failed:
        return _ptx_v3_module
    try:
        from sglang.kernels.jit.utils import load_jit

        _ptx_v3_module = load_jit(
            "mxfp4_w4a16_ptx_v3",
            cuda_files=["moe/mxfp4_w4a16_ptx_v3.cuh"],
            cuda_wrappers=[
                ("run", "W4A16PtxV3Kernel::run"),
                ("reduce", "W4A16PtxV3Reduce::run"),
            ],
        )
    except Exception:
        _ptx_v3_failed = True
        _ptx_v3_module = None
    return _ptx_v3_module


# cfg -> (n tiles per warp, PRMT dequant on/off)
W4A16_V3_CFGS = {
    0: (1, False),
    1: (1, True),
    2: (2, True),
    3: (4, True),
    4: (2, False),
    5: (4, False),
    6: (8, True),
}


# cfg -> k splits across grid.z. 1 for the shapes the kernel has always had.
_KS_CFG = {7: 2, 8: 4, 9: 4, 10: 8, 11: 8, 12: 8, 13: 2}
# cfg -> the same (NT, PRMT) shape with the split removed, to fall back to.
_KS_UNFOLD = {7: 3, 8: 3, 9: 2, 10: 3, 11: 2, 12: 1, 13: 2}
# Splitting k only pays while the grid is too small to fill the card. Measured on a
# 2080 Ti at the production gemm1 shape (n=2048, k=4096, NT=4 -> grid.x=64) on TIDY
# buffers: 1.20x at 384 blocks, 1.01x at 448, 1.00x at 512, 0.92x at 640, 0.85x at 1024.
_KS_MAX_BLOCKS = 448
# At the REAL buffer shapes the split is worth nothing, and the reason is the partial
# buffer. sorted_ids/expert_ids are the align buffer's CAPACITY (num_tokens*topk +
# (E+1)*(block_m-1) = 3861 rows at decode) rather than the 6 live rows, because
# num_valid lives on the device to stay graph-capturable. So the fp32 partial is
# KS x 3861 x 2048 x 4 = 63 MB per split, and summing it costs as much as the GEMM:
# measured KS1 0.115 ms, KS2 0.116 ms, KS4 0.116 ms. The cfgs stay available for
# sweeps and for a caller that sizes its buffers to live rows, but production does not
# request them -- see SGLANG_SM75_W4A16_KSPLIT.
# At a single-token batch the card is not full, so spend the occupancy budget on more
# blocks rather than on activation reuse. Measured at the real bs=1 shapes on a 2080 Ti
# (grid 64 x 6 = 384 blocks on 68 SMs, 5.6 warps per SM), median of 7 interleaved rounds:
#
#   gemm1  NT4 KS1 0.116   NT2 KS1 0.098   NT4 KS2 0.087   NT4 KS4 0.090 ms
#   gemm2  NT4 KS1 0.060   NT2 KS1 0.054   NT4 KS2 0.048   NT4 KS4 0.053 ms
#
# NT4 KS2 is the best, and it is what production already asks for when
# SGLANG_SM75_W4A16_KSPLIT=2, so this table only matters when the split is switched off:
# a single-token batch then still gets NT=2's 1.18x/1.11x instead of nothing. The two
# levers do not stack -- NT2 KS2 measures 0.107/0.063, worse than either alone, because
# NT=2 gives back the activation reuse that makes the wider grid worth having.
_SMALL_BATCH_CFG = {3: 2}  # NT4 KS1 -> NT2 KS1 when the batch is a single token


def mxfp4_w4a16_gemm_ptx_v3(
    a: torch.Tensor,
    w_rep: torch.Tensor,
    s: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    out: torch.Tensor,
    sentinel: int,
    num_valid: Optional[torch.Tensor] = None,
    cfg: int = 3,
    num_tokens: Optional[int] = None,
) -> torch.Tensor:
    """Repacked-layout grouped GEMM, v3 (activation reuse + PRMT dequant).

    Same inputs and same results as mxfp4_w4a16_gemm_ptx; `cfg` picks the
    (n-tiles-per-warp, PRMT, k-splits) triple so one build can be swept.

    `num_tokens`, when the caller supplies it, lets a single-token batch use fewer
    n-tiles per warp. NT is an occupancy trade: NT=4 halves grid.x and reuses each
    m16n8k8 activation fragment across two tiles, which pays once the card is full,
    while NT=2 doubles grid.x and wins when it is not. Measured at the real decode
    shapes on a 2080 Ti (6 live experts, grid.y = 241 capacity blocks):

        gemm1  NT1 0.144  NT2 0.095  NT4 0.116 ms     NT2 best
        gemm2  NT1 0.080  NT2 0.050  NT4 0.053 ms     NT2 best
        gemm1 at 2 tokens (12 experts)      NT2 0.183  NT4 0.152 ms   NT4 best
        gemm2 at 2 tokens                   NT2 0.093  NT4 0.069 ms   NT4 best

    The crossover is between 1 and 2 tokens, so the rule is num_tokens == 1. It is
    decided in Python from a value that is fixed per CUDA graph capture, so each
    captured batch size bakes its own choice and nothing synchronizes.
    """
    if num_valid is None:
        num_valid = torch.full(
            (1,),
            sorted_token_ids.shape[0],
            dtype=torch.int32,
            device=sorted_token_ids.device,
        )
    mod = get_ptx_v3_module()
    if mod is None:
        raise RuntimeError("mxfp4_w4a16_ptx_v3 module unavailable")
    if not w_rep.is_contiguous():
        w_rep = w_rep.contiguous()
    if s.dtype == torch.float8_e8m0fnu:
        s = s.view(torch.uint8)
    elif s.dtype != torch.uint8:
        raise ValueError("v3 needs a 1-byte UE8M0 scale; got %s" % s.dtype)
    if not s.is_contiguous():
        s = s.contiguous()

    if num_tokens == 1 and cfg in _SMALL_BATCH_CFG:
        cfg = _SMALL_BATCH_CFG[cfg]

    ks = _KS_CFG.get(cfg, 1)
    if ks > 1:
        n_tiles = s.shape[1] // 8
        # NT per split cfg, matching the GO() table in the .cuh: cfg 12 is NT1,
        # cfgs 9/11/13 are NT2, cfgs 7/8/10 are NT4. cfg 2 never reaches this branch
        # because it is not a split cfg.
        nt = 1 if cfg == 12 else (2 if cfg in (9, 11, 13) else 4)
        blocks = ((n_tiles + nt - 1) // nt) * expert_ids.shape[0]
        if blocks > _KS_MAX_BLOCKS:
            cfg, ks = _KS_UNFOLD[cfg], 1
    if ks == 1:
        mod.run(
            a,
            w_rep,
            s,
            sorted_token_ids,
            expert_ids,
            out,
            num_valid,
            float(sentinel),
            float(cfg),
        )
        return out

    n = s.shape[1]
    # torch.empty, matching what the KS=1 path already tolerates: the kernel writes
    # every (slot, n) inside the blocks it owns, and the rows it can leave unwritten are
    # those past num_valid, which the MoE epilogue never reads -- it gathers real slots
    # out of this buffer by sorted id. The reduce is elementwise, so an unwritten row
    # cannot contaminate a written one.
    part = torch.empty(
        (ks, sorted_token_ids.shape[0], n), dtype=torch.float32, device=out.device
    )
    mod.run(
        a,
        w_rep,
        s,
        sorted_token_ids,
        expert_ids,
        part,
        num_valid,
        float(sentinel),
        float(cfg),
    )
    # The fused reduce reads num_valid on the device and combines only the live rows.
    # torch.sum over dim 0 would touch the buffer's full capacity -- 3861 rows at
    # decode against 6 live ones -- and that 63 MB of extra traffic is exactly what
    # made the split measure 1.00x in production where it measures 1.20x on tidy
    # buffers. Older modules built without the reduce entry fall back to torch.sum.
    reduce = getattr(mod, "reduce", None)
    if reduce is not None:
        reduce(part, out, num_valid, float(ks))
    else:
        torch.sum(part, dim=0, out=out.view(sorted_token_ids.shape[0], n))
    return out
