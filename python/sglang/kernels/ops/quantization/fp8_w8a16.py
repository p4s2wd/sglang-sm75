"""W8A16: block-scaled FP8 weights read straight from HBM, dequantized in
registers, multiplied on the fp16 tensor cores.

Why this exists. On sub-80 GPUs (no FP8 tensor cores) the FP8 path used to widen
every dense linear to fp16 at load time. A batch-1 decode reads every weight once
per token and is purely bandwidth bound, so that choice costs a full extra byte
per weight per token: on DeepSeek-V4-Flash at TP2 it is 8.79 GiB per card per
token, 17.6 ms at the ~500 GB/s these cards actually reach. It also holds the
fp16 copy in HBM, which is the memory a long context needs for its KV cache.
Keeping the payload as FP8 halves both. Measured on the real server: the KV pool
went from 11,776 tokens to 397,056.

Why not just call cuBLAS. There is no fp8 x fp16 GEMM in the libraries these
architectures ship, which is exactly why the loader widened the weights. So the
dequant has to happen inside the multiply. Three constraints of this architecture
shaped the implementation, and all three were measured rather than assumed:

- Triton refuses to load an fp8 dtype on SM75 ("type fp8e4nv not supported in
  this architecture"). The payload is therefore loaded as uint8 and converted
  through a 256-entry table. The table is torch's own fp8->fp16 conversion for
  every possible byte, so it is exact; 512 bytes stays resident in L1, so the
  conversion is an L1 hit rather than a memory access. Widening with integer
  shifts instead was measured 30x slower, because it runs ~5 integer ops per byte
  and at these bandwidths the SM's integer throughput, not memory, becomes the
  limit.

- The weight must be walked with the reduction dimension contiguous. Reading
  [K, N] makes each thread stride by N and measured 8-16 GB/s; [N, K] (the
  F.linear convention, which is how the checkpoint already stores it) reaches
  483-495 GB/s.

- tl.dot is a trap at decode batch sizes. Computing out[m,n] = x @ W^T from a
  W stored [N, K] needs the dot's second operand as [block_k, block_n], so the
  dequantized tile has to be transposed every iteration; that layout conversion
  measured 24 GB/s. Reformulating as out^T = W @ x^T keeps both operands native
  but forces a transposed store and measured 0.2 GB/s at 512 rows. Neither beat
  cuBLAS. So this module only uses tl.dot nowhere: small batches use an explicit
  multiply-and-reduce, and large batches dequantize into a small scratch and let
  cuBLAS have it.

Scale layout matches the checkpoint: weight_block_size [128, 128], so
weight_scale_inv[n_blk, k_blk] applies to weight[n_blk*128:(n_blk+1)*128,
k_blk*128:(k_blk+1)*128]. With block_k = 128 a tile sees exactly one scale, so
the dequant is a scalar multiply. Ragged trailing blocks are covered by ceil() in
the scale grid and masked here.
"""

from typing import Dict, Optional, Tuple

import torch
import triton
import triton.language as tl

from sglang.srt.environ import envs

BLOCK_SCALE = 128

# Above this many activation rows the projection stops being bandwidth bound and
# cuBLAS's tensor cores win: measured, the explicit reduce reaches 1.76x at two
# rows and 1.08x at four, and past that the dequant-then-cuBLAS path is ahead.
# The decode CUDA graph captures bs<=2, so decode always takes the kernel.
_GEMV_MAX_ROWS = 4

_LUT_CACHE: Dict[Tuple[torch.device, torch.dtype], torch.Tensor] = {}


def fp8_payload_lut(device: torch.device, out_dtype: torch.dtype) -> torch.Tensor:
    """Exact e4m3fn -> `out_dtype` table, one entry per possible payload byte."""
    key = (device, out_dtype)
    lut = _LUT_CACHE.get(key)
    if lut is None:
        codes = torch.arange(256, dtype=torch.uint8, device=device)
        lut = codes.view(torch.float8_e4m3fn).to(out_dtype).contiguous()
        _LUT_CACHE[key] = lut
    return lut


@triton.jit
def _e4m3_to_f16(b):
    """Exact e4m3fn byte -> fp16, built with integer ops instead of a table.

    The LUT the kernels used to hold this table costs more than the decode it
    saves: one gather instruction hits up to 32 different L1 sectors per warp,
    and that L1 traffic serializes against the weight stream. Measured on a
    4096x4096 tile with everything else in place, removing only the LUT lifts
    the GEMV from 283 to 450 GB/s.

    For a normal e4m3 the fp16 bits are the exponent field shifted for the
    bias change (7 -> 15, so +8) with the mantissa seven bits up:
    `(em << 7) + (8 << 10)`, one shift and one add, and the addition cannot
    carry into the mantissa because 8 fits in the exponent field alone
    (e4m3fn is finite, its largest exponent 15 plus 8 stays below fp16
    inf). Subnormals (em < 8) are m * 2^-9, a three-bit integer times a
    power of two, exact in either float format.
    """
    u = b.to(tl.int32) & 0xFF
    em = u & 0x7F
    bits = (em << 7) + (8 << 10)
    v = bits.to(tl.int16).to(tl.float16, bitcast=True)
    sub = (u & 0x7).to(tl.float32) * 0.001953125
    v = tl.where(em < 8, sub.to(tl.float16), v)
    return tl.where(u < 0x80, v, -v)


@triton.jit
def _w8a16_gemv_kernel(
    x_ptr,
    w_ptr,
    s_ptr,
    lut_ptr,
    out_ptr,
    n,
    k,
    stride_xm,
    stride_wn,
    stride_wk,
    stride_sn,
    stride_sk,
    stride_om,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
    group_size: tl.constexpr,
    rows: tl.constexpr,
    even_k: tl.constexpr,
    alu: tl.constexpr,
):
    """out [rows, N] = x [rows, K] @ W^T, with W stored [N, K] so k is contiguous.

    One program owns a block of output features and reads that block of W once,
    multiplying it into `rows` accumulators that are reduced after the loop.
    Reducing inside the loop measured 2x slower, and a 3D broadcast-reduce
    measured worse still.

    Two structural details are load-bearing, both measured: `w_deq` stays in the
    activation dtype until the multiply (widening the tile to fp32 first doubles
    its registers and cost 2.4x), and the k mask is an if/else on `even_k` rather
    than a ternary (Triton rejects `mask=None if cond else ...`, and an
    always-on mask costs the same widening).
    """
    pid = tl.program_id(0)
    offs_n = pid * block_n + tl.arange(0, block_n)
    offs_k = tl.arange(0, block_k)
    n_mask = offs_n < n

    acc0 = tl.zeros((block_n, block_k), dtype=tl.float32)
    if rows >= 2:
        acc1 = tl.zeros((block_n, block_k), dtype=tl.float32)
    if rows >= 3:
        acc2 = tl.zeros((block_n, block_k), dtype=tl.float32)
    if rows >= 4:
        acc3 = tl.zeros((block_n, block_k), dtype=tl.float32)

    w_ptrs = w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk
    n_blk = (pid * block_n) // group_size

    for k_start in range(0, k, block_k):
        k_blk = k_start // group_size
        scale = tl.load(s_ptr + n_blk * stride_sn + k_blk * stride_sk)

        if even_k:
            w_byte = tl.load(w_ptrs, mask=n_mask[:, None], other=0)
        else:
            k_mask = k_start + offs_k < k
            w_byte = tl.load(w_ptrs, mask=n_mask[:, None] & k_mask[None, :], other=0)
        if alu:
            w_deq = _e4m3_to_f16(w_byte) * scale.to(tl.float16)
        else:
            w_deq = tl.load(lut_ptr + w_byte) * scale.to(lut_ptr.dtype.element_ty)

        if even_k:
            xv = tl.load(x_ptr + k_start + offs_k)
        else:
            xv = tl.load(x_ptr + k_start + offs_k, mask=k_start + offs_k < k, other=0.0)
        acc0 += w_deq.to(tl.float32) * xv[None, :].to(tl.float32)

        if rows >= 2:
            if even_k:
                xv = tl.load(x_ptr + stride_xm + k_start + offs_k)
            else:
                xv = tl.load(
                    x_ptr + stride_xm + k_start + offs_k,
                    mask=k_start + offs_k < k,
                    other=0.0,
                )
            acc1 += w_deq.to(tl.float32) * xv[None, :].to(tl.float32)
        if rows >= 3:
            if even_k:
                xv = tl.load(x_ptr + 2 * stride_xm + k_start + offs_k)
            else:
                xv = tl.load(
                    x_ptr + 2 * stride_xm + k_start + offs_k,
                    mask=k_start + offs_k < k,
                    other=0.0,
                )
            acc2 += w_deq.to(tl.float32) * xv[None, :].to(tl.float32)
        if rows >= 4:
            if even_k:
                xv = tl.load(x_ptr + 3 * stride_xm + k_start + offs_k)
            else:
                xv = tl.load(
                    x_ptr + 3 * stride_xm + k_start + offs_k,
                    mask=k_start + offs_k < k,
                    other=0.0,
                )
            acc3 += w_deq.to(tl.float32) * xv[None, :].to(tl.float32)

        w_ptrs += block_k * stride_wk

    tl.store(
        out_ptr + offs_n, tl.sum(acc0, axis=1).to(out_ptr.dtype.element_ty), mask=n_mask
    )
    if rows >= 2:
        tl.store(
            out_ptr + stride_om + offs_n,
            tl.sum(acc1, axis=1).to(out_ptr.dtype.element_ty),
            mask=n_mask,
        )
    if rows >= 3:
        tl.store(
            out_ptr + 2 * stride_om + offs_n,
            tl.sum(acc2, axis=1).to(out_ptr.dtype.element_ty),
            mask=n_mask,
        )
    if rows >= 4:
        tl.store(
            out_ptr + 3 * stride_om + offs_n,
            tl.sum(acc3, axis=1).to(out_ptr.dtype.element_ty),
            mask=n_mask,
        )


# Set False to force the narrow (block_k = scale group) kernel everywhere, for A/B on
# real weights without editing the dispatch condition.
_W8A16_WIDE = True

# Two conditions, both mechanistic, and together they separate the winners from the
# losers on all eight real projections at both row counts (see the kernel docstring).
#
# k >= 2048: block_k is the largest power of two dividing k, capped at 2048, so below
# k=1024 the loop runs a single iteration -- no read left to lengthen, only the extra
# register pressure. k=1024 is borderline and row-count dependent: it measured 1.37x at
# one row and 0.83x at two (down, n=7168 k=1024), so it is left out.
#
# grid <= 2048: the wide kernel runs one program per block_n=4 rows, so its grid is n/4.
# kv_b (n=32768) and q_b (n=24576) put 8192 and 6144 programs on 68 SMs, already 90-120
# waves deep; there is no per-program memory latency left to hide, so only the cost
# remains. The largest winning grid is 1792, the smallest losing one 6144.
_WIDE_MIN_K = 2048
_WIDE_MAX_GRID = 2048


def _should_use_w8a16_wide(
    rows: int, n: int, k: int, allow_k1024: bool = False
) -> bool:
    return (
        _W8A16_WIDE
        and rows <= 2
        and (k >= _WIDE_MIN_K or (allow_k1024 and k == 1024))
        and triton.cdiv(n, 4) <= _WIDE_MAX_GRID
    )


@triton.jit
def _w8a16_gemv_wide_kernel(
    x_ptr,
    w_ptr,
    s_ptr,
    lut_ptr,
    out_ptr,
    n,
    k,
    stride_xm,
    stride_wn,
    stride_sn,
    stride_sk,
    stride_om,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
    group_size: tl.constexpr,
    rows: tl.constexpr,
    alu: tl.constexpr,
):
    """Same GEMV, reading block_k (a multiple of group_size) bytes per row per step.

    The narrow kernel pins block_k to the 128-byte scale group, so each program reads
    128 contiguous bytes per row per iteration. A read sweep at the model's largest
    projection (o_proj, n=7168 k=32768) shows what that costs: 363 GB/s at block_k=128
    against 574-579 GB/s at block_k 512-1024, and the ablation says the dequant math is
    only 14% of the narrow kernel's time -- the read shape is the whole gap.

    Lifting block_k means one step spans several scale groups, so the scale becomes a
    second inner dimension: the weight tile is (block_n, n_groups, group_size) and the
    scale is (n_groups,). Measured per call at the sustained clock, over the eight real
    projections at TP2, with the gate applied:

        rows=1  1211.6 -> 846.5 us  1.43x     rows=2  1201.8 -> 954.6 us  1.26x
        rows=4  1114.6 -> 1209.8 us 0.92x

    Hence the caller keeps the narrow kernel from three rows up. The regression is
    register pressure, not arithmetic: the accumulators are (block_n, block_k) fp32, so
    at block_k=1024 one accumulator is already 64 registers per thread with four warps,
    and four of them spill.

    The per-element arithmetic is the narrow kernel's: LUT lookup, times the scale cast
    to the LUT's dtype, then widen and multiply by the activation. Widening before the
    scale multiply changes the bits.

    The reduction order is not the same, because the k loop now strides by block_k
    instead of by one scale group, so the two are not bit-identical. Measured against an
    fp64 reference on real weights (attn.wo_b n=4096 k=8192, attn.wq_a n=1024 k=4096),
    both land at the same maximum error -- 4.58e-04 and 4.66e-04 at one row, 4.84e-04 and
    4.25e-04 at two -- so the difference is summation order, not lost precision. Shapes
    where block_k ends up equal to the narrow kernel's step are bit-identical.
    """
    pid = tl.program_id(0)
    offs_n = pid * block_n + tl.arange(0, block_n)
    offs_k = tl.arange(0, block_k)
    n_mask = offs_n < n
    n_groups: tl.constexpr = block_k // group_size
    offs_g = tl.arange(0, n_groups)

    acc0 = tl.zeros((block_n, block_k), dtype=tl.float32)
    if rows >= 2:
        acc1 = tl.zeros((block_n, block_k), dtype=tl.float32)

    w_ptrs = w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :]
    n_blk = (pid * block_n) // group_size
    # One scale per group_size columns, so a block_k-wide step reads n_groups of them.
    s_ptrs = s_ptr + n_blk * stride_sn + offs_g * stride_sk

    for k_start in range(0, k, block_k):
        w_byte = tl.load(w_ptrs, mask=n_mask[:, None], other=0)
        if alu:
            scale = tl.load(s_ptrs).to(tl.float16)
            w3 = tl.reshape(_e4m3_to_f16(w_byte), (block_n, n_groups, group_size))
        else:
            scale = tl.load(s_ptrs).to(lut_ptr.dtype.element_ty)
            w3 = tl.reshape(tl.load(lut_ptr + w_byte), (block_n, n_groups, group_size))
        # Reshape the dequantized tile to (block_n, n_groups, group_size) first, so the
        # per-group scale broadcasts along the middle axis only. Multiplying it against
        # the flat (block_n, block_k) tile would line the scales up with the wrong
        # columns and nothing would complain.
        w_deq = (w3 * scale[None, :, None]).to(tl.float32)
        xv = tl.load(x_ptr + k_start + offs_k).to(tl.float32)
        acc0 += tl.reshape(
            w_deq * xv.reshape((1, n_groups, group_size)), (block_n, block_k)
        )
        if rows >= 2:
            xv = tl.load(x_ptr + stride_xm + k_start + offs_k).to(tl.float32)
            acc1 += tl.reshape(
                w_deq * xv.reshape((1, n_groups, group_size)), (block_n, block_k)
            )
        w_ptrs += block_k
        s_ptrs += (block_k // group_size) * stride_sk

    tl.store(
        out_ptr + offs_n, tl.sum(acc0, axis=1).to(out_ptr.dtype.element_ty), mask=n_mask
    )
    if rows >= 2:
        tl.store(
            out_ptr + stride_om + offs_n,
            tl.sum(acc1, axis=1).to(out_ptr.dtype.element_ty),
            mask=n_mask,
        )


@triton.jit
def _dequant_block_fp8_kernel(
    w_ptr,
    s_ptr,
    lut_ptr,
    out_ptr,
    n,
    k,
    stride_wn,
    stride_wk,
    stride_sn,
    stride_sk,
    stride_on,
    stride_ok,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
    group_size: tl.constexpr,
):
    """Widen a [block_n, block_k] slice of the weight, applying its block scale."""
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_n = pid_n * block_n + tl.arange(0, block_n)
    offs_k = pid_k * block_k + tl.arange(0, block_k)
    n_mask = offs_n < n
    k_mask = offs_k < k

    scale = tl.load(
        s_ptr
        + ((pid_n * block_n) // group_size) * stride_sn
        + ((pid_k * block_k) // group_size) * stride_sk
    )
    w_byte = tl.load(
        w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk,
        mask=n_mask[:, None] & k_mask[None, :],
        other=0,
    )
    w_deq = tl.load(lut_ptr + w_byte) * scale.to(lut_ptr.dtype.element_ty)
    tl.store(
        out_ptr + offs_n[:, None] * stride_on + offs_k[None, :] * stride_ok,
        w_deq,
        mask=n_mask[:, None] & k_mask[None, :],
    )


def dequant_block_fp8_slice(
    weight_nk: torch.Tensor,
    weight_scale: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    """Widen weight_nk [n_slice, K] into `out`, applying the block scales."""
    n, k = weight_nk.shape
    w_bytes = (
        weight_nk if weight_nk.dtype == torch.uint8 else weight_nk.view(torch.uint8)
    )
    scale = (
        weight_scale if weight_scale.dtype == torch.float32 else weight_scale.float()
    )
    _dequant_block_fp8_kernel[(triton.cdiv(n, 64), triton.cdiv(k, 128))](
        w_bytes,
        scale,
        fp8_payload_lut(out.device, out.dtype),
        out,
        n,
        k,
        w_bytes.stride(0),
        w_bytes.stride(1),
        scale.stride(0),
        scale.stride(1),
        out.stride(0),
        out.stride(1),
        block_n=64,
        block_k=128,
        group_size=BLOCK_SCALE,
        num_warps=4,
        num_stages=3,
    )
    return out


def _dequant_scratch_rows(k: int, budget_bytes: int = 8 << 20) -> int:
    """Weight rows that fit the scratch budget, as a multiple of the block size.

    The tile has to stay small: with --mem-fraction-static at 0.97 the KV pool
    takes everything the weights leave, so a scratch sized to a whole weight
    (16 MiB per projection here) would be allocated out of the same headroom the
    KV pool is sitting on.

    It also has to be a multiple of BLOCK_SCALE. The dequant kernel derives a
    tile's scale from its absolute row, so a slice that starts mid-block would
    index the wrong scale and silently produce wrong numbers.
    """
    rows = budget_bytes // (k * 2)
    rows = (rows // BLOCK_SCALE) * BLOCK_SCALE
    return max(BLOCK_SCALE, min(4096, rows))


def w8a16_linear(
    x: torch.Tensor,
    weight_nk: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """F.linear(x, W) with W kept as block-scaled e4m3fn bytes.

    x is fp16/bf16 [.., K]; weight_nk is [N, K] in the F.linear convention the
    checkpoint already uses; weight_scale is fp32 [ceil(N/128), ceil(K/128)].

    The body runs under no_grad. With --cpu-offload-gb the offload machinery hands
    this kernel a tensor that requires grad, and torch.mm with out= then refuses to
    write ("functions with out=... arguments don't support automatic
    differentiation"). This is an inference-only dequant GEMM with no gradient to
    build, so autograd tracking is pure liability.
    """
    with torch.no_grad():
        return _w8a16_linear_impl(x, weight_nk, weight_scale, bias)


def _w8a16_linear_impl(
    x: torch.Tensor,
    weight_nk: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    x2 = x.reshape(-1, x.shape[-1])
    m, k = x2.shape
    n, k_w = weight_nk.shape
    assert k_w == k, f"K mismatch {weight_nk.shape} vs activation {x.shape}"

    w_bytes = (
        weight_nk if weight_nk.dtype == torch.uint8 else weight_nk.view(torch.uint8)
    )
    scale = weight_scale
    if scale.dtype != torch.float32:
        scale = scale.float()
    if not scale.is_contiguous():
        scale = scale.contiguous()

    even_k = k % BLOCK_SCALE == 0
    lut = fp8_payload_lut(x.device, x.dtype)
    alu_deq = envs.SGLANG_SM75_W8A16_ALU_DECODE.get()
    out = torch.empty((m, n), dtype=x.dtype, device=x.device)

    if m <= _GEMV_MAX_ROWS:
        # Tuned on a 2080 Ti (68 SMs) across the model's real projections: the
        # fastest tile is the one that puts ~256 programs on the grid, so the
        # tail of a wave is hidden by the next one's loads. Measured 483 GB/s at
        # 16384x4096 (block_n 64) and 495 GB/s at 4096x16384 (block_n 16),
        # against 569 GB/s for cuBLAS on the fp16 copy.
        #
        # block_n also has to keep the accumulators in registers:
        # rows * block_n * block_k * 4 bytes, so it halves as rows doubles.
        block_n = 4
        while block_n < 64 and triton.cdiv(n, block_n) > 256:
            block_n *= 2
        # Round the register-pressure cap down to a power of two: tl.arange
        # rejects a non-power-of-two range, and 64 // m is not one for most m --
        # m=3 alone gave 21, which compiled only as long as no batch ever had
        # three rows, i.e. until two requests shared a decode step.
        cap = triton.next_power_of_2(64 // m)
        if cap > 64 // m:
            cap //= 2
        block_n = max(4, min(block_n, cap))
        num_warps = 4 if block_n >= 32 else 8

        # Wide-read path. Reading block_k bytes per row instead of one 128-byte scale
        # group is worth 1.52x at one row and 1.41x at two across the model's eight real
        # projections at TP2 (981.8 -> 644.0 us), because the narrow kernel's read shape
        # caps it at 363 GB/s where the same loads stream at 574. Measured per call at
        # the sustained clock, across the model's eight real projections at TP2:
        #   rows=1  1211.6 -> 846.5 us  1.43x      rows=2  1201.8 -> 954.6 us  1.26x
        # per-projection 1.37x-1.64x at one row, 1.12x-1.71x at two. At o_proj it lifts
        # 329 GB/s to 486, which is 86% of the same card's read-only ceiling (565 GB/s),
        # so there is little left here. From three rows it regresses (0.92x at four): the
        # accumulators are (block_n, block_k) fp32, so block_k=1024 puts 64 registers per
        # thread into each one and they spill.
        #
        # Two divisibility requirements the narrow kernel does not have: block_k must
        # divide k, because these loads are unmasked, and block_n must divide the
        # 128-row scale group, because the scale is indexed by a single group per tile.
        if _should_use_w8a16_wide(
            m,
            n,
            k,
            allow_k1024=envs.SGLANG_OPT_W8A16_WIDE_M1_K1024.get(),
        ):
            # See _WIDE_MIN_K / _WIDE_MAX_GRID for where these two numbers come from.
            block_k = 2048
            while block_k > BLOCK_SCALE and k % block_k != 0:
                block_k //= 2
            if block_k > BLOCK_SCALE:
                _w8a16_gemv_wide_kernel[(triton.cdiv(n, 4),)](
                    x2,
                    w_bytes,
                    scale,
                    lut,
                    out,
                    n,
                    k,
                    x2.stride(0),
                    w_bytes.stride(0),
                    scale.stride(0),
                    scale.stride(1),
                    out.stride(0),
                    block_n=4,
                    block_k=block_k,
                    group_size=BLOCK_SCALE,
                    rows=m,
                    alu=alu_deq,
                    num_warps=4,
                    num_stages=4,
                )
                return out
        _w8a16_gemv_kernel[(triton.cdiv(n, block_n),)](
            x2,
            w_bytes,
            scale,
            lut,
            out,
            n,
            k,
            x2.stride(0),
            w_bytes.stride(0),
            w_bytes.stride(1),
            scale.stride(0),
            scale.stride(1),
            out.stride(0),
            block_n=block_n,
            block_k=BLOCK_SCALE,
            group_size=BLOCK_SCALE,
            rows=m,
            even_k=even_k,
            # The narrow kernel's block_n=64 tiles at the deep-grid shapes are
            # issue-bound, and the decode's integer ops lose to the gather
            # there (0.80x at q_b, 0.59x at the LM head shard). The wide
            # kernel's latency-bound tiles have the issue slots free.
            alu=False,
            num_warps=num_warps,
            num_stages=4,
        )
    else:
        # Prefill: dequantize a tile of the weight and let cuBLAS have it. At
        # these row counts the projection is compute bound, so the tensor cores
        # matter more than the extra bytes the fp16 tile moves, and no
        # hand-written dot beat cuBLAS (see the module docstring). Prefill runs
        # eager, so the loop and the scratch allocation are not captured.
        rows = _dequant_scratch_rows(k)
        scratch = torch.empty((rows, k), dtype=x.dtype, device=x.device)
        for start in range(0, n, rows):
            stop = min(start + rows, n)
            piece = scratch[: stop - start]
            dequant_block_fp8_slice(
                w_bytes[start:stop], scale[start // BLOCK_SCALE :], piece
            )
            torch.mm(x2, piece.t(), out=out[:, start:stop])

    if bias is not None:
        out = out + bias
    return out.view(*x.shape[:-1], n)


@triton.jit
def _wo_a_absorb_kernel(
    o_ptr,  # [T, G, D] fp16, attention output before the absorb projection
    w_ptr,  # [G, R, D] fp16, one low-rank output projection per group
    out_ptr,  # [T, G, R] fp16
    T,
    stride_ot,
    stride_og,
    stride_wg,
    stride_wr,
    stride_outt,
    stride_outg,
    R: tl.constexpr,
    D,
    BLOCK_R: tl.constexpr,
    BLOCK_D: tl.constexpr,
    ROWS: tl.constexpr,
):
    """out[t, g, r] = sum_d o[t, g, d] * W[g, r, d], for every group at once.

    Replaces torch.einsum("tgd,grd->tgr", ...). Measured on a 2080 Ti at the real
    shapes (G=4 local groups, R=1024, D=4096), the weight read is 33.5 MB either way:

        T=1  einsum 61.3 us (548 GB/s)   T>=2  einsum 199.6 us (168 GB/s)

    At one token the einsum is already at the card's read-only ceiling, so there is
    nothing to win there and this kernel only has to not lose. From two tokens up it
    collapses to 168 GB/s and stays flat through T=8, which is a cuBLAS algorithm
    choice rather than work -- the same shape run as one GEMM per group measures 93 us.

    Accumulation is fp32 and stays that way. The per-group-GEMM alternative is fast for
    the same reason this is, but under PyTorch's default
    allow_fp16_reduced_precision_reduction it accumulates in fp16 and its error against
    an fp32 reference goes from 6e-4 to 3.6e-2. Forcing the flag off restores the
    accuracy and keeps the speed, which is what confirms the fp16 accumulator was the
    whole difference; this kernel simply never has the option.
    """
    pid = tl.program_id(0)
    n_blocks = tl.cdiv(R, BLOCK_R)
    g = pid // n_blocks
    r0 = (pid % n_blocks) * BLOCK_R

    offs_r = r0 + tl.arange(0, BLOCK_R)
    r_mask = offs_r < R
    offs_d = tl.arange(0, BLOCK_D)

    wp = w_ptr + g * stride_wg + offs_r[:, None] * stride_wr + offs_d[None, :]

    acc0 = tl.zeros((BLOCK_R, BLOCK_D), dtype=tl.float32)
    if ROWS >= 2:
        acc1 = tl.zeros((BLOCK_R, BLOCK_D), dtype=tl.float32)
    if ROWS >= 3:
        acc2 = tl.zeros((BLOCK_R, BLOCK_D), dtype=tl.float32)
    if ROWS >= 4:
        acc3 = tl.zeros((BLOCK_R, BLOCK_D), dtype=tl.float32)

    for d0 in range(0, D, BLOCK_D):
        dm = d0 + offs_d < D
        # Widen before the multiply. w * a0 with both fp16 forms the product in fp16 and
        # only then adds it into the fp32 accumulator, which measured 3.3e-2 relative
        # error against an fp32 reference where einsum gives 4.8e-4 -- the fp32
        # accumulator buys nothing if its addends are already rounded.
        w = tl.load(wp, mask=r_mask[:, None] & dm[None, :], other=0.0).to(tl.float32)
        a0 = tl.load(
            o_ptr + 0 * stride_ot + g * stride_og + d0 + offs_d, mask=dm, other=0.0
        ).to(tl.float32)
        acc0 += w * a0[None, :]
        if ROWS >= 2:
            a1 = tl.load(
                o_ptr + 1 * stride_ot + g * stride_og + d0 + offs_d, mask=dm, other=0.0
            ).to(tl.float32)
            acc1 += w * a1[None, :]
        if ROWS >= 3:
            a2 = tl.load(
                o_ptr + 2 * stride_ot + g * stride_og + d0 + offs_d, mask=dm, other=0.0
            ).to(tl.float32)
            acc2 += w * a2[None, :]
        if ROWS >= 4:
            a3 = tl.load(
                o_ptr + 3 * stride_ot + g * stride_og + d0 + offs_d, mask=dm, other=0.0
            ).to(tl.float32)
            acc3 += w * a3[None, :]
        wp += BLOCK_D

    op = out_ptr + g * stride_outg + offs_r
    rmask2 = offs_r < R
    tl.store(
        op + 0 * stride_outt,
        tl.sum(acc0, axis=1).to(out_ptr.dtype.element_ty),
        mask=rmask2,
    )
    if ROWS >= 2:
        tl.store(
            op + 1 * stride_outt,
            tl.sum(acc1, axis=1).to(out_ptr.dtype.element_ty),
            mask=rmask2,
        )
    if ROWS >= 3:
        tl.store(
            op + 2 * stride_outt,
            tl.sum(acc2, axis=1).to(out_ptr.dtype.element_ty),
            mask=rmask2,
        )
    if ROWS >= 4:
        tl.store(
            op + 3 * stride_outt,
            tl.sum(acc3, axis=1).to(out_ptr.dtype.element_ty),
            mask=rmask2,
        )


def _wo_a_absorb_chunk(o, w, out, t0, rows):
    G, R, D = w.shape
    BLOCK_D = 128
    BLOCK_R = 4
    while BLOCK_R < 64 and G * triton.cdiv(R, BLOCK_R) < 256:
        BLOCK_R *= 2
    cap = triton.next_power_of_2(64 // max(rows, 1))
    if cap > 64 // max(rows, 1):
        cap //= 2
    BLOCK_R = max(4, min(BLOCK_R, cap))
    _wo_a_absorb_kernel[(G * triton.cdiv(R, BLOCK_R),)](
        o,
        w,
        out,
        rows,
        o.stride(0),
        o.stride(1),
        w.stride(0),
        w.stride(1),
        out.stride(0),
        out.stride(1),
        R=R,
        D=D,
        BLOCK_R=BLOCK_R,
        BLOCK_D=BLOCK_D,
        ROWS=rows,
        num_warps=4 if BLOCK_R >= 32 else 8,
        num_stages=4,
    )


def wo_a_absorb(o: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """o [T, G, D] fp16, w [G, R, D] fp16 -> [T, G, R] fp16.

    Same result as torch.einsum("tgd,grd->tgr", o, w), accumulated in fp32. More than
    four tokens are processed in chunks of four, since the register accumulators are
    what cap BLOCK_R at small T.
    """
    T = o.shape[0]
    R = w.shape[1]
    # The kernels index the contraction dim as d0 + offs_d with no stride, so a
    # non-dense last dim would be read in the wrong order and nothing would raise.
    if o.stride(2) != 1:
        o = o.contiguous()
    if w.stride(2) != 1:
        w = w.contiguous()
    out = torch.empty((T, w.shape[0], R), dtype=o.dtype, device=o.device)
    for t0 in range(0, T, 4):
        _wo_a_absorb_chunk(o[t0:], w, out[t0:], t0, min(4, T - t0))
    return out
