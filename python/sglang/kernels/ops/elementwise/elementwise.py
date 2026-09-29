import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

from sglang.kernels.jit.utils import is_arch_support_pdl
from sglang.srt.utils import is_hip

_is_hip = is_hip()


rmsnorm_autotune = triton.autotune(
    configs=[
        triton.Config(kwargs={"BLOCK_SIZE": 1024}, num_warps=4, num_stages=1),
        triton.Config(kwargs={"BLOCK_SIZE": 1024}, num_warps=8, num_stages=1),
        triton.Config(kwargs={"BLOCK_SIZE": 1024}, num_warps=16, num_stages=1),
        triton.Config(kwargs={"BLOCK_SIZE": 1024}, num_warps=4),
        triton.Config(kwargs={"BLOCK_SIZE": 1024}, num_warps=8),
        triton.Config(kwargs={"BLOCK_SIZE": 1024}, num_warps=16),
        triton.Config(kwargs={"BLOCK_SIZE": 1024}, num_warps=4, num_stages=4),
        triton.Config(kwargs={"BLOCK_SIZE": 1024}, num_warps=8, num_stages=4),
        triton.Config(kwargs={"BLOCK_SIZE": 1024}, num_warps=16, num_stages=4),
        triton.Config(kwargs={"BLOCK_SIZE": 1024}, num_warps=8, num_stages=8),
        triton.Config(kwargs={"BLOCK_SIZE": 1024}, num_warps=16, num_stages=8),
        triton.Config(kwargs={"BLOCK_SIZE": 2048}, num_warps=8),
        triton.Config(kwargs={"BLOCK_SIZE": 2048}, num_warps=16),
        triton.Config(kwargs={"BLOCK_SIZE": 2048}, num_warps=8, num_stages=4),
        triton.Config(kwargs={"BLOCK_SIZE": 2048}, num_warps=16, num_stages=4),
        triton.Config(kwargs={"BLOCK_SIZE": 4096}, num_warps=8),
        triton.Config(kwargs={"BLOCK_SIZE": 4096}, num_warps=16),
        triton.Config(kwargs={"BLOCK_SIZE": 8192}, num_warps=8),
        triton.Config(kwargs={"BLOCK_SIZE": 8192}, num_warps=16),
        triton.Config(kwargs={"BLOCK_SIZE": 8192}, num_warps=32),
        triton.Config(kwargs={"BLOCK_SIZE": 8192}, num_warps=8, num_stages=1),
        triton.Config(kwargs={"BLOCK_SIZE": 8192}, num_warps=16, num_stages=1),
        triton.Config(kwargs={"BLOCK_SIZE": 8192}, num_warps=32, num_stages=1),
        triton.Config(kwargs={"BLOCK_SIZE": 8192}, num_warps=8, num_stages=4),
        triton.Config(kwargs={"BLOCK_SIZE": 8192}, num_warps=16, num_stages=4),
        triton.Config(kwargs={"BLOCK_SIZE": 8192}, num_warps=32, num_stages=4),
        triton.Config(kwargs={"BLOCK_SIZE": 16384}, num_warps=8),
        triton.Config(kwargs={"BLOCK_SIZE": 16384}, num_warps=16),
        triton.Config(kwargs={"BLOCK_SIZE": 16384}, num_warps=32),
        triton.Config(kwargs={"BLOCK_SIZE": 16384}, num_warps=8, num_stages=1),
        triton.Config(kwargs={"BLOCK_SIZE": 16384}, num_warps=16, num_stages=1),
        triton.Config(kwargs={"BLOCK_SIZE": 16384}, num_warps=32, num_stages=1),
        triton.Config(kwargs={"BLOCK_SIZE": 16384}, num_warps=8, num_stages=4),
        triton.Config(kwargs={"BLOCK_SIZE": 16384}, num_warps=16, num_stages=4),
        triton.Config(kwargs={"BLOCK_SIZE": 16384}, num_warps=32, num_stages=4),
    ],
    key=["hidden_dim"],
)


@triton.jit
def fused_dual_residual_rmsnorm_kernel(
    output_ptr,
    mid_ptr,
    activ_ptr,
    residual_ptr,
    weight1_ptr,
    weight2_ptr,
    eps: tl.constexpr,
    hidden_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    input_start = pid * hidden_dim

    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < hidden_dim

    a_ = tl.load(activ_ptr + input_start + offsets, mask=mask, other=0.0)
    a = a_.to(tl.float32)
    rms = tl.sqrt(tl.sum(a * a, axis=0) / hidden_dim + eps)

    r = tl.load(residual_ptr + input_start + offsets, mask=mask, other=0.0)
    w1_ = tl.load(weight1_ptr + offsets, mask=mask, other=0.0)
    w1 = w1_.to(tl.float32)

    a2r = r + (a / rms * w1).to(r.dtype)
    tl.store(
        mid_ptr + input_start + offsets,
        a2r,
        mask=mask,
    )

    a2r = a2r.to(tl.float32)
    rms2 = tl.sqrt(tl.sum(a2r * a2r, axis=0) / hidden_dim + eps)

    w2_ = tl.load(weight2_ptr + offsets, mask=mask, other=0.0)
    w2 = w2_.to(tl.float32)

    tl.store(
        output_ptr + input_start + offsets,
        a2r / rms2 * w2,  # implicitly casts to output dtype here
        mask=mask,
    )


fused_dual_residual_rmsnorm_kernel_autotune = rmsnorm_autotune(
    fused_dual_residual_rmsnorm_kernel
)


def fused_dual_residual_rmsnorm(x, residual, weight1, weight2, eps, autotune=False):
    assert len(x.shape) == 2
    assert x.shape == residual.shape and x.dtype == residual.dtype, (
        f"{x.shape=} {residual.shape=} {x.dtype=} {residual.dtype=}"
    )
    output, mid = torch.empty_like(x), torch.empty_like(x)
    bs, hidden_dim = x.shape
    if autotune:
        fused_dual_residual_rmsnorm_kernel_autotune[(bs,)](
            output, mid, x, residual, weight1, weight2, eps=eps, hidden_dim=hidden_dim
        )
    else:
        max_warps = 16 if _is_hip else 32
        config = {
            "BLOCK_SIZE": triton.next_power_of_2(hidden_dim),
            "num_warps": max(
                min(triton.next_power_of_2(triton.cdiv(hidden_dim, 256)), max_warps), 4
            ),
        }

        fused_dual_residual_rmsnorm_kernel[(bs,)](
            output,
            mid,
            x,
            residual,
            weight1,
            weight2,
            eps=eps,
            hidden_dim=hidden_dim,
            **config,
        )

    return output, mid


@triton.jit
def fused_rmsnorm_kernel(
    output_ptr,
    activ_ptr,
    weight_ptr,
    eps: tl.constexpr,
    hidden_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0).to(tl.int64)
    input_start = pid * hidden_dim

    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < hidden_dim

    a_ = tl.load(activ_ptr + input_start + offsets, mask=mask, other=0.0)
    a = a_.to(tl.float32)
    rms = tl.sqrt(tl.sum(a * a, axis=0) / hidden_dim + eps)

    w1_ = tl.load(weight_ptr + offsets, mask=mask, other=0.0)
    w1 = w1_.to(tl.float32)

    a_rms = a / rms * w1

    tl.store(
        output_ptr + input_start + offsets,
        a_rms,  # implicitly casts to output dtype here
        mask=mask,
    )


def fused_rmsnorm(x, weight, eps, autotune=False, inplace=False):
    assert len(x.shape) == 2
    if inplace:
        output = x
    else:
        output = torch.empty_like(x)
    bs, hidden_dim = x.shape
    max_warps = 16 if _is_hip else 32
    config = {
        "BLOCK_SIZE": triton.next_power_of_2(hidden_dim),
        "num_warps": max(
            min(triton.next_power_of_2(triton.cdiv(hidden_dim, 256)), max_warps), 4
        ),
    }

    fused_rmsnorm_kernel[(bs,)](
        output, x, weight, eps=eps, hidden_dim=hidden_dim, **config
    )
    return output


# gelu on first half of vector
@triton.jit
def gelu_and_mul_kernel(
    out_hidden_states_ptr,  # (bs, hidden_dim)
    out_scales_ptr,  # (bs,)
    hidden_states_ptr,  # (bs, hidden_dim * 2)
    quant_max: tl.constexpr,
    static_scale: tl.constexpr,
    hidden_dim: tl.constexpr,  # the output hidden_dim
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    input_start = pid * hidden_dim * 2
    output_start = pid * hidden_dim

    input1_offs = tl.arange(0, BLOCK_SIZE)
    mask = tl.arange(0, BLOCK_SIZE) < hidden_dim  # shared for input1, input3, output
    input3_offs = hidden_dim + tl.arange(0, BLOCK_SIZE)
    output_offs = tl.arange(0, BLOCK_SIZE)

    x1 = tl.load(
        hidden_states_ptr + input_start + input1_offs, mask=mask, other=0.0
    ).to(tl.float32)
    x3 = tl.load(
        hidden_states_ptr + input_start + input3_offs, mask=mask, other=0.0
    ).to(tl.float32)

    # gelu
    # cast down before mul to better match training?
    gelu_x1 = 0.5 * (1.0 + tl.erf(x1 * 0.7071067811865475)) * x1
    out = x3 * gelu_x1.to(hidden_states_ptr.dtype.element_ty)

    if quant_max is not None:
        raise NotImplementedError()

    tl.store(out_hidden_states_ptr + output_start + output_offs, out, mask=mask)


def gelu_and_mul_triton(
    hidden_states,
    scales=None,
    quantize=None,  # dtype to quantize to
    out=None,
):
    bs, in_hidden_dim = hidden_states.shape
    hidden_dim = in_hidden_dim // 2

    if out is None:
        out_hidden_states = torch.empty(
            (bs, hidden_dim),
            dtype=quantize or hidden_states.dtype,
            device=hidden_states.device,
        )
    else:
        assert out.shape == (bs, hidden_dim)
        assert out.dtype == (quantize or hidden_states.dtype)
        out_hidden_states = out
    out_scales = None
    static_scale = False
    if quantize is not None:
        if scales is None:
            out_scales = torch.empty(
                (bs,), dtype=torch.float32, device=hidden_states.device
            )
        else:
            out_scales = scales
            static_scale = True

    max_warps = 16 if _is_hip else 32
    config = {
        # 8 ele per thread (not tuned)
        "num_warps": max(
            min(triton.next_power_of_2(triton.cdiv(hidden_dim, 8 * 32)), max_warps), 4
        ),
    }

    gelu_and_mul_kernel[(bs,)](
        out_hidden_states,
        out_scales,
        hidden_states,
        quant_max=torch.finfo(quantize).max if quantize is not None else None,
        static_scale=static_scale,
        hidden_dim=hidden_dim,
        BLOCK_SIZE=triton.next_power_of_2(hidden_dim),
        **config,
    )

    if quantize is not None:
        return out_hidden_states, out_scales
    else:
        return out_hidden_states, None


# silu on first half of vector
@triton.jit
def silu_and_mul_kernel(
    out_hidden_states_ptr,  # (bs, hidden_dim)
    out_scales_ptr,  # (bs,)
    hidden_states_ptr,  # (bs, hidden_dim * 2)
    quant_max: tl.constexpr,
    static_scale: tl.constexpr,
    hidden_dim: tl.constexpr,  # the output hidden_dim
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    input_start = pid * hidden_dim * 2
    output_start = pid * hidden_dim

    input1_offs = tl.arange(0, BLOCK_SIZE)
    mask = tl.arange(0, BLOCK_SIZE) < hidden_dim  # shared for input1, input3, output
    input3_offs = hidden_dim + tl.arange(0, BLOCK_SIZE)
    output_offs = tl.arange(0, BLOCK_SIZE)

    x1 = tl.load(
        hidden_states_ptr + input_start + input1_offs, mask=mask, other=0.0
    ).to(tl.float32)
    x3 = tl.load(
        hidden_states_ptr + input_start + input3_offs, mask=mask, other=0.0
    ).to(tl.float32)

    # silu
    # cast down before mul to better match training?
    silu_x1 = x1 * tl.sigmoid(x1)
    out = x3 * silu_x1.to(hidden_states_ptr.dtype.element_ty)

    if quant_max is not None:
        raise NotImplementedError()

    tl.store(out_hidden_states_ptr + output_start + output_offs, out, mask=mask)


def silu_and_mul_triton(
    hidden_states,
    scales=None,
    quantize=None,  # dtype to quantize to
    out=None,
):
    bs, in_hidden_dim = hidden_states.shape
    hidden_dim = in_hidden_dim // 2

    if out is None:
        out_hidden_states = torch.empty(
            (bs, hidden_dim),
            dtype=quantize or hidden_states.dtype,
            device=hidden_states.device,
        )
    else:
        assert out.shape == (bs, hidden_dim)
        assert out.dtype == (quantize or hidden_states.dtype)
        out_hidden_states = out
    out_scales = None
    static_scale = False
    if quantize is not None:
        if scales is None:
            out_scales = torch.empty(
                (bs,), dtype=torch.float32, device=hidden_states.device
            )
        else:
            out_scales = scales
            static_scale = True

    max_warps = 16 if _is_hip else 32
    config = {
        # 8 ele per thread (not tuned)
        "num_warps": max(
            min(triton.next_power_of_2(triton.cdiv(hidden_dim, 8 * 32)), max_warps), 4
        ),
    }

    silu_and_mul_kernel[(bs,)](
        out_hidden_states,
        out_scales,
        hidden_states,
        quant_max=torch.finfo(quantize).max if quantize is not None else None,
        static_scale=static_scale,
        hidden_dim=hidden_dim,
        BLOCK_SIZE=triton.next_power_of_2(hidden_dim),
        **config,
    )

    if quantize is not None:
        return out_hidden_states, out_scales
    else:
        return out_hidden_states, None


@triton.jit
def _fused_sigmoid_mul_kernel(
    output_ptr,
    attn_output_ptr,
    gate_ptr,
    gate_stride_row,
    gate_stride_head,
    hidden_dim: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """Fuse sigmoid(gate) * attn_output into a single kernel."""
    pid_row = tl.program_id(0).to(tl.int64)
    pid_block = tl.program_id(1)

    offsets = pid_block * BLOCK_H + tl.arange(0, BLOCK_H)
    mask = offsets < hidden_dim
    head = offsets // HEAD_DIM
    d = offsets - head * HEAD_DIM

    attn_off = pid_row * hidden_dim + offsets
    attn = tl.load(attn_output_ptr + attn_off, mask=mask, other=0.0).to(tl.float32)

    gate_off = pid_row * gate_stride_row + head * gate_stride_head + d
    g = tl.load(gate_ptr + gate_off, mask=mask, other=0.0).to(tl.float32)

    result = attn * tl.sigmoid(g)
    tl.store(output_ptr + attn_off, result, mask=mask)


def fused_sigmoid_mul(
    attn_output: torch.Tensor,
    gate: torch.Tensor,
    inplace: bool = False,
) -> torch.Tensor:
    """
    Fused sigmoid-mul for attention output gating.

    Equivalent to: attn_output * sigmoid(gate)

    The production Qwen3.5 path passes a 3D strided gate. A single hidden-block
    Triton kernel handles both that path and flat contiguous inputs.

    When inplace=True, writes result back to attn_output and returns it.

    Supports strided gate: if gate is 3D (num_tokens, num_heads, head_dim)
    and attn_output is 2D (num_tokens, hidden_dim), the kernel reads gate
    via explicit strides without requiring a contiguous copy.
    """
    if gate.ndim == 3 and attn_output.ndim == 2:
        # Strided gate path: gate is 3D (num_tokens, num_heads, head_dim)
        num_tokens, num_heads, head_dim = gate.shape
        hidden_dim = num_heads * head_dim
        assert attn_output.shape == (num_tokens, hidden_dim)
        gate_stride_row = gate.stride(0)
        gate_stride_head = gate.stride(1)
    else:
        # Flat path: both tensors have the same shape
        assert attn_output.shape == gate.shape, (
            "attn_output and gate must have the same shape"
        )
        hidden_dim = attn_output.shape[-1]
        num_tokens = attn_output.numel() // hidden_dim
        head_dim = hidden_dim
        gate_stride_row = hidden_dim
        gate_stride_head = hidden_dim

    out = attn_output if inplace else torch.empty_like(attn_output)
    block_h = 1024 if num_tokens < 1024 else 2048
    grid = (num_tokens, triton.cdiv(hidden_dim, block_h))
    _fused_sigmoid_mul_kernel[grid](
        out,
        attn_output,
        gate,
        gate_stride_row,
        gate_stride_head,
        hidden_dim,
        HEAD_DIM=head_dim,
        BLOCK_H=block_h,
        num_warps=4,
    )
    return out


@triton.jit
def _fused_gate_sigmoid_mul_add_kernel(
    hidden_states_ptr,  # [num_tokens, hidden_dim]
    gate_weight_ptr,  # [hidden_dim]
    shared_output_ptr,  # [num_tokens, hidden_dim]
    output_ptr,  # [num_tokens, hidden_dim], optionally also the addend
    hidden_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    DO_ADD: tl.constexpr = True,
    USE_PDL: tl.constexpr = False,
):
    pid = tl.program_id(axis=0).to(tl.int64)
    row_offset = pid * hidden_dim

    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < hidden_dim

    w = tl.load(gate_weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    if USE_PDL:
        tl.extra.cuda.gdc_wait()

    h = tl.load(hidden_states_ptr + row_offset + offsets, mask=mask, other=0.0).to(
        tl.float32
    )
    s = tl.load(shared_output_ptr + row_offset + offsets, mask=mask, other=0.0).to(
        tl.float32
    )
    if DO_ADD:
        f = tl.load(output_ptr + row_offset + offsets, mask=mask, other=0.0).to(
            tl.float32
        )

    if USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()

    gate_val = tl.sigmoid(tl.sum(h * w, axis=0))
    result = gate_val * s
    if DO_ADD:
        result += f

    tl.store(output_ptr + row_offset + offsets, result, mask=mask)


def _launch_fused_gate_sigmoid_mul(
    hidden_states: torch.Tensor,
    gate_weight: torch.Tensor,
    shared_output: torch.Tensor,
    output: torch.Tensor,
    *,
    do_add: bool,
) -> None:
    assert hidden_states.is_contiguous(), "hidden_states must be contiguous"
    assert gate_weight.is_contiguous(), "gate_weight must be contiguous"
    assert shared_output.is_contiguous(), "shared_output must be contiguous"
    assert output.is_contiguous(), "output must be contiguous"

    num_tokens, hidden_dim = hidden_states.shape
    assert gate_weight.shape == (hidden_dim,)
    assert shared_output.shape == (num_tokens, hidden_dim)
    assert output.shape == (num_tokens, hidden_dim)

    max_warps = 16 if _is_hip else 32
    config = {
        "BLOCK_SIZE": triton.next_power_of_2(hidden_dim),
        "num_warps": max(
            min(triton.next_power_of_2(triton.cdiv(hidden_dim, 256)), max_warps), 4
        ),
    }

    if num_tokens >= 1024:
        config["num_warps"] = min(config["num_warps"], 8)

    use_pdl = is_arch_support_pdl()
    pdl_kwargs = {"launch_pdl": True} if use_pdl else {}

    _fused_gate_sigmoid_mul_add_kernel[(num_tokens,)](
        hidden_states,
        gate_weight,
        shared_output,
        output,
        hidden_dim=hidden_dim,
        DO_ADD=do_add,
        USE_PDL=use_pdl,
        **config,
        **pdl_kwargs,
    )


def fused_gate_sigmoid_mul(
    hidden_states: torch.Tensor,
    gate_weight: torch.Tensor,
    shared_output: torch.Tensor,
) -> torch.Tensor:
    """Materialize the gated shared-expert contribution without an add/copy."""
    output = torch.empty_like(shared_output)
    _launch_fused_gate_sigmoid_mul(
        hidden_states,
        gate_weight,
        shared_output,
        output,
        do_add=False,
    )
    return output


def fused_gate_sigmoid_mul_add(
    hidden_states: torch.Tensor,
    gate_weight: torch.Tensor,
    shared_output: torch.Tensor,
    final_hidden_states: torch.Tensor,
) -> None:
    """Add the gated shared-expert contribution to routed-expert output."""
    _launch_fused_gate_sigmoid_mul(
        hidden_states,
        gate_weight,
        shared_output,
        final_hidden_states,
        do_add=True,
    )


@triton.jit
def _moe_swiglu_clamp_kernel(
    src_ptr,  # [rows, 2*INTER] fp16, gate in [:INTER], up in [INTER:]
    dst_ptr,  # [rows, INTER] fp16
    src_stride,  # row stride of src, in elements
    dst_stride,  # row stride of dst, in elements
    INTER,
    LIMIT,  # <= 0 disables the clamp
    HAS_LIMIT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Fused gate/up split, SiLU, clamp and multiply for the MoE activation.

    Replaces seven torch kernels -- chunk (two strided views), silu on a fp32 cast,
    two clamps, a multiply and a cast back -- which cost 20 us of device time at the
    decode shape (96 rows x 2048) against a ~1 us bandwidth floor. Two of the seven run
    on the non-vectorized elementwise path (8.8 us for the pair) because chunk() hands
    out strided views, so torch cannot prove the inner dimension is contiguous.

    The arithmetic is deliberately the same order as the torch code it replaces:
    silu in fp32, clamp silu's result, clamp up in fp32, multiply in fp32, cast to fp16
    once at the end. Doing the multiply in fp16, or clamping before silu, would change
    the bits.
    """
    row = tl.program_id(0)
    offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < INTER
    src = src_ptr + row.to(tl.int64) * src_stride
    gate = tl.load(src + offs, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(src + INTER + offs, mask=mask, other=0.0).to(tl.float32)
    # libdevice.exp, not tl.exp. tl.exp lowers to ex2.approx and lands a few fp32 ulp
    # off torch's silu on 34% of the fp16 grid; __nv_expf is the accurate expf that
    # torch's own CUDA kernel calls, and with it the fp16 result differs on 1 value out
    # of all 44160 finite fp16 magnitudes below 100. The remaining difference is a
    # single rounding tie, not a systematic bias.
    act = gate * (1.0 / (1.0 + libdevice.exp(-gate)))
    if HAS_LIMIT:
        act = tl.minimum(act, LIMIT)
        up = tl.minimum(tl.maximum(up, -LIMIT), LIMIT)
    out = (act * up).to(dst_ptr.dtype.element_ty)
    tl.store(dst_ptr + row.to(tl.int64) * dst_stride + offs, out, mask=mask)


def moe_swiglu_clamp(src: torch.Tensor, limit=None, out: torch.Tensor = None):
    """out = (silu(gate).clamp(max=limit) * up.clamp(-limit, limit)).half().

    `src` is [rows, 2*INTER] with gate first, matching the w13 output where the two
    halves are concatenated along the last dim. `limit` None skips both clamps.
    """
    rows, inter2 = src.shape
    inter = inter2 // 2
    # The kernel indexes within a row as src + row*stride(0) + col, so it assumes the
    # inner dimension is dense. A stride(1) != 1 input is not something Triton would
    # catch: it reads the wrong elements and returns plausible-looking numbers, which
    # measured 98300 of 98304 wrong before this check. Copy instead of failing, since a
    # strided input is a legitimate thing for a caller to hand over.
    if src.stride(1) != 1:
        src = src.contiguous()
    if out is not None and out.stride(1) != 1:
        raise ValueError("moe_swiglu_clamp: out must be dense along its last dim")
    if out is None:
        out = torch.empty((rows, inter), dtype=src.dtype, device=src.device)
    BLOCK = min(triton.next_power_of_2(inter), 2048)
    _moe_swiglu_clamp_kernel[(rows, triton.cdiv(inter, BLOCK))](
        src,
        out,
        src.stride(0),
        out.stride(0),
        inter,
        0.0 if limit is None else float(limit),
        HAS_LIMIT=limit is not None,
        BLOCK=BLOCK,
        num_warps=8 if BLOCK >= 1024 else 4,
    )
    return out


@triton.jit
def _moe_combine_kernel(
    down_ptr,  # [num_slots, HIDDEN] fp16, one row per padded slot
    sorted_ptr,  # [num_slots] int32, the (token*topk + k) each slot belongs to
    weight_ptr,  # [num_tokens*TOPK] fp32 routing weights
    out_ptr,  # [num_tokens, HIDDEN] output dtype
    num_valid_ptr,  # [1] int32, live slot count, read on the device
    num_slots,  # host int: the padded range, which live slots are scattered over
    down_stride,
    out_stride,
    TOPK: tl.constexpr,
    SCALE,
    HAS_SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
    USE_SLOT_MAP: tl.constexpr,
    slot_map_ptr,  # [num_tokens, TOPK] int32, slot of each (token, k), or -1
):
    """Weighted sum over topk of the routed-expert outputs, in one kernel.

    The torch version materialises a [num_tokens*topk+1, HIDDEN] fp16 buffer, scatters
    into it with index_copy_, slices and views it as [tokens, topk, hidden], casts the
    whole thing to fp32, multiplies by the routing weights and reduces over topk. That
    is 13 kernels and 30.6 us of device time at the decode shape against a ~1 us floor.

    Accumulation is fp32 with one cast at the end, matching the torch order: fp32
    multiply, fp32 sum over topk, then cast to the output dtype.

    Two ways to find a token's rows, because the right one depends on the shape:

    USE_SLOT_MAP=True walks a precomputed [num_tokens, TOPK] table of slot ids, so
    the inner loop is exactly TOPK iterations. USE_SLOT_MAP=False keeps the original
    scan of every padded slot, testing the owner scalar. The scan is fine at decode
    (a handful of slots) but quadratic-ish at prefill: with 512 tokens x topk 6 the
    padding runs to ~3072 slots and there are 4096 programs, so it issues ~12.6M
    scalar owner loads to deliver 24576 row-block loads. The table turns that into
    6 iterations per program and was measured at prefill shape on a 2080 Ti, where
    this kernel was 2.7 ms/call -- 9.4% of PP3's device time.
    """
    tok = tl.program_id(0)
    offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    acc = tl.zeros([BLOCK], dtype=tl.float32)

    if USE_SLOT_MAP:
        for k in tl.static_range(TOPK):
            s = tl.load(slot_map_ptr + tok * TOPK + k)
            if s >= 0:
                w = tl.load(weight_ptr + tok * TOPK + k)
                v = tl.load(down_ptr + s.to(tl.int64) * down_stride + offs)
                acc += w * v.to(tl.float32)
    else:
        num_valid = tl.load(num_valid_ptr)
        lo = tok * TOPK
        hi = lo + TOPK
        for s in range(0, num_slots):
            owner = tl.load(sorted_ptr + s)
            # Invalid means the pad sentinel or a position at or past num_valid, where
            # sorted_ids is uninitialised and would otherwise index outside the output.
            if (owner >= lo) & (owner < hi) & (s < num_valid):
                w = tl.load(weight_ptr + owner)
                v = tl.load(down_ptr + s.to(tl.int64) * down_stride + offs)
                acc += w * v.to(tl.float32)
    if HAS_SCALE:
        acc = acc * SCALE
    tl.store(
        out_ptr + tok.to(tl.int64) * out_stride + offs, acc.to(out_ptr.dtype.element_ty)
    )


@triton.jit
def _moe_slot_map_scatter_kernel(
    sorted_ptr,  # [num_slots] int32 owner = t*topk + k, or the pad sentinel
    num_valid_ptr,  # [1] int32
    map_ptr,  # [num_tokens*TOPK] int32, -1 where the pair has no live slot
    num_slots,
    NUM_PAIRS,  # num_tokens * topk
    TOPK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """map[t*topk + k] = the slot whose owner is t*topk + k, or -1.

    Scatter, not scan. One program per (owner, slot-block): it reads BLOCK owners,
    and for each one stores its own position into map[owner]. Every slot is visited
    exactly once, so this is O(num_slots) total work instead of the
    O(num_slots * num_tokens * topk) a per-pair scan would do.

    The map is pre-filled with -1 by the caller, so a pair with no live slot (its
    experts were all filtered out) keeps -1 and the combine kernel skips it. Writing
    -1 from here too would race: a later slot for the same owner would not be able to
    tell it had already been written.

    Graph-safe: no nonzero(), no host sync; num_valid is read on the device.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < num_slots
    owner = tl.load(sorted_ptr + offs, mask=mask, other=-1)
    num_valid = tl.load(num_valid_ptr)
    # Drop the pad sentinel and anything at or past num_valid, where sorted_ids is
    # uninitialised and could index outside the map.
    ok = mask & (owner >= 0) & (owner < NUM_PAIRS) & (offs < num_valid)
    tl.store(map_ptr + owner, offs.to(tl.int32), mask=ok)


def moe_combine(
    down_slots: torch.Tensor,
    sorted_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    num_valid: torch.Tensor,
    num_tokens: int,
    topk: int,
    scale=None,
    out: torch.Tensor = None,
    use_slot_map: bool = None,
):
    """out[t] = sum over the topk slots owned by t of weight * down_slots[slot].

    `sorted_ids` gives each padded slot its owner as t*topk + k; slots at or past
    `num_valid` and slots holding the pad sentinel contribute nothing. The sentinel is
    num_tokens*topk, which is above every valid owner, so the owner range test rejects it
    without a separate comparison. `topk_weights` is indexed flat as t*topk + k.
    """
    num_slots, hidden = down_slots.shape
    if down_slots.stride(1) != 1:
        down_slots = down_slots.contiguous()
    # Indexed flat as t*topk + k, so a non-contiguous weight tensor would be read in
    # the wrong order. Cheap to check and it fails loudly rather than silently.
    if topk_weights.stride(0) != topk:
        topk_weights = topk_weights.contiguous()
    topk_weights = topk_weights.reshape(-1)
    if out is None:
        out = torch.empty(
            (num_tokens, hidden), dtype=down_slots.dtype, device=down_slots.device
        )
    # 512 measured best or within 2% of best at one, two and eight tokens. Smaller
    # blocks add programs but each still scans every padded slot, so the win is small;
    # larger ones leave the SMs idle at the decode batch sizes this path serves.
    BLOCK = min(triton.next_power_of_2(hidden), 512)
    if hidden % BLOCK != 0:
        # The kernel issues one masked-free load per block, so a remainder would be
        # dropped rather than masked. Fall back to a block that divides hidden.
        BLOCK = next(
            (b for b in (256, 128, 64, 32, 16, 8, 4, 2, 1) if hidden % b == 0), 1
        )
    # The slot table costs a fixed ~0.13 ms (a fill_ plus a scatter over num_slots)
    # and turns the inner loop from num_slots to topk. It only pays once the scan is
    # long enough to dwarf that: measured on a 2080 Ti at H=2048, topk=6,
    #   B=128 slots=1024 -> scan 0.106 ms, table 0.222 ms  (0.48x, keep the scan)
    # B=256 slots=1792 -> scan 0.251 ms, table 0.221 ms  (1.13x, break even)
    #   B=512 slots=3072 -> scan 0.660 ms, table 0.232 ms  (2.85x)
    #   B=512 slots=4096 -> scan 0.853 ms, table 0.222 ms  (3.84x)
    # The crossover sits near B=256, i.e. where num_slots >= ~28x topk. The
    # num_tokens floor keeps small batches out even when the padding ratio is high:
    # at B=8 a padded num_slots can clear 28x topk while the whole combine is only
    # tens of microseconds, so the table's fixed cost would dominate. Decode
    # (B=1, num_slots ~ 8x topk) stays on the scan, which is what it wants.
    if use_slot_map is None:
        use_slot_map = num_tokens >= 256 and num_slots >= 28 * topk
    slot_map = None
    if use_slot_map:
        npairs = num_tokens * topk
        # Filled with -1 first so pairs with no live slot stay -1; the scatter then
        # writes the real slot ids. fill_ is a device op, so this stays graph-safe.
        slot_map = torch.full(
            (npairs,), -1, dtype=torch.int32, device=down_slots.device
        )
        SBLOCK = 256
        _moe_slot_map_scatter_kernel[(triton.cdiv(num_slots, SBLOCK),)](
            sorted_ids,
            num_valid,
            slot_map,
            num_slots,
            npairs,
            TOPK=topk,
            BLOCK=SBLOCK,
            num_warps=4,
        )
    _moe_combine_kernel[(num_tokens, hidden // BLOCK)](
        down_slots,
        sorted_ids,
        topk_weights,
        out,
        num_valid,
        num_slots,
        down_slots.stride(0),
        out.stride(0),
        TOPK=topk,
        SCALE=1.0 if scale is None else float(scale),
        HAS_SCALE=scale is not None,
        BLOCK=BLOCK,
        USE_SLOT_MAP=use_slot_map,
        slot_map_ptr=slot_map if slot_map is not None else sorted_ids,
        num_warps=2 if BLOCK <= 512 else 4,
    )
    return out
