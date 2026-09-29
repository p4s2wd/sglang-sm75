"""Triton FP8 MQA logits for the DSA indexer on GPUs without FP8 tensor cores.

Replaces ``deep_gemm.fp8_mqa_logits`` on sub-90 (SM75/SM80). SM75 Triton has
no e4m3 cast, so q/k are dequantized to fp16 with torch elementwise ops first
(cheap: q is [Q, H, 128], k is [K, 128]); the kernel then runs the fp16 tensor
cores (Turing mma.sync) with the DeepGEMM semantics:

    logits[q, j] = sum_h weights[q, h] * relu(dot(q[q, h], k[j])) * k_scale[j]
                   for ks[q] <= j < ke[q], else 0.

SMEM budget: SM75 has 64 KB. The real indexer shape is H=64 heads x D=128, so
loading all heads of a q-row block at once (BLOCK_Q*H*D*2 = 128 KB) does not
fit; the head dimension is tiled (BLOCK_H) and accumulated in registers.

  q:     [Q, H, D] fp16
  k:     [K, D]    fp16
  k_scale: [K]     fp32
  weights: [Q, H]  fp32
  ks, ke:  [Q]     int32
  out:     [Q, max_seqlen_k] fp32
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _fp16_mqa_logits_kernel(
    q_ptr,  # [Q, H, D] fp16
    k_ptr,  # [K, D] fp16
    ks_ptr,  # [K] fp32 (k scales)
    w_ptr,  # [Q, H] fp32
    ks_idx_ptr,  # [Q] int32
    ke_idx_ptr,  # [Q] int32
    out_ptr,  # [Q, N] fp32
    Q,
    N,
    H: tl.constexpr,
    D: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_q = pid * BLOCK_Q + tl.arange(0, BLOCK_Q)
    q_mask = offs_q < Q

    ks = tl.load(ks_idx_ptr + offs_q, mask=q_mask, other=0)
    ke = tl.load(ke_idx_ptr + offs_q, mask=q_mask, other=0)
    k_max = tl.max(ke, axis=0)

    offs_k = tl.arange(0, BLOCK_K)
    offs_bh = tl.arange(0, BLOCK_H)
    offs_d = tl.arange(0, D)
    num_h_tiles = H // BLOCK_H

    for k0 in range(0, k_max, BLOCK_K):
        g_k = k0 + offs_k
        k_mask = g_k < k_max
        k_v = tl.load(
            k_ptr + g_k[:, None] * D + offs_d[None, :],
            mask=k_mask[:, None],
            other=0.0,
        )  # [BLOCK_K, D]

        acc = tl.zeros((BLOCK_Q, BLOCK_K), tl.float32)
        for ht in range(0, num_h_tiles):
            offs_h = ht * BLOCK_H + offs_bh
            # q tile: [BLOCK_Q, BLOCK_H, D] -> [BLOCK_Q*BLOCK_H, D]
            q_off = (
                offs_q[:, None, None] * (H * D)
                + offs_h[None, :, None] * D
                + offs_d[None, None, :]
            )
            q_v = tl.load(q_ptr + q_off, mask=q_mask[:, None, None], other=0.0)
            q_v = tl.reshape(q_v, (BLOCK_Q * BLOCK_H, D))

            s = tl.dot(q_v, tl.trans(k_v), out_dtype=tl.float32)  # [BQ*BH, BK]
            s = tl.maximum(s, 0.0)  # relu
            s = tl.reshape(s, (BLOCK_Q, BLOCK_H, BLOCK_K))
            w = tl.load(
                w_ptr + offs_q[:, None] * H + offs_h[None, :],
                mask=q_mask[:, None],
                other=0.0,
            )  # [BLOCK_Q, BLOCK_H]
            acc += tl.sum(s * w[:, :, None], axis=1)  # [BLOCK_Q, BLOCK_K]

        k_scale = tl.load(ks_ptr + g_k, mask=k_mask, other=0.0)
        acc = acc * k_scale[None, :]

        valid = (g_k[None, :] >= ks[:, None]) & (g_k[None, :] < ke[:, None])
        acc = tl.where(valid, acc, 0.0)

        tl.store(
            out_ptr + offs_q[:, None] * N + g_k[None, :],
            acc,
            mask=q_mask[:, None] & k_mask[None, :],
        )


def fp16_mqa_logits_triton(
    q_fp8: torch.Tensor,  # [Q, H, D] float8_e4m3fn
    k_fp8: torch.Tensor,  # [K, D] float8_e4m3fn
    k_scale: torch.Tensor,  # [K] float32
    weights: torch.Tensor,  # [Q, H] float32
    ks: torch.Tensor,  # [Q] int32
    ke: torch.Tensor,  # [Q] int32
    out: torch.Tensor,  # [Q, N] float32
    block_q: int = 8,
    block_k: int = 64,
    block_h: int = 8,
) -> torch.Tensor:
    Q, H, D = q_fp8.shape
    q16 = q_fp8.to(torch.float16)
    k16 = k_fp8.to(torch.float16)
    grid = (triton.cdiv(Q, block_q),)
    _fp16_mqa_logits_kernel[grid](
        q16,
        k16,
        k_scale,
        weights,
        ks,
        ke,
        out,
        Q,
        out.shape[1],
        H=H,
        D=D,
        BLOCK_Q=block_q,
        BLOCK_K=block_k,
        BLOCK_H=block_h,
        num_warps=4,
        num_stages=1,
    )
    return out
