// SM75 (Turing) MXFP4 W4A16 grouped GEMM via inline PTX mma.sync.m16n8k8.
//
// Why hand-written: Triton on sm_75 must round-trip register-computed
// operands through shared memory before ldmatrix/mma (its hard limit; the
// Triton kernel tops out at ~12 GB/s of weight traffic). Here the e2m1
// nibbles are dequantized in registers and fed to the tensor cores directly,
// with zero shared memory and zero __syncthreads.
//
// Turing uses m16n8k8 (k=8); m16n8k16 is sm_80+. Warp tile: BM=16, BN=8,
// BK=32 (four mma k-steps of 8, sharing one MXFP4 scale group).
//
// Repacked weight layout (produced by repack_mxfp4 in the test):
//   [E][n_tile][k_step][lane]  (uint32), lane = gid*4 + c4, gid=lane>>2, c4=lane&3
//   byte j (j=0..3) of the lane's u32 = packed byte W[e][nt*8+gid][ks*16 + j*4 + c4]
//   which is exactly the B-fragment byte for mma k-step j.
// So one 128-byte contiguous load per warp per 32-k-block covers the whole
// warp's B fragments for four mma steps.

#include <cuda_fp16.h>
#include <cstdint>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>
#include <sgl_kernel/utils.cuh>

namespace sglang {
namespace {

__device__ __forceinline__ uint32_t h2_from_byte(uint32_t byte) {
  auto deq = [](uint32_t n) -> uint32_t {
    const uint32_t sign = (n & 0x8u) << 12;
    const uint32_t e = (n >> 1) & 0x3u;
    const uint32_t m = n & 0x1u;
    const uint32_t exp_bits = (e + 14u) << 10;
    const uint32_t mant = (e != 0u) ? (m << 9) : 0u;
    const uint32_t bits = (e == 0u && m == 0u) ? 0u : (sign | exp_bits | mant);
    return bits;
  };
  return deq(byte & 0xFu) | (deq((byte >> 4) & 0xFu) << 16);
}

__device__ __forceinline__ void mma_m16n8k8(float c[4], const uint32_t a[2], const uint32_t b) {
  asm volatile(
      "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(b));
}

struct W4A16Param {
  const uint16_t* a;
  const uint32_t* w;
  // UE8M0 bytes, value = 2^(byte - 127). Taking the raw byte instead of a
  // pre-expanded fp32 scale keeps the repacked path memory-neutral: fp32 scales
  // would add E*N*K/32*4 = 1.6 GiB per card on top of the weights, which is what
  // made repacking look unaffordable on a 22 GB card. The direct kernel already
  // reads raw bytes the same way.
  const uint8_t* s;
  const int32_t* sorted;
  const int32_t* eids;
  // Device scalar: entries of sorted/eids that moe_align_block_size actually
  // wrote. The buffers are capacity-sized torch.empty allocations, so blocks
  // past this read a garbage expert id (an out-of-bounds weight address) and
  // write past `out`. The direct kernel has the same guard.
  const int32_t* num_valid;
  uint16_t* out;
  int64_t m_total;
  int64_t n;
  int64_t k_steps;
  int64_t n_tiles;
  int64_t stride_a;
  int64_t stride_out;
};

__global__ void w4a16_ptx_kernel(const W4A16Param p) {
  // n on blockIdx.x, m on blockIdx.y: see the note in W4A16PtxKernel::run.
  const int m_blk = blockIdx.y;
  const int nt = blockIdx.x;
  const int lane = threadIdx.x;
  const int gid = lane >> 2;
  const int c4 = lane & 3;

  // moe_align_block_size writes only the first *num_valid entries of the
  // capacity-sized sorted/eids buffers; everything past that is uninitialized
  // torch.empty memory. Launching those blocks reads a garbage expert id (an
  // out-of-bounds weight/scale address) and writes past `out`.
  if (static_cast<int64_t>(m_blk) * 16 >= static_cast<int64_t>(*p.num_valid)) return;

  const int expert = p.eids[m_blk];
  const int row0 = m_blk * 16;

  const int id_lo = p.sorted[row0 + gid];
  const int id_hi = p.sorted[row0 + gid + 8];
  const bool v_lo = static_cast<int64_t>(id_lo) < p.m_total;
  const bool v_hi = static_cast<int64_t>(id_hi) < p.m_total;
  const uint16_t* a_lo = p.a + (v_lo ? id_lo : 0) * p.stride_a;
  const uint16_t* a_hi = p.a + (v_hi ? id_hi : 0) * p.stride_a;

  const uint32_t* wp =
      p.w + ((static_cast<int64_t>(expert) * p.n_tiles + nt) * p.k_steps) * 32 + lane;
  const int n_col = nt * 8 + gid;
  const uint8_t* sp = p.s + (static_cast<int64_t>(expert) * p.n + n_col) * p.k_steps;

  float acc[4] = {0.f, 0.f, 0.f, 0.f};

  uint32_t wv = *wp;  // software pipeline: weight word for this iteration
  const uint32_t* wp_next = wp + 32;
#pragma unroll 4
  for (int ks = 0; ks < p.k_steps; ++ks) {
    const int kc = ks * 32 + 2 * c4;  // k offset of this lane's pair within the block
    const uint32_t wv_next = (ks + 1 < p.k_steps) ? *wp_next : 0u;
    wp_next += 32;
    // The scale belongs to the weight row (B's n = gid), so bake it into the
    // B fragments; the accumulator's columns are 2*c4, not gid.
    // UE8M0 byte b is the value 2^(b-127); in fp16 that is exactly the bit
    // pattern (b-112)<<10 (fp16 exponent bias is 15). Exact for b in [113,142];
    // this checkpoint's measured scale-byte range is [118, 126]. Same identity
    // the direct kernel uses.
    const uint32_t s_bits = static_cast<uint32_t>(
        static_cast<int32_t>(*sp) - 112) << 10;
    const uint32_t s2u = s_bits | (s_bits << 16);  // both halves = the scale
    const __half2 s2 = *reinterpret_cast<const __half2*>(&s2u);
    float block_acc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int kj = kc + 8 * j;
      uint32_t a[2];
      a[0] = v_lo ? *reinterpret_cast<const uint32_t*>(a_lo + kj) : 0u;
      a[1] = v_hi ? *reinterpret_cast<const uint32_t*>(a_hi + kj) : 0u;
      uint32_t braw = h2_from_byte((wv >> (8 * j)) & 0xFFu);
      __half2 bh = *reinterpret_cast<const __half2*>(&braw);
      bh = __hmul2(bh, s2);
      mma_m16n8k8(block_acc, a, *reinterpret_cast<const uint32_t*>(&bh));
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) acc[i] += block_acc[i];
    wv = wv_next;
    wp += 32;
    sp += 1;
  }

  const int n0 = nt * 8 + 2 * c4;
  const int64_t orow_lo = static_cast<int64_t>(row0 + gid) * p.stride_out;
  const int64_t orow_hi = static_cast<int64_t>(row0 + gid + 8) * p.stride_out;
  const __half2 h_lo = __float22half2_rn(make_float2(acc[0], acc[1]));
  const __half2 h_hi = __float22half2_rn(make_float2(acc[2], acc[3]));
  *reinterpret_cast<__half2*>(p.out + orow_lo + n0) = h_lo;
  *reinterpret_cast<__half2*>(p.out + orow_hi + n0) = h_hi;
}

}  // namespace

struct W4A16PtxKernel {
  static void run(tvm::ffi::TensorView a, tvm::ffi::TensorView w, tvm::ffi::TensorView s,
                  tvm::ffi::TensorView sorted, tvm::ffi::TensorView eids,
                  tvm::ffi::TensorView out, tvm::ffi::TensorView num_valid,
                  double m_total_d) {
    using namespace host;
    const auto device_ = a.device();

    const int64_t m_total = static_cast<int64_t>(m_total_d);
    const int64_t k = a.size(1);
    const int64_t n = s.size(1);
    const int64_t k_steps = k / 32;
    const int64_t n_tiles = n / 8;
    const int64_t num_m_blocks = eids.size(0);

    W4A16Param p;
    p.a = static_cast<const uint16_t*>(a.data_ptr());
    p.w = static_cast<const uint32_t*>(w.data_ptr());
    p.s = static_cast<const uint8_t*>(s.data_ptr());
    p.sorted = static_cast<const int32_t*>(sorted.data_ptr());
    p.eids = static_cast<const int32_t*>(eids.data_ptr());
    p.num_valid = static_cast<const int32_t*>(num_valid.data_ptr());
    p.out = static_cast<uint16_t*>(out.data_ptr());
    p.m_total = m_total;
    p.n = n;
    p.k_steps = k_steps;
    p.n_tiles = n_tiles;
    p.stride_a = a.stride(0);
    p.stride_out = out.stride(0);

    // blockIdx.x advances fastest, so it carries the n tile: all n tiles of one
    // m block then run back to back and share that block's 128 KB of activation
    // rows through L2. The old order (m on x) spread 256 m blocks' rows -- 32 MB
    // at the prefill shape -- across a 4 MB L2 and re-fetched them per n tile.
    dim3 grid(static_cast<uint32_t>(n_tiles), static_cast<uint32_t>(num_m_blocks), 1);
    LaunchKernel(grid, 32, device_)(w4a16_ptx_kernel, p);
  }
};

}  // namespace sglang
