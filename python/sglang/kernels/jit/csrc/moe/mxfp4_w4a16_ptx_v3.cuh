// SM75 MXFP4 W4A16 grouped GEMM, v3: activation reuse + PRMT dequant.
//
// Why this file exists rather than a tweak to mxfp4_w4a16_ptx.cuh. The shipped
// kernel runs at 115-122 GB/s of weight traffic and the objective wants ~500. A
// read-only probe with the identical grid, identical weight addresses and no
// compute at all streams the same bytes at 521-575 GB/s, so the access pattern
// and the memory system are not the limit -- the kernel is compute-bound. Two
// probes that each remove one cost pin the gap down:
//
//   read-only, no compute                  521-575 GB/s   (the ceiling)
//   activation loads replaced by a const   169-174 GB/s   -> loads cost ~34%
//   nibble dequant replaced by a move      146-150 GB/s   -> dequant costs ~21%
//   shipped                                115-122 GB/s
//
// So the two fixes are, in order of size:
//
// 1. NT n tiles per warp. The m16n8k8 A fragment for a given k step is the same
//    for every n tile, but the shipped kernel re-loads it for each one: eight
//    4-byte loads per warp per k step against a single 128-byte weight load.
//    Those loads are also strided (the fragment scatters across 8 rows), so they
//    are expensive per byte. Owning NT consecutive n tiles in one warp keeps the
//    fragments in registers and reuses them, cutting activation loads per weight
//    byte by NT.
//
// 2. PRMT dequant. The shipped h2_from_byte builds each fp16 from the e2m1
//    nibble with a shift, a mask, an add, a conditional and two ors -- roughly
//    two dozen integer ops per weight byte. Every e2m1 magnitude is one of eight
//    fp16 values whose low byte is always 0x00, so the whole conversion is a
//    byte lookup: PRMT picks the fp16 high byte for both nibbles of a byte in a
//    single instruction, and the sign bits are one masked or.
//
// NT and USE_PRMT are template parameters so both can be attributed separately.
// With NT=1 and USE_PRMT=0 the arithmetic is the shipped kernel's, and results
// are bit-identical to it in every configuration: the mma sequence per 16x8 tile
// and its order are unchanged.

#include <cuda_fp16.h>
#include <cstdint>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>
#include <sgl_kernel/utils.cuh>

namespace sglang {
namespace {

__device__ __forceinline__ uint32_t h2_from_byte_v3(uint32_t byte) {
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

// e2m1 nibble -> packed half2, via a register-resident byte lookup.
//
// The eight e2m1 magnitudes 0, .5, 1, 1.5, 2, 3, 4, 6 are fp16 0x0000, 0x3800,
// 0x3C00, 0x3E00, 0x4000, 0x4200, 0x4400, 0x4600: the low byte is always zero,
// so a magnitude is fully described by its high byte. Those eight high bytes sit
// in two registers, and PRMT selects from the eight bytes of a register pair, so
// one PRMT places the high bytes for both nibbles of a packed byte directly into
// halves 0 and 1 of the result. The sign is then a single masked or: bit 3 of the
// low nibble belongs at bit 15 and bit 7 of the high nibble at bit 31.
//
// Verified against h2_from_byte_v3 for all 256 packed bytes. The only difference
// is nibble 8 (e2m1 negative zero): the arithmetic form discards its sign and
// yields +0.0, this one yields -0.0. They are equal in every fp16 operation the
// mma performs, so the products and sums match; a raw bit compare of an output
// tile whose terms are all zero would still see 0x8000 vs 0x0000, so the
// correctness check normalises zeros by adding 0.0 before comparing.
__device__ __forceinline__ uint32_t h2_from_byte_prmt(uint32_t byte) {
  const uint32_t r0 = 0x3E3C3800u;  // magnitude high bytes 0, .5, 1, 1.5
  const uint32_t r1 = 0x46444240u;  // magnitude high bytes 2, 3, 4, 6
  // Selector byte 0 supplies the low nibble's magnitude and selector byte 1 the
  // high nibble's (byte & 0x70 is already bits 4-6, so <<4 moves it to byte 1).
  // The looked-up bytes then need one byte of shift to become the two fp16 high
  // bytes, which is the <<8 below.
  //
  // The selector positions are not what the ISA text suggests. A device-side
  // probe (a=b=byte-index constants, one selector per lane) showed that on this
  // sm_75/nvcc combination selector byte p feeds destination byte 2*p, so the
  // literal reading -- indices in selector bytes 1 and 3 -- leaves both
  // magnitudes one byte low and is wrong on 255 of the 256 packed bytes. This
  // form was then checked against h2_from_byte_v3 for all 256 bytes.
  const uint32_t sel = (byte & 0x7u) | ((byte & 0x70u) << 4);
  uint32_t mag;
  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(mag) : "r"(r0), "r"(r1), "r"(sel));
  const uint32_t sgn = ((byte & 0x8u) << 12) | ((byte & 0x80u) << 24);
  return (mag << 8) | sgn;
}

__device__ __forceinline__ void mma_m16n8k8_v3(float c[4], const uint32_t a[2], const uint32_t b) {
  asm volatile(
      "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(b));
}

struct W4A16V3Param {
  const uint16_t* a;
  const uint32_t* w;
  const uint8_t* s;
  const int32_t* sorted;
  const int32_t* eids;
  uint16_t* out;
  const int32_t* num_valid;
  int64_t m_total;
  int64_t n;
  int64_t k_steps;
  int64_t n_tiles;
  int64_t stride_a;
  int64_t stride_out;
  // Row stride in float elements of the KS-way partial buffer, used only when KS>1.
  int64_t split_stride;
};

template <int NT, bool USE_PRMT, int KS>
__global__ __launch_bounds__(32) void w4a16_v3_kernel(const W4A16V3Param p) {
  const int m_blk = blockIdx.y;
  const int nt0 = blockIdx.x * NT;   // first of NT consecutive n tiles
  const int lane = threadIdx.x;
  const int gid = lane >> 2;
  const int c4 = lane & 3;

  // KS-way split of the k reduction across blockIdx.z. Each split owns a disjoint
  // slice of BOTH the weight and the activation, so no weight byte is read twice and
  // no activation fragment is re-read: the KS blocks covering one tile walk disjoint k
  // ranges and together load exactly what the single KS=1 block used to.
  //
  // Decode is occupancy-starved rather than bandwidth-bound. At the production shape
  // (topk=6 experts, TP2) the grid is [64, 6] = 384 blocks on 68 SMs = 5.6 warps/SM,
  // and a block-count sweep of this same kernel with identical bytes per block gives
  // 202 GB/s at 384 blocks, 262 at 512, 315 at 768, 337 at 1536, 366 at 3072. The
  // limit is bytes in flight, not DRAM, so the fix is more blocks.
  //
  // Raising NT was measured to fail (NT=1 also gives 1536 blocks but is slower,
  // because each block then re-loads the A fragments every k step; NT=8 spills).
  // KS adds blocks without touching either cost.
  const int ks_lo = (static_cast<int>(p.k_steps) * static_cast<int>(blockIdx.z)) / KS;
  const int ks_hi = (static_cast<int>(p.k_steps) * (static_cast<int>(blockIdx.z) + 1)) / KS;

  // Same guard as the shipped kernel: sorted/eids are capacity-sized torch.empty
  // buffers whose tail holds garbage that would index the weights out of bounds.
  if (static_cast<int64_t>(m_blk) * 16 >= static_cast<int64_t>(*p.num_valid)) return;

  const int expert = p.eids[m_blk];
  const int row0 = m_blk * 16;

  const int id_lo = p.sorted[row0 + gid];
  const int id_hi = p.sorted[row0 + gid + 8];
  const bool v_lo = static_cast<int64_t>(id_lo) < p.m_total;
  const bool v_hi = static_cast<int64_t>(id_hi) < p.m_total;
  const uint16_t* a_lo = p.a + (v_lo ? id_lo : 0) * p.stride_a;
  const uint16_t* a_hi = p.a + (v_hi ? id_hi : 0) * p.stride_a;

  // Weight and scale pointers for each owned tile. Tiles are NT apart in the
  // repacked [E][n_tile][k_step][lane] layout, so the stride between them is one
  // full k walk.
  // The grid rounds n_tiles up to a multiple of NT, so the last block can own
  // tiles that do not exist. Clamping the pointers keeps those loads in bounds;
  // their accumulators are computed and then discarded at the store.
  //
  // wp carries the +ks_lo start because it is a walking pointer (wp[t][0] then += 32
  // per iteration). a and sp are indexed by the absolute ks below, so they must NOT be
  // offset here or the split start would be counted twice.
  const uint32_t* wp[NT];
  const uint8_t* sp[NT];
#pragma unroll
  for (int t = 0; t < NT; ++t) {
    const int tile = min(nt0 + t, static_cast<int>(p.n_tiles) - 1);
    wp[t] = p.w + ((static_cast<int64_t>(expert) * p.n_tiles + tile) * p.k_steps + ks_lo) * 32 + lane;
    sp[t] = p.s + (static_cast<int64_t>(expert) * p.n + tile * 8 + gid) * p.k_steps;
  }

  float acc[NT][4];
#pragma unroll
  for (int t = 0; t < NT; ++t)
#pragma unroll
    for (int i = 0; i < 4; ++i) acc[t][i] = 0.f;

  uint32_t wv[NT];
#pragma unroll
  for (int t = 0; t < NT; ++t) wv[t] = wp[t][0];

  for (int ks = ks_lo; ks < ks_hi; ++ks) {
    const int kc = ks * 32 + 2 * c4;

    // Prefetch the next k step's weight words while this one is consumed.
    uint32_t wv_next[NT];
    if (ks + 1 < ks_hi) {
#pragma unroll
      for (int t = 0; t < NT; ++t) wv_next[t] = wp[t][32];
    } else {
#pragma unroll
      for (int t = 0; t < NT; ++t) wv_next[t] = 0u;
    }

    // The A fragments are shared by every owned tile: loaded once per k step and
    // fed to NT times as many mma as the shipped kernel would use per load.
    uint32_t afr[4][2];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int kj = kc + 8 * j;
      afr[j][0] = v_lo ? *reinterpret_cast<const uint32_t*>(a_lo + kj) : 0u;
      afr[j][1] = v_hi ? *reinterpret_cast<const uint32_t*>(a_hi + kj) : 0u;
    }

#pragma unroll
    for (int t = 0; t < NT; ++t) {
      const uint32_t s_bits =
          static_cast<uint32_t>(static_cast<int32_t>(sp[t][ks]) - 112) << 10;
      const uint32_t s2u = s_bits | (s_bits << 16);
      const __half2 s2 = *reinterpret_cast<const __half2*>(&s2u);
      float bacc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        uint32_t braw;
        if (USE_PRMT) braw = h2_from_byte_prmt((wv[t] >> (8 * j)) & 0xFFu);
        else braw = h2_from_byte_v3((wv[t] >> (8 * j)) & 0xFFu);
        __half2 bh = *reinterpret_cast<const __half2*>(&braw);
        bh = __hmul2(bh, s2);
        mma_m16n8k8_v3(bacc, afr[j], *reinterpret_cast<const uint32_t*>(&bh));
      }
#pragma unroll
      for (int i = 0; i < 4; ++i) acc[t][i] += bacc[i];
    }

#pragma unroll
    for (int t = 0; t < NT; ++t) {
      wv[t] = wv_next[t];
      wp[t] += 32;
    }
  }

  // KS=1 writes the final fp16 result. KS>1 writes an fp32 partial per split, which
  // the caller sums: accumulating each k slice in fp32 and rounding once after the sum
  // makes the numerics a pure change of summation order, exactly as in any k-split GEMM,
  // instead of rounding each of KS partials to fp16 before combining them.
  //
  // p.stride_out is in fp16 elements (the caller's out tensor) and p.split_stride is in
  // float elements (the partial buffer), so the two paths use different row strides and
  // must not share a row offset.
#pragma unroll
  for (int t = 0; t < NT; ++t) {
    if (nt0 + t >= p.n_tiles) break;
    const int n0 = (nt0 + t) * 8 + 2 * c4;
    if (KS == 1) {
      const int64_t orow_lo = static_cast<int64_t>(row0 + gid) * p.stride_out;
      const int64_t orow_hi = static_cast<int64_t>(row0 + gid + 8) * p.stride_out;
      const __half2 h_lo = __float22half2_rn(make_float2(acc[t][0], acc[t][1]));
      const __half2 h_hi = __float22half2_rn(make_float2(acc[t][2], acc[t][3]));
      *reinterpret_cast<__half2*>(p.out + orow_lo + n0) = h_lo;
      *reinterpret_cast<__half2*>(p.out + orow_hi + n0) = h_hi;
    } else {
      float* po = reinterpret_cast<float*>(p.out)
                + static_cast<int64_t>(blockIdx.z) * p.split_stride;
      const int64_t lo = static_cast<int64_t>(row0 + gid) * p.n + n0;
      const int64_t hi = static_cast<int64_t>(row0 + gid + 8) * p.n + n0;
      *reinterpret_cast<float2*>(po + lo) = make_float2(acc[t][0], acc[t][1]);
      *reinterpret_cast<float2*>(po + hi) = make_float2(acc[t][2], acc[t][3]);
    }
  }
}

// Combine the KS-way k-split partials, touching only the rows that are live.
//
// torch.sum over the partial buffer is not usable here: the buffer is indexed by slot
// and the slot buffers are the align buffer's CAPACITY (num_tokens*topk +
// (E+1)*(block_m-1), 3861 rows at decode) rather than the handful of rows that are
// live, because num_valid lives on the device to keep the whole path graph-capturable.
// Summing all 3861 rows moves KS*3861*2048*4 = 63 MB per split, which is as much
// traffic as the GEMM itself and erased the split's entire 1.20x gain.
//
// This kernel reads num_valid on the device, so the bound costs nothing to enforce and
// stays capturable: at decode it touches 96 rows instead of 3861, 40x less traffic.
__global__ void w4a16_ks_reduce_kernel(const float* __restrict__ part,
                                       uint16_t* __restrict__ out,
                                       const int32_t* __restrict__ num_valid,
                                       int64_t n, int64_t stride_out,
                                       int64_t split_stride, int ks) {
  const int64_t live = static_cast<int64_t>(*num_valid);
  const int64_t total = live * n;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < total; i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t row = i / n, col = i - row * n;
    const float* p = part + row * n + col;
    float acc = p[0];
#pragma unroll 1
    for (int k = 1; k < ks; ++k) acc += p[k * split_stride];
    out[row * stride_out + col] = __half_as_ushort(__float2half_rn(acc));
  }
}

}  // namespace

struct W4A16PtxV3Kernel {
  // cfg selects (NT, PRMT): 0 = (1,off) the shipped shape, 1 = (1,on),
  // 2 = (2,on), 3 = (4,on), 4 = (2,off), 5 = (4,off), 6 = (8,on).
  static void run(tvm::ffi::TensorView a, tvm::ffi::TensorView w, tvm::ffi::TensorView s,
                  tvm::ffi::TensorView sorted, tvm::ffi::TensorView eids,
                  tvm::ffi::TensorView out, tvm::ffi::TensorView num_valid,
                  double m_total_d, double cfg_d) {
    using namespace host;
    const auto device_ = a.device();
    const int cfg = static_cast<int>(cfg_d);

    // The dequant probe takes no GEMM parameters at all, so it must run before
    // the parameter block reads s.size(1) -- the caller passes a 1-D scratch.

    W4A16V3Param p;
    p.a = static_cast<const uint16_t*>(a.data_ptr());
    p.w = static_cast<const uint32_t*>(w.data_ptr());
    p.s = static_cast<const uint8_t*>(s.data_ptr());
    p.sorted = static_cast<const int32_t*>(sorted.data_ptr());
    p.eids = static_cast<const int32_t*>(eids.data_ptr());
    p.out = static_cast<uint16_t*>(out.data_ptr());
    p.num_valid = static_cast<const int32_t*>(num_valid.data_ptr());
    p.m_total = static_cast<int64_t>(m_total_d);
    p.n = s.size(1);
    p.k_steps = a.size(1) / 32;
    p.n_tiles = p.n / 8;
    p.stride_a = a.stride(0);
    p.stride_out = out.stride(0);
    p.split_stride = 0;  // set per-launch below when KS>1; never read when KS==1
    const int64_t num_m_blocks = eids.size(0);

// split_stride is only read when KS>1, and it is only *valid* then: the KS=1 output
// is the 2-D fp16 tensor, which has no dim 2 to ask for. Computing it inside the KS
// branch keeps the KS=1 path from indexing a dimension that does not exist.
#define GO(NT, PR, KS)                                                          \
  {                                                                            \
    if (KS > 1) p.split_stride = out.size(1) * out.size(2);                     \
    const int64_t gx = (p.n_tiles + NT - 1) / NT;                              \
    dim3 g(static_cast<uint32_t>(gx), static_cast<uint32_t>(num_m_blocks), KS);  \
    LaunchKernel(g, 32, device_)(w4a16_v3_kernel<NT, PR, KS>, p);               \
  }
    switch (cfg) {
      case 0: GO(1, false, 1); break;
      case 1: GO(1, true, 1); break;
      case 2: GO(2, true, 1); break;
      case 3: GO(4, true, 1); break;
      case 4: GO(2, false, 1); break;
      case 5: GO(4, false, 1); break;
      case 6: GO(8, true, 1); break;
      // k-split variants: same NT=4/PRMT shape as cfg 3, more blocks.
      case 7: GO(4, true, 2); break;
      case 8: GO(4, true, 4); break;
      case 9: GO(2, true, 4); break;
      case 10: GO(4, true, 8); break;
      case 11: GO(2, true, 8); break;
      case 12: GO(1, true, 8); break;
      case 13: GO(2, true, 2); break;

      default: GO(1, false, 1); break;
    }
#undef GO
  }
};

// Exported separately so the Python wrapper can combine k-split partials without a
// host-visible row count. See w4a16_ks_reduce_kernel for why torch.sum is unusable.
struct W4A16PtxV3Reduce {
  static void run(tvm::ffi::TensorView part, tvm::ffi::TensorView out,
                  tvm::ffi::TensorView num_valid, double ks_d) {
    using namespace host;
    const auto device_ = part.device();
    const int ks = static_cast<int>(ks_d);
    const int64_t n = out.size(1);
    const int64_t split_stride = part.size(1) * part.size(2);
    // The grid is sized to the capacity, not to num_valid, because the live count is
    // only known on the device; blocks past the end exit on the first loop test. One
    // block per 256 threads over the capacity would be 30k blocks at decode, so cap the
    // grid and let each block stride, which is the usual grid-stride loop.
    const int64_t threads = 256;
    const int64_t total = part.size(1) * n;
    const int64_t blocks = std::min<int64_t>((total + threads - 1) / threads, 2048);
    LaunchKernel(dim3(static_cast<uint32_t>(blocks)), static_cast<uint32_t>(threads),
                 device_)(w4a16_ks_reduce_kernel,
                          static_cast<const float*>(part.data_ptr()),
                          static_cast<uint16_t*>(out.data_ptr()),
                          static_cast<const int32_t*>(num_valid.data_ptr()),
                          n, out.stride(0), split_stride, ks);
  }
};

}  // namespace sglang
