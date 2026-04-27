#include <algorithm>
#include <cstdlib>

#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <torch/python.h>

#include "jit_kernels/impls/sm120_mqa_logits_fallback.hpp"
#include "utils/exception.hpp"

namespace deep_gemm {
namespace sm120_fallback {

__device__ __forceinline__ float fp8_e4m3fn_to_float(uint8_t raw) {
    const uint8_t mag = raw & 0x7fu;
    if (mag == 0)
        return 0.0f;
    const int fp8_exp = static_cast<int>((mag >> 3) & 0x0fu);
    const int mant = static_cast<int>(mag & 0x07u);
    const float value =
        fp8_exp == 0
            ? ldexpf(static_cast<float>(mant), -9)
            : ldexpf(1.0f + static_cast<float>(mant) * 0.125f, fp8_exp - 7);
    return (raw & 0x80u) ? -value : value;
}

__device__ __forceinline__ float fp4_e2m1_to_float(uint8_t code) {
    const uint8_t value_idx = code & 0x07;
    float value = 0.0f;
    switch (value_idx) {
        case 0: value = 0.0f; break;
        case 1: value = 0.5f; break;
        case 2: value = 1.0f; break;
        case 3: value = 1.5f; break;
        case 4: value = 2.0f; break;
        case 5: value = 3.0f; break;
        case 6: value = 4.0f; break;
        default: value = 6.0f; break;
    }
    return (code & 0x08) && value_idx != 0 ? -value : value;
}

template <typename out_t>
__device__ __forceinline__ void store_logit(out_t* out, int64_t offset,
                                            float value) {
    out[offset] = static_cast<out_t>(value);
}

template <>
__device__ __forceinline__ void store_logit<__nv_bfloat16>(
    __nv_bfloat16* out, int64_t offset, float value) {
    out[offset] = __float2bfloat16(value);
}

template <typename out_t>
__global__ void paged_fp8_mqa_logits_fast_kernel(
    const uint8_t* q, const uint8_t* kv, const float* kv_sf,
    const float* weights, const int32_t* context_lens,
    const int32_t* block_table, out_t* logits, int batch_size, int next_n,
    int num_heads, int head_dim, int block_kv, int kv_stride0,
    int kv_stride1, int kv_sf_stride0, int block_table_stride,
    int logits_stride, int max_context_len, bool is_context_lens_2d) {
    __shared__ float partials[256];
    __shared__ float head_values[64];

    const int n = blockIdx.x;
    const int row = blockIdx.y;
    const int b = row / next_n;
    const int t = row - b * next_n;
    if (b >= batch_size || n >= max_context_len)
        return;

    const int q_limit = is_context_lens_2d
                            ? context_lens[b * next_n + t]
                            : context_lens[b] - next_n + t;
    float result = -INFINITY;
    if (n <= q_limit) {
        const int block_offset = n / block_kv;
        const int token_offset = n - block_offset * block_kv;
        const int block_idx = block_table[b * block_table_stride + block_offset];
        const float k_scale = kv_sf[block_idx * kv_sf_stride0 + token_offset];
        const int group_threads = 256 / num_heads;
        const int h = threadIdx.x / group_threads;
        const int lane = threadIdx.x - h * group_threads;
        float local = 0.0f;
        if (h < num_heads) {
            const int64_t q_base =
                static_cast<int64_t>(row * num_heads + h) * head_dim;
            const int kv_base = block_idx * kv_stride0 + token_offset * kv_stride1;
            for (int d = lane; d < head_dim; d += group_threads) {
                local += fp8_e4m3fn_to_float(q[q_base + d]) *
                         (fp8_e4m3fn_to_float(kv[kv_base + d]) * k_scale);
            }
        }
        partials[threadIdx.x] = local;
        __syncthreads();

        if (h < num_heads && lane == 0) {
            float dot = 0.0f;
            for (int i = 0; i < group_threads; ++i)
                dot += partials[h * group_threads + i];
            head_values[h] = fmaxf(dot, 0.0f) * weights[row * num_heads + h];
        }
        __syncthreads();

        float sum = 0.0f;
        for (int i = threadIdx.x; i < num_heads; i += blockDim.x)
            sum += head_values[i];
        partials[threadIdx.x] = sum;
        __syncthreads();
        for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
            if (threadIdx.x < offset)
                partials[threadIdx.x] += partials[threadIdx.x + offset];
            __syncthreads();
        }
        result = partials[0];
    }

    if (threadIdx.x == 0)
        store_logit(logits, static_cast<int64_t>(row) * logits_stride + n,
                    result);
}

__device__ __forceinline__ float ue8m0_scale_from_pack(int32_t packed_sf,
                                                       int group_idx) {
    const auto raw = static_cast<uint32_t>(packed_sf);
    const int exp = static_cast<int>((raw >> (8 * group_idx)) & 0xffu);
    return __uint_as_float(static_cast<uint32_t>(exp) << 23);
}

__device__ __forceinline__ float load_fp4(const int8_t* data, int packed_offset,
                                          int32_t packed_sf, int dim) {
    const uint8_t packed = static_cast<uint8_t>(data[packed_offset + dim / 2]);
    const uint8_t code = (dim & 1) ? (packed >> 4) : (packed & 0x0f);
    return fp4_e2m1_to_float(code) *
           ue8m0_scale_from_pack(packed_sf, dim / 32);
}

// Vectorized FP8 mqa_logits for SM_120: 16-byte uint4 loads, hardware
// __nv_fp8x4_e4m3 -> float4 conversion. Specialized for head_dim==128 (DSv4
// indexer head). Drops global-load instruction count by 16x and halves total
// instructions issued per dot-product. KV row reused across heads via register
// cache. kv_scale hoisted out of inner loop (FP32-accumulate associativity is
// preserved well below the 1e-3 tolerance the unit test asserts).
template <typename out_t, int kHeadDim>
__global__ void mqa_logits_fp8_vec_kernel(
    const __nv_fp8_e4m3* __restrict__ q_ptr,
    const __nv_fp8_e4m3* __restrict__ kv_ptr,
    const float* __restrict__ kv_sf_ptr,
    const float* __restrict__ weights,
    const int32_t* __restrict__ cu_seq_len_k_start,
    const int32_t* __restrict__ cu_seq_len_k_end,
    out_t* logits, int seq_len, int seq_len_kv, int num_heads, int out_cols,
    int logits_stride, bool compressed_logits) {
    static_assert(kHeadDim % 16 == 0, "kHeadDim must be a multiple of 16");
    constexpr int kVecsPerHead = kHeadDim / 16;  // 16 fp8 elems / uint4 load

    const int64_t total = static_cast<int64_t>(seq_len) * out_cols;
    for (int64_t linear = blockIdx.x * blockDim.x + threadIdx.x;
         linear < total;
         linear += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        const int m = static_cast<int>(linear / out_cols);
        const int out_col =
            static_cast<int>(linear - static_cast<int64_t>(m) * out_cols);
        const int start = max(0, min(cu_seq_len_k_start[m], seq_len_kv));
        const int end = max(0, min(cu_seq_len_k_end[m], seq_len_kv));
        const int n = compressed_logits ? start + out_col : out_col;

        float result = -INFINITY;
        if (n >= start && n < end) {
            const float kv_scale = kv_sf_ptr[n];

            // Stage KV row into registers — reused across all heads.
            const uint4* kv_vec_ptr = reinterpret_cast<const uint4*>(
                kv_ptr + static_cast<int64_t>(n) * kHeadDim);
            uint4 kv_vecs[kVecsPerHead];
            #pragma unroll
            for (int v = 0; v < kVecsPerHead; ++v) {
                kv_vecs[v] = kv_vec_ptr[v];
            }

            float sum = 0.0f;
            for (int h = 0; h < num_heads; ++h) {
                const uint4* q_vec_ptr = reinterpret_cast<const uint4*>(
                    q_ptr +
                    (static_cast<int64_t>(m) * num_heads + h) * kHeadDim);
                float dot = 0.0f;
                #pragma unroll
                for (int v = 0; v < kVecsPerHead; ++v) {
                    uint4 q4 = q_vec_ptr[v];
                    uint4 k4 = kv_vecs[v];
                    // Each uint32 holds 4 fp8 e4m3 values. HW cvt to float4.
                    float4 qf, kf;
                    qf = static_cast<float4>(
                        *reinterpret_cast<__nv_fp8x4_e4m3*>(&q4.x));
                    kf = static_cast<float4>(
                        *reinterpret_cast<__nv_fp8x4_e4m3*>(&k4.x));
                    dot += qf.x * kf.x + qf.y * kf.y + qf.z * kf.z +
                           qf.w * kf.w;
                    qf = static_cast<float4>(
                        *reinterpret_cast<__nv_fp8x4_e4m3*>(&q4.y));
                    kf = static_cast<float4>(
                        *reinterpret_cast<__nv_fp8x4_e4m3*>(&k4.y));
                    dot += qf.x * kf.x + qf.y * kf.y + qf.z * kf.z +
                           qf.w * kf.w;
                    qf = static_cast<float4>(
                        *reinterpret_cast<__nv_fp8x4_e4m3*>(&q4.z));
                    kf = static_cast<float4>(
                        *reinterpret_cast<__nv_fp8x4_e4m3*>(&k4.z));
                    dot += qf.x * kf.x + qf.y * kf.y + qf.z * kf.z +
                           qf.w * kf.w;
                    qf = static_cast<float4>(
                        *reinterpret_cast<__nv_fp8x4_e4m3*>(&q4.w));
                    kf = static_cast<float4>(
                        *reinterpret_cast<__nv_fp8x4_e4m3*>(&k4.w));
                    dot += qf.x * kf.x + qf.y * kf.y + qf.z * kf.z +
                           qf.w * kf.w;
                }
                sum += fmaxf(dot * kv_scale, 0.0f) * weights[m * num_heads + h];
            }
            result = sum;
        }
        store_logit(logits, static_cast<int64_t>(m) * logits_stride + out_col,
                    result);
    }
}

// FP8 MMA mqa_logits for SM_120: uses mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32
// (SM_89-class fp8 tensor-core instruction, available on SM_120).
//
// **Q-stationary block layout**: 1 block per m, 8 warps share Q[m, :, :] in
// shared memory and each warp picks an n-tile in stride. This amortizes the
// Q DRAM read across all warps in the block, which the prior per-tile design
// failed to do (Q got re-read once per (m, 8-n) tile, blowing the L1 reuse
// window at mid-size contexts and regressing e2e despite passing numerics).
//
// Per warp the work is 4 H-iters × 4 K-iters = 16 mma instructions for one
// (m, 8-n) output tile, identical math to the per-tile design, but A operand
// reads now come from shared memory rather than gmem.
//
// Shared memory layout:
//   q_smem  : uint32_t[64 × 36]  (8 fp8 cols of stride pad = 4 u32, 9 KB)
//   w_smem  : float[64]          (256 B)
// The 33-vs-36 row stride choice: pad = 4 u32 makes 36 ≡ 4 (mod 32 banks) and
// the A-operand bank index simplifies to (lane_id + const) — all 32 lanes hit
// distinct banks. Pad = 1 leaves a residual 4-way conflict.
//
// Cooperative Q load: 256 threads × 8 u32/thread = 2048 u32 = full Q[m,:,:].
// Thread tid loads row tid/4, col_u32 (tid%4)*8 .. (tid%4)*8+7 as 2 uint4s.
//
// PTX m16n8k32 D-fragment (thread = (g, l)):
//   d0 = D[g,   2l]    head=h_base+g,   n_local=2l
//   d1 = D[g,   2l+1]  head=h_base+g,   n_local=2l+1
//   d2 = D[g+8, 2l]    head=h_base+g+8, n_local=2l
//   d3 = D[g+8, 2l+1]  head=h_base+g+8, n_local=2l+1
//
// Boundary handling: n positions outside [start, end) zero out their B column
// so the dot product becomes 0; the writing thread emits -INFINITY based on
// its own n_valid check so the output matches the scalar reference.
//
// Specialized for num_heads == 64, head_dim == 128, fp8_e4m3 — DSv4 indexer.
template <typename out_t>
__global__ void mqa_logits_fp8_mma_kernel(
    const __nv_fp8_e4m3* __restrict__ q_ptr,
    const __nv_fp8_e4m3* __restrict__ kv_ptr,
    const float* __restrict__ kv_sf_ptr,
    const float* __restrict__ weights,
    const int32_t* __restrict__ cu_seq_len_k_start,
    const int32_t* __restrict__ cu_seq_len_k_end,
    out_t* logits, int seq_len, int seq_len_kv, int out_cols, int logits_stride,
    bool compressed_logits) {
    constexpr int kHeadDim = 128;
    constexpr int kNumHeads = 64;
    constexpr int kKTile = 32;
    constexpr int kNTile = 8;
    constexpr int kHTile = 16;
    constexpr int kKIters = kHeadDim / kKTile;          // 4
    constexpr int kHIters = kNumHeads / kHTile;         // 4
    constexpr int kQRowU32 = kHeadDim / 4;              // 32 u32 per row
    constexpr int kQRowStrideU32 = kQRowU32 + 4;        // pad 4: bank-conflict-free A reads
    constexpr int kWarpsPerBlock = 8;

    __shared__ uint32_t q_smem[kNumHeads * kQRowStrideU32];
    __shared__ float    w_smem[kNumHeads];

    const int m = blockIdx.x;
    if (m >= seq_len) return;

    const int tid = threadIdx.x;
    const int warp_in_block = tid >> 5;
    const int lane = tid & 31;
    const int g = lane >> 2;
    const int l = lane & 3;

    // Cooperative Q[m, :, :] -> q_smem.
    {
        const int row = tid >> 2;
        const int col_u32_base = (tid & 3) * 8;
        const uint4* src4 = reinterpret_cast<const uint4*>(
            q_ptr + (static_cast<int64_t>(m) * kNumHeads + row) * kHeadDim
                  + col_u32_base * 4);
        uint32_t* dst = q_smem + row * kQRowStrideU32 + col_u32_base;
        const uint4 a = src4[0];
        const uint4 b = src4[1];
        dst[0] = a.x; dst[1] = a.y; dst[2] = a.z; dst[3] = a.w;
        dst[4] = b.x; dst[5] = b.y; dst[6] = b.z; dst[7] = b.w;
    }
    if (tid < kNumHeads) {
        w_smem[tid] = weights[static_cast<int64_t>(m) * kNumHeads + tid];
    }
    __syncthreads();

    const int start = max(0, min(cu_seq_len_k_start[m], seq_len_kv));
    const int end   = max(0, min(cu_seq_len_k_end[m],   seq_len_kv));
    const int n_tiles_per_m = (out_cols + kNTile - 1) / kNTile;

    for (int n_tile = warp_in_block; n_tile < n_tiles_per_m;
         n_tile += kWarpsPerBlock) {
        const int out_col_base = n_tile * kNTile;

        // n owned by thread (g, l) on the B-operand (col=g of B-tile).
        const int out_col_g = out_col_base + g;
        const int n_for_g = compressed_logits ? start + out_col_g : out_col_g;
        const bool n_g_valid = (n_for_g >= start && n_for_g < end);
        const int safe_n_g = n_g_valid ? n_for_g : 0;

        // Pre-load B for all 4 k_iters into registers.
        uint32_t b_regs[kKIters * 2];
        const __nv_fp8_e4m3* k_row = kv_ptr +
            static_cast<int64_t>(safe_n_g) * kHeadDim;
        #pragma unroll
        for (int k_iter = 0; k_iter < kKIters; ++k_iter) {
            const int k_base = k_iter * kKTile;
            b_regs[k_iter * 2 + 0] = *reinterpret_cast<const uint32_t*>(
                k_row + k_base + 4 * l);
            b_regs[k_iter * 2 + 1] = *reinterpret_cast<const uint32_t*>(
                k_row + k_base + 4 * l + 16);
        }
        if (!n_g_valid) {
            #pragma unroll
            for (int i = 0; i < kKIters * 2; ++i) b_regs[i] = 0u;
        }

        // n positions this thread will eventually write logits for.
        const int out_col_2l = out_col_base + 2 * l;
        const int out_col_2l_p1 = out_col_2l + 1;
        const int n_2l = compressed_logits ? start + out_col_2l : out_col_2l;
        const int n_2l_p1 =
            compressed_logits ? start + out_col_2l_p1 : out_col_2l_p1;
        const bool n_2l_valid = (n_2l >= start && n_2l < end);
        const bool n_2l_p1_valid = (n_2l_p1 >= start && n_2l_p1 < end);
        const float kv_scale_2l =
            n_2l_valid ? kv_sf_ptr[n_2l] : 0.0f;
        const float kv_scale_2l_p1 =
            n_2l_p1_valid ? kv_sf_ptr[n_2l_p1] : 0.0f;

        float acc_2l = 0.0f;
        float acc_2l_p1 = 0.0f;

        #pragma unroll
        for (int h_iter = 0; h_iter < kHIters; ++h_iter) {
            const int h_base = h_iter * kHTile;
            const float w_g  = w_smem[h_base + g];
            const float w_g8 = w_smem[h_base + g + 8];

            float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;

            const int row_g_off  = (h_base + g)     * kQRowStrideU32;
            const int row_g8_off = (h_base + g + 8) * kQRowStrideU32;

            #pragma unroll
            for (int k_iter = 0; k_iter < kKIters; ++k_iter) {
                const int k_u32_base = k_iter * (kKTile / 4);  // 8 u32
                const uint32_t a0 = q_smem[row_g_off  + k_u32_base + l];
                const uint32_t a1 = q_smem[row_g8_off + k_u32_base + l];
                const uint32_t a2 = q_smem[row_g_off  + k_u32_base + l + 4];
                const uint32_t a3 = q_smem[row_g8_off + k_u32_base + l + 4];
                const uint32_t b0 = b_regs[k_iter * 2 + 0];
                const uint32_t b1 = b_regs[k_iter * 2 + 1];

                asm volatile(
                    "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                    "{%0, %1, %2, %3}, "
                    "{%4, %5, %6, %7}, "
                    "{%8, %9}, "
                    "{%0, %1, %2, %3};\n"
                    : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
                    : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                      "r"(b0), "r"(b1));
            }

            const float e0 = fmaxf(d0 * kv_scale_2l,    0.0f) * w_g;
            const float e1 = fmaxf(d1 * kv_scale_2l_p1, 0.0f) * w_g;
            const float e2 = fmaxf(d2 * kv_scale_2l,    0.0f) * w_g8;
            const float e3 = fmaxf(d3 * kv_scale_2l_p1, 0.0f) * w_g8;
            acc_2l    += e0 + e2;
            acc_2l_p1 += e1 + e3;
        }

        // Warp reduce across the 8 lanes that share l (lanes l, l+4, ..., l+28).
        acc_2l    += __shfl_xor_sync(0xffffffffu, acc_2l,    16);
        acc_2l_p1 += __shfl_xor_sync(0xffffffffu, acc_2l_p1, 16);
        acc_2l    += __shfl_xor_sync(0xffffffffu, acc_2l,     8);
        acc_2l_p1 += __shfl_xor_sync(0xffffffffu, acc_2l_p1,  8);
        acc_2l    += __shfl_xor_sync(0xffffffffu, acc_2l,     4);
        acc_2l_p1 += __shfl_xor_sync(0xffffffffu, acc_2l_p1,  4);

        if (g == 0) {
            if (out_col_2l < out_cols) {
                const float v = n_2l_valid ? acc_2l : -INFINITY;
                store_logit(logits,
                            static_cast<int64_t>(m) * logits_stride + out_col_2l,
                            v);
            }
            if (out_col_2l_p1 < out_cols) {
                const float v = n_2l_p1_valid ? acc_2l_p1 : -INFINITY;
                store_logit(logits,
                            static_cast<int64_t>(m) * logits_stride + out_col_2l_p1,
                            v);
            }
        }
    }
}

template <typename out_t, bool kIsFP4>
__global__ void mqa_logits_kernel(const void* q_ptr, const int32_t* q_sf_ptr,
                                  const void* kv_ptr, const void* kv_sf_ptr,
                                  const float* weights,
                                  const int32_t* cu_seq_len_k_start,
                                  const int32_t* cu_seq_len_k_end,
                                  out_t* logits, int seq_len, int seq_len_kv,
                                  int num_heads, int head_dim,
                                  int packed_head_dim, int out_cols,
                                  int logits_stride, bool compressed_logits) {
    const int64_t total = static_cast<int64_t>(seq_len) * out_cols;
    for (int64_t linear = blockIdx.x * blockDim.x + threadIdx.x;
         linear < total; linear += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        const int m = static_cast<int>(linear / out_cols);
        const int out_col = static_cast<int>(linear - static_cast<int64_t>(m) * out_cols);
        const int start = max(0, min(cu_seq_len_k_start[m], seq_len_kv));
        const int end = max(0, min(cu_seq_len_k_end[m], seq_len_kv));
        const int n = compressed_logits ? start + out_col : out_col;

        float result = -INFINITY;
        if (n >= start && n < end) {
            float sum = 0.0f;
            for (int h = 0; h < num_heads; ++h) {
                float dot = 0.0f;
                if constexpr (kIsFP4) {
                    const auto* q = static_cast<const int8_t*>(q_ptr);
                    const auto* kv = static_cast<const int8_t*>(kv_ptr);
                    const int32_t q_sf = q_sf_ptr[m * num_heads + h];
                    const int32_t kv_sf = static_cast<const int32_t*>(kv_sf_ptr)[n];
                    const int q_base = (m * num_heads + h) * packed_head_dim;
                    const int kv_base = n * packed_head_dim;
                    for (int d = 0; d < head_dim; ++d) {
                        dot += load_fp4(q, q_base, q_sf, d) *
                               load_fp4(kv, kv_base, kv_sf, d);
                    }
                } else {
                    const auto* q = static_cast<const __nv_fp8_e4m3*>(q_ptr);
                    const auto* kv = static_cast<const __nv_fp8_e4m3*>(kv_ptr);
                    const float kv_scale = static_cast<const float*>(kv_sf_ptr)[n];
                    const int q_base = (m * num_heads + h) * head_dim;
                    const int kv_base = n * head_dim;
                    for (int d = 0; d < head_dim; ++d) {
                        dot += static_cast<float>(q[q_base + d]) *
                               (static_cast<float>(kv[kv_base + d]) * kv_scale);
                    }
                }
                sum += fmaxf(dot, 0.0f) * weights[m * num_heads + h];
            }
            result = sum;
        }
        store_logit(logits,
                    static_cast<int64_t>(m) * logits_stride + out_col,
                    result);
    }
}

template <typename out_t, bool kIsFP4>
__global__ void paged_mqa_logits_kernel(
    const void* q_ptr, const int32_t* q_sf_ptr, const void* kv_ptr,
    const void* kv_sf_ptr, const float* weights, const int32_t* context_lens,
    const int32_t* block_table, out_t* logits, int batch_size, int next_n,
    int num_heads, int head_dim, int packed_head_dim, int block_kv,
    int kv_stride0, int kv_stride1, int kv_sf_stride0, int block_table_stride,
    int logits_stride, int max_context_len, bool is_context_lens_2d) {
    const int num_rows = batch_size * next_n;
    const int64_t total = static_cast<int64_t>(num_rows) * max_context_len;
    for (int64_t linear = blockIdx.x * blockDim.x + threadIdx.x;
         linear < total; linear += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        const int row = static_cast<int>(linear / max_context_len);
        const int n = static_cast<int>(linear - static_cast<int64_t>(row) * max_context_len);
        const int b = row / next_n;
        const int t = row - b * next_n;
        const int q_limit = is_context_lens_2d
                                ? context_lens[b * next_n + t]
                                : context_lens[b] - next_n + t;

        float result = -INFINITY;
        if (n <= q_limit && n < max_context_len) {
            const int block_offset = n / block_kv;
            const int token_offset = n - block_offset * block_kv;
            const int block_idx = block_table[b * block_table_stride + block_offset];
            float sum = 0.0f;
            for (int h = 0; h < num_heads; ++h) {
                float dot = 0.0f;
                if constexpr (kIsFP4) {
                    const auto* q = static_cast<const int8_t*>(q_ptr);
                    const auto* kv = static_cast<const int8_t*>(kv_ptr);
                    const auto* kv_sf = static_cast<const int32_t*>(kv_sf_ptr);
                    const int32_t q_sf = q_sf_ptr[(row * num_heads) + h];
                    const int32_t k_sf = kv_sf[block_idx * kv_sf_stride0 + token_offset];
                    const int q_base = (row * num_heads + h) * packed_head_dim;
                    const int kv_base = block_idx * kv_stride0 + token_offset * kv_stride1;
                    for (int d = 0; d < head_dim; ++d) {
                        dot += load_fp4(q, q_base, q_sf, d) *
                               load_fp4(kv, kv_base, k_sf, d);
                    }
                } else {
                    const auto* q = static_cast<const __nv_fp8_e4m3*>(q_ptr);
                    const auto* kv = static_cast<const __nv_fp8_e4m3*>(kv_ptr);
                    const auto* kv_sf = static_cast<const float*>(kv_sf_ptr);
                    const float k_scale =
                        kv_sf[block_idx * kv_sf_stride0 + token_offset];
                    const int q_base = (row * num_heads + h) * head_dim;
                    const int kv_base = block_idx * kv_stride0 + token_offset * kv_stride1;
                    for (int d = 0; d < head_dim; ++d) {
                        dot += static_cast<float>(q[q_base + d]) *
                               (static_cast<float>(kv[kv_base + d]) * k_scale);
                    }
                }
                sum += fmaxf(dot, 0.0f) * weights[row * num_heads + h];
            }
            result = sum;
        }
        store_logit(logits, static_cast<int64_t>(row) * logits_stride + n, result);
    }
}

int fallback_grid(int64_t total) {
    constexpr int threads = 256;
    const int64_t blocks = (total + threads - 1) / threads;
    return static_cast<int>(std::min<int64_t>(blocks, 4096));
}

inline bool use_mma_indexer() {
    static const bool v = []() {
        const char* env = std::getenv("DG_SM120_MMA_INDEXER");
        return env != nullptr && env[0] != '\0' && env[0] != '0';
    }();
    return v;
}


template <bool kIsFP4>
void launch_mqa_logits(const torch::Tensor& q, const torch::Tensor& q_sf,
                       const torch::Tensor& kv, const torch::Tensor& kv_sf,
                       const torch::Tensor& weights,
                       const torch::Tensor& cu_seq_len_k_start,
                       const torch::Tensor& cu_seq_len_k_end,
                       const torch::Tensor& logits,
                       const at::ScalarType& logits_dtype, int seq_len,
                       int seq_len_kv, int num_heads, int head_dim, int out_cols,
                       int logits_stride, bool compressed_logits) {
    constexpr int threads = 256;
    const auto stream = at::cuda::getCurrentCUDAStream();
    const int64_t total = static_cast<int64_t>(seq_len) * out_cols;
    const int grid = fallback_grid(total);
    const int packed_head_dim = kIsFP4 ? head_dim / 2 : head_dim;
    const int32_t* q_sf_ptr = nullptr;
    if constexpr (kIsFP4)
        q_sf_ptr = q_sf.data_ptr<int32_t>();

    // FP8 MMA tensor-core fast path: gated behind DG_SM120_MMA_INDEXER=1.
    // Specialized for num_heads == 64, head_dim == 128, fp8_e4m3 (DSv4 indexer).
    // Q-stationary: 1 block per m, 8 warps share Q[m] in shared memory.
    if constexpr (!kIsFP4) {
        if (head_dim == 128 && num_heads == 64 && use_mma_indexer()) {
            constexpr int mma_threads = 256;  // 8 warps per block
            const int mma_blocks = seq_len;
            const auto* q_fp8 =
                reinterpret_cast<const __nv_fp8_e4m3*>(q.data_ptr());
            const auto* kv_fp8 =
                reinterpret_cast<const __nv_fp8_e4m3*>(kv.data_ptr());
            if (logits_dtype == torch::kFloat32) {
                mqa_logits_fp8_mma_kernel<float>
                    <<<mma_blocks, mma_threads, 0, stream>>>(
                        q_fp8, kv_fp8, kv_sf.data_ptr<float>(),
                        weights.data_ptr<float>(),
                        cu_seq_len_k_start.data_ptr<int32_t>(),
                        cu_seq_len_k_end.data_ptr<int32_t>(),
                        logits.data_ptr<float>(), seq_len, seq_len_kv, out_cols,
                        logits_stride, compressed_logits);
                DG_CUDA_RUNTIME_CHECK(cudaGetLastError());
                return;
            } else if (logits_dtype == torch::kBFloat16) {
                mqa_logits_fp8_mma_kernel<__nv_bfloat16>
                    <<<mma_blocks, mma_threads, 0, stream>>>(
                        q_fp8, kv_fp8, kv_sf.data_ptr<float>(),
                        weights.data_ptr<float>(),
                        cu_seq_len_k_start.data_ptr<int32_t>(),
                        cu_seq_len_k_end.data_ptr<int32_t>(),
                        reinterpret_cast<__nv_bfloat16*>(logits.data_ptr()),
                        seq_len, seq_len_kv, out_cols, logits_stride,
                        compressed_logits);
                DG_CUDA_RUNTIME_CHECK(cudaGetLastError());
                return;
            }
        }
    }

    // Vectorized FP8 fast path for head_dim == 128 (DSv4 indexer head dim).
    // Uses 16-byte uint4 loads + hardware fp8x4 -> float4 conversion.
    if constexpr (!kIsFP4) {
        if (head_dim == 128) {
            const auto* q_fp8 =
                reinterpret_cast<const __nv_fp8_e4m3*>(q.data_ptr());
            const auto* kv_fp8 =
                reinterpret_cast<const __nv_fp8_e4m3*>(kv.data_ptr());
            if (logits_dtype == torch::kFloat32) {
                mqa_logits_fp8_vec_kernel<float, 128>
                    <<<grid, threads, 0, stream>>>(
                        q_fp8, kv_fp8, kv_sf.data_ptr<float>(),
                        weights.data_ptr<float>(),
                        cu_seq_len_k_start.data_ptr<int32_t>(),
                        cu_seq_len_k_end.data_ptr<int32_t>(),
                        logits.data_ptr<float>(), seq_len, seq_len_kv,
                        num_heads, out_cols, logits_stride, compressed_logits);
                DG_CUDA_RUNTIME_CHECK(cudaGetLastError());
                return;
            } else if (logits_dtype == torch::kBFloat16) {
                mqa_logits_fp8_vec_kernel<__nv_bfloat16, 128>
                    <<<grid, threads, 0, stream>>>(
                        q_fp8, kv_fp8, kv_sf.data_ptr<float>(),
                        weights.data_ptr<float>(),
                        cu_seq_len_k_start.data_ptr<int32_t>(),
                        cu_seq_len_k_end.data_ptr<int32_t>(),
                        reinterpret_cast<__nv_bfloat16*>(logits.data_ptr()),
                        seq_len, seq_len_kv, num_heads, out_cols, logits_stride,
                        compressed_logits);
                DG_CUDA_RUNTIME_CHECK(cudaGetLastError());
                return;
            }
        }
    }

    if (logits_dtype == torch::kFloat32) {
        mqa_logits_kernel<float, kIsFP4><<<grid, threads, 0, stream>>>(
            q.data_ptr(), q_sf_ptr, kv.data_ptr(), kv_sf.data_ptr(),
            weights.data_ptr<float>(), cu_seq_len_k_start.data_ptr<int32_t>(),
            cu_seq_len_k_end.data_ptr<int32_t>(), logits.data_ptr<float>(),
            seq_len, seq_len_kv, num_heads, head_dim, packed_head_dim, out_cols,
            logits_stride, compressed_logits);
    } else if (logits_dtype == torch::kBFloat16) {
        mqa_logits_kernel<__nv_bfloat16, kIsFP4><<<grid, threads, 0, stream>>>(
            q.data_ptr(), q_sf_ptr, kv.data_ptr(), kv_sf.data_ptr(),
            weights.data_ptr<float>(), cu_seq_len_k_start.data_ptr<int32_t>(),
            cu_seq_len_k_end.data_ptr<int32_t>(),
            reinterpret_cast<__nv_bfloat16*>(logits.data_ptr()), seq_len,
            seq_len_kv, num_heads, head_dim, packed_head_dim, out_cols,
            logits_stride, compressed_logits);
    } else {
        DG_HOST_UNREACHABLE("Unsupported logits dtype for SM120 fallback");
    }
    DG_CUDA_RUNTIME_CHECK(cudaGetLastError());
}

template <bool kIsFP4>
void launch_paged_mqa_logits(
    const torch::Tensor& q, const torch::Tensor& q_sf,
    const torch::Tensor& kv_cache, const torch::Tensor& kv_cache_sf,
    const torch::Tensor& weights, const torch::Tensor& context_lens,
    const torch::Tensor& logits, const torch::Tensor& block_table,
    const at::ScalarType& logits_dtype, int batch_size, int next_n,
    int num_heads, int head_dim, int block_kv, bool is_context_lens_2d,
    int logits_stride, int block_table_stride, int max_context_len) {
    constexpr int threads = 256;
    const auto stream = at::cuda::getCurrentCUDAStream();
    const int64_t total = static_cast<int64_t>(batch_size) * next_n * max_context_len;
    const int grid = fallback_grid(total);
    const int packed_head_dim = kIsFP4 ? head_dim / 2 : head_dim;
    const int kv_stride0 = static_cast<int>(kv_cache.stride(0));
    const int kv_stride1 = static_cast<int>(kv_cache.stride(1));
    const int kv_sf_stride0 = static_cast<int>(kv_cache_sf.stride(0));
    const int32_t* q_sf_ptr = nullptr;
    if constexpr (kIsFP4)
        q_sf_ptr = q_sf.data_ptr<int32_t>();

    if constexpr (!kIsFP4) {
        const dim3 grid(max_context_len, batch_size * next_n);
        if (logits_dtype == torch::kFloat32) {
            paged_fp8_mqa_logits_fast_kernel<float><<<grid, threads, 0, stream>>>(
                reinterpret_cast<const uint8_t*>(q.data_ptr()),
                reinterpret_cast<const uint8_t*>(kv_cache.data_ptr()),
                kv_cache_sf.data_ptr<float>(), weights.data_ptr<float>(),
                context_lens.data_ptr<int32_t>(), block_table.data_ptr<int32_t>(),
                logits.data_ptr<float>(), batch_size, next_n, num_heads,
                head_dim, block_kv, kv_stride0, kv_stride1, kv_sf_stride0,
                block_table_stride, logits_stride, max_context_len,
                is_context_lens_2d);
        } else if (logits_dtype == torch::kBFloat16) {
            paged_fp8_mqa_logits_fast_kernel<__nv_bfloat16>
                <<<grid, threads, 0, stream>>>(
                    reinterpret_cast<const uint8_t*>(q.data_ptr()),
                    reinterpret_cast<const uint8_t*>(kv_cache.data_ptr()),
                    kv_cache_sf.data_ptr<float>(), weights.data_ptr<float>(),
                    context_lens.data_ptr<int32_t>(),
                    block_table.data_ptr<int32_t>(),
                    reinterpret_cast<__nv_bfloat16*>(logits.data_ptr()),
                    batch_size, next_n, num_heads, head_dim, block_kv,
                    kv_stride0, kv_stride1, kv_sf_stride0, block_table_stride,
                    logits_stride, max_context_len, is_context_lens_2d);
        } else {
            DG_HOST_UNREACHABLE("Unsupported logits dtype for SM120 fallback");
        }
        DG_CUDA_RUNTIME_CHECK(cudaGetLastError());
        return;
    }

    if (logits_dtype == torch::kFloat32) {
        paged_mqa_logits_kernel<float, kIsFP4><<<grid, threads, 0, stream>>>(
            q.data_ptr(), q_sf_ptr, kv_cache.data_ptr(), kv_cache_sf.data_ptr(),
            weights.data_ptr<float>(), context_lens.data_ptr<int32_t>(),
            block_table.data_ptr<int32_t>(), logits.data_ptr<float>(), batch_size,
            next_n, num_heads, head_dim, packed_head_dim, block_kv, kv_stride0,
            kv_stride1, kv_sf_stride0, block_table_stride, logits_stride,
            max_context_len, is_context_lens_2d);
    } else if (logits_dtype == torch::kBFloat16) {
        paged_mqa_logits_kernel<__nv_bfloat16, kIsFP4>
            <<<grid, threads, 0, stream>>>(
                q.data_ptr(), q_sf_ptr, kv_cache.data_ptr(),
                kv_cache_sf.data_ptr(), weights.data_ptr<float>(),
                context_lens.data_ptr<int32_t>(), block_table.data_ptr<int32_t>(),
                reinterpret_cast<__nv_bfloat16*>(logits.data_ptr()), batch_size,
                next_n, num_heads, head_dim, packed_head_dim, block_kv,
                kv_stride0, kv_stride1, kv_sf_stride0, block_table_stride,
                logits_stride, max_context_len, is_context_lens_2d);
    } else {
        DG_HOST_UNREACHABLE("Unsupported logits dtype for SM120 fallback");
    }
    DG_CUDA_RUNTIME_CHECK(cudaGetLastError());
}

} // namespace sm120_fallback

void sm120_fp8_mqa_logits_fallback(
    const torch::Tensor& q, const torch::Tensor& kv, const torch::Tensor& kv_sf,
    const torch::Tensor& weights, const torch::Tensor& cu_seq_len_k_start,
    const torch::Tensor& cu_seq_len_k_end, const torch::Tensor& logits,
    const at::ScalarType& logits_dtype, int seq_len, int seq_len_kv,
    int max_seqlen_k, int logits_stride, int num_heads, int head_dim) {
    const int out_cols = max_seqlen_k > 0 ? max_seqlen_k : seq_len_kv;
    sm120_fallback::launch_mqa_logits<false>(
        q, torch::Tensor(), kv, kv_sf, weights, cu_seq_len_k_start,
        cu_seq_len_k_end, logits, logits_dtype, seq_len, seq_len_kv, num_heads,
        head_dim, out_cols, logits_stride, max_seqlen_k > 0);
}

void sm120_fp4_mqa_logits_fallback(
    const torch::Tensor& q, const torch::Tensor& q_sf, const torch::Tensor& kv,
    const torch::Tensor& kv_sf, const torch::Tensor& weights,
    const torch::Tensor& cu_seq_len_k_start,
    const torch::Tensor& cu_seq_len_k_end, const torch::Tensor& logits,
    const at::ScalarType& logits_dtype, int seq_len, int seq_len_kv,
    int max_seqlen_k, int logits_stride, int num_heads, int head_dim) {
    const int out_cols = max_seqlen_k > 0 ? max_seqlen_k : seq_len_kv;
    sm120_fallback::launch_mqa_logits<true>(
        q, q_sf, kv, kv_sf, weights, cu_seq_len_k_start, cu_seq_len_k_end,
        logits, logits_dtype, seq_len, seq_len_kv, num_heads, head_dim, out_cols,
        logits_stride, max_seqlen_k > 0);
}

void sm120_fp8_paged_mqa_logits_fallback(
    const torch::Tensor& q, const torch::Tensor& kv_cache,
    const torch::Tensor& kv_cache_sf, const torch::Tensor& weights,
    const torch::Tensor& context_lens, const torch::Tensor& logits,
    const torch::Tensor& block_table, const at::ScalarType& logits_dtype,
    int batch_size, int next_n, int num_heads, int head_dim, int block_kv,
    bool is_context_lens_2d, int logits_stride, int block_table_stride,
    int max_context_len) {
    sm120_fallback::launch_paged_mqa_logits<false>(
        q, torch::Tensor(), kv_cache, kv_cache_sf, weights, context_lens, logits,
        block_table, logits_dtype, batch_size, next_n, num_heads, head_dim,
        block_kv, is_context_lens_2d, logits_stride, block_table_stride,
        max_context_len);
}

void sm120_fp4_paged_mqa_logits_fallback(
    const torch::Tensor& q, const torch::Tensor& q_sf,
    const torch::Tensor& kv_cache, const torch::Tensor& kv_cache_sf,
    const torch::Tensor& weights, const torch::Tensor& context_lens,
    const torch::Tensor& logits, const torch::Tensor& block_table,
    const at::ScalarType& logits_dtype, int batch_size, int next_n,
    int num_heads, int head_dim, int block_kv, bool is_context_lens_2d,
    int logits_stride, int block_table_stride, int max_context_len) {
    sm120_fallback::launch_paged_mqa_logits<true>(
        q, q_sf, kv_cache, kv_cache_sf, weights, context_lens, logits, block_table,
        logits_dtype, batch_size, next_n, num_heads, head_dim, block_kv,
        is_context_lens_2d, logits_stride, block_table_stride, max_context_len);
}

} // namespace deep_gemm
