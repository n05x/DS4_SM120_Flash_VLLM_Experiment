#include <algorithm>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <limits>
#include <tuple>

#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <pybind11/pybind11.h>
#include <torch/python.h>

#include "jit_kernels/impls/sm120_sparse_mla_decode.hpp"
#include "sm120_profile.hpp"
#include "utils/exception.hpp"

namespace deep_gemm {
namespace sm120_mla {
namespace {

constexpr int kHeadDim = 512;
constexpr int kFp8Dim = 448;
constexpr int kBf16Dim = 64;
constexpr int kQuantBlock = 64;
constexpr int kNumQuantBlocks = 7;
constexpr int kTokenDataBytes = kFp8Dim + kBf16Dim * 2;
constexpr int kScaleBytes = 8;
constexpr int kThreads = 256;
constexpr int kWarpSize = 32;
constexpr int kScoreGroupSize = 8;
constexpr int kScoreCandidatesPerBlock = kThreads / kScoreGroupSize;
constexpr int kGroupedScoreGroupSize = 16;
constexpr int kGroupedScoreGroups = kThreads / kGroupedScoreGroupSize;
constexpr int kMaxGroupedCandidateSlots = 768;

__device__ __forceinline__ float decode_ue8m0_scale(uint8_t exponent) {
    if (exponent == 0)
        return 0.0f;
    return exp2f(static_cast<float>(exponent) - 127.0f);
}

__device__ __forceinline__ float fp8_e4m3fn_to_float(uint8_t raw) {
    const uint8_t mag = raw & 0x7fu;
    if (mag == 0)
        return 0.0f;
    const int fp8_exp = static_cast<int>((mag >> 3) & 0x0fu);
    const int mant = static_cast<int>(mag & 0x07u);
    const float value =
        fp8_exp == 0
            ? ldexpf(static_cast<float>(mant), -9)
            : ldexpf(1.0f + static_cast<float>(mant) * 0.125f,
                     fp8_exp - 7);
    return raw & 0x80u ? -value : value;
}

__device__ __forceinline__ const uint8_t* token_ptr_from_linear(
    const uint8_t* cache_flat, int64_t block_stride_bytes, int block_size,
    int64_t linear_index) {
    const int64_t block_id = linear_index / block_size;
    const int block_offset =
        static_cast<int>(linear_index - block_id * block_size);
    const uint8_t* block = cache_flat + block_id * block_stride_bytes;
    return block + static_cast<int64_t>(block_offset) * kTokenDataBytes;
}

__device__ __forceinline__ const uint8_t* scale_ptr_from_linear(
    const uint8_t* cache_flat, int64_t block_stride_bytes, int block_size,
    int64_t linear_index) {
    const int64_t block_id = linear_index / block_size;
    const int block_offset =
        static_cast<int>(linear_index - block_id * block_size);
    const uint8_t* block = cache_flat + block_id * block_stride_bytes;
    return block + static_cast<int64_t>(block_size) * kTokenDataBytes +
           static_cast<int64_t>(block_offset) * kScaleBytes;
}

__device__ __forceinline__ float load_cache_value_from_token(
    const uint8_t* token, const float* scales, int dim) {
    if (dim < kFp8Dim) {
        return fp8_e4m3fn_to_float(token[dim]) * scales[dim / kQuantBlock];
    }

    const auto* rope = reinterpret_cast<const __nv_bfloat16*>(token + kFp8Dim);
    return __bfloat162float(rope[dim - kFp8Dim]);
}

template <typename T>
__device__ __forceinline__ T* align_shared_ptr(unsigned char*& ptr) {
    constexpr uintptr_t alignment = alignof(T);
    uintptr_t value = reinterpret_cast<uintptr_t>(ptr);
    value = (value + alignment - 1) & ~(alignment - 1);
    ptr = reinterpret_cast<unsigned char*>(value);
    return reinterpret_cast<T*>(ptr);
}

__host__ __forceinline__ size_t align_up_size(size_t value, size_t alignment) {
    return (value + alignment - 1) & ~(alignment - 1);
}

__host__ __forceinline__ size_t grouped_decode_shared_bytes(
    int candidate_slots) {
    size_t bytes = static_cast<size_t>(kHeadDim) * sizeof(float);
    bytes += static_cast<size_t>(candidate_slots) * sizeof(float);
    bytes = align_up_size(bytes, alignof(int64_t));
    bytes += static_cast<size_t>(candidate_slots) * sizeof(int64_t);
    bytes = align_up_size(bytes, alignof(int));
    bytes += static_cast<size_t>(candidate_slots) * sizeof(int);
    bytes = align_up_size(bytes, alignof(float));
    bytes += static_cast<size_t>(candidate_slots) * kNumQuantBlocks *
             sizeof(float);
    return bytes;
}

__host__ __forceinline__ size_t full_context_decode_shared_bytes(
    int candidate_slots) {
    size_t bytes = static_cast<size_t>(kHeadDim) * sizeof(float);
    bytes += static_cast<size_t>(candidate_slots) * sizeof(float);
    bytes = align_up_size(bytes, alignof(int64_t));
    bytes += static_cast<size_t>(candidate_slots) * sizeof(int64_t);
    bytes = align_up_size(bytes, alignof(float));
    bytes += static_cast<size_t>(candidate_slots) * kNumQuantBlocks *
             sizeof(float);
    return bytes;
}

__host__ __forceinline__ int sm120_active_heads(int num_heads) {
    const char* env = std::getenv("DG_SM120_ACTIVE_HEADS");
    if (env == nullptr || env[0] == '\0')
        return num_heads;
    char* end = nullptr;
    const long value = std::strtol(env, &end, 10);
    if (end == env || value <= 0)
        return num_heads;
    return std::min(num_heads, static_cast<int>(value));
}

template <typename T>
__device__ __forceinline__ float load_q_value(const T* q, int64_t offset) {
    return static_cast<float>(q[offset]);
}

template <>
__device__ __forceinline__ float load_q_value<__nv_bfloat16>(
    const __nv_bfloat16* q, int64_t offset) {
    return __bfloat162float(q[offset]);
}

template <>
__device__ __forceinline__ float load_q_value<half>(const half* q,
                                                    int64_t offset) {
    return __half2float(q[offset]);
}

template <typename T>
__device__ __forceinline__ void store_out_value(T* out, int64_t offset,
                                                float value) {
    out[offset] = static_cast<T>(value);
}

template <>
__device__ __forceinline__ void store_out_value<__nv_bfloat16>(
    __nv_bfloat16* out, int64_t offset, float value) {
    out[offset] = __float2bfloat16(value);
}

template <>
__device__ __forceinline__ void store_out_value<half>(half* out, int64_t offset,
                                                      float value) {
    out[offset] = __float2half(value);
}

__device__ __forceinline__ float load_cache_value(const uint8_t* cache_flat,
                                                  int64_t block_stride_bytes,
                                                  int block_size,
                                                  int64_t linear_index,
                                                  int dim) {
    const uint8_t* token = token_ptr_from_linear(
        cache_flat, block_stride_bytes, block_size, linear_index);

    if (dim < kFp8Dim) {
        const uint8_t* scales = scale_ptr_from_linear(
            cache_flat, block_stride_bytes, block_size, linear_index);
        return fp8_e4m3fn_to_float(token[dim]) *
               decode_ue8m0_scale(scales[dim / kQuantBlock]);
    }

    const auto* rope = reinterpret_cast<const __nv_bfloat16*>(token + kFp8Dim);
    return __bfloat162float(rope[dim - kFp8Dim]);
}

template <typename T>
__device__ __forceinline__ float load_workspace_value(const T* kv,
                                                      int64_t offset) {
    return static_cast<float>(kv[offset]);
}

template <>
__device__ __forceinline__ float load_workspace_value<__nv_bfloat16>(
    const __nv_bfloat16* kv, int64_t offset) {
    return __bfloat162float(kv[offset]);
}

template <>
__device__ __forceinline__ float load_workspace_value<half>(
    const half* kv, int64_t offset) {
    return __half2float(kv[offset]);
}

template <typename index_t>
__device__ __forceinline__ int64_t load_index(const void* ptr, int64_t offset) {
    return static_cast<int64_t>(static_cast<const index_t*>(ptr)[offset]);
}

__device__ __forceinline__ int load_length_value(const void* ptr, int kind,
                                                 int64_t offset,
                                                 int default_value) {
    if (ptr == nullptr)
        return default_value;
    if (kind == 0)
        return static_cast<int>(static_cast<const int32_t*>(ptr)[offset]);
    return static_cast<int>(static_cast<const int64_t*>(ptr)[offset]);
}

template <typename out_t, typename seq_t, typename block_t, typename gather_t>
__global__ void dequantize_gather_k_cache_kernel(
    out_t* __restrict__ out, const uint8_t* __restrict__ k_cache,
    const seq_t* __restrict__ seq_lens,
    const gather_t* __restrict__ gather_lens,
    const block_t* __restrict__ block_table, int num_reqs, int max_out_rows,
    int head_dim, int block_size, int offset, int64_t cache_blocks,
    int64_t cache_stride0_bytes, int64_t out_stride_b, int64_t out_stride_s,
    int64_t out_stride_d, int64_t block_stride_b, int64_t block_stride_s) {
    const int dim = blockIdx.x * blockDim.x + threadIdx.x;
    const int out_idx = blockIdx.y;
    const int batch_idx = blockIdx.z;
    if (batch_idx >= num_reqs || out_idx >= max_out_rows || dim >= head_dim)
        return;

    const int seq_len = max(0, static_cast<int>(seq_lens[batch_idx]));
    int gather_len = gather_lens == nullptr
                         ? seq_len
                         : max(0, static_cast<int>(gather_lens[batch_idx]));
    gather_len = min(gather_len, max_out_rows);

    float value = 0.0f;
    if (out_idx < gather_len) {
        const int start_pos = max(0, seq_len - gather_len);
        const int pos = start_pos + out_idx;
        const int block_in_seq = pos / block_size;
        const int pos_in_block = pos - block_in_seq * block_size;
        const int64_t physical_block =
            static_cast<int64_t>(block_table[static_cast<int64_t>(batch_idx) *
                                             block_stride_b +
                                             static_cast<int64_t>(block_in_seq) *
                                             block_stride_s]);
        if (physical_block >= 0 && physical_block < cache_blocks &&
            pos_in_block >= 0 && pos_in_block < block_size) {
            const int64_t linear_index =
                physical_block * static_cast<int64_t>(block_size) + pos_in_block;
            value = load_cache_value(k_cache, cache_stride0_bytes, block_size,
                                     linear_index, dim);
        }
    }

    store_out_value<out_t>(
        out, static_cast<int64_t>(batch_idx) * out_stride_b +
                 static_cast<int64_t>(offset + out_idx) * out_stride_s +
                 static_cast<int64_t>(dim) * out_stride_d,
        value);
}

template <typename out_t, typename index_t, typename len_t>
__global__ void dequantize_gather_indexed_k_cache_kernel(
    out_t* __restrict__ out, const uint8_t* __restrict__ k_cache,
    const void* __restrict__ indices, const len_t* __restrict__ topk_length,
    int batch_size, int topk, int head_dim, int block_size, int offset,
    int64_t cache_blocks, int64_t cache_stride0_bytes, int64_t out_stride_b,
    int64_t out_stride_s, int64_t out_stride_d, int64_t index_stride_b,
    int64_t index_stride_s, int64_t index_stride_k, bool indices_3d) {
    const int dim = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y;
    const int b = blockIdx.z;
    if (b >= batch_size || j >= topk || dim >= head_dim)
        return;

    const int limit =
        topk_length == nullptr ? topk : max(0, min(topk, static_cast<int>(topk_length[b])));
    float value = 0.0f;
    if (j < limit) {
        const int64_t index_offset =
            static_cast<int64_t>(b) * index_stride_b +
            (indices_3d ? static_cast<int64_t>(0) * index_stride_s : 0) +
            static_cast<int64_t>(j) * index_stride_k;
        const int64_t linear = load_index<index_t>(indices, index_offset);
        if (linear >= 0 && linear < cache_blocks * static_cast<int64_t>(block_size)) {
            value = load_cache_value(k_cache, cache_stride0_bytes, block_size,
                                     linear, dim);
        }
    }

    store_out_value<out_t>(
        out, static_cast<int64_t>(b) * out_stride_b +
                 static_cast<int64_t>(offset + j) * out_stride_s +
                 static_cast<int64_t>(dim) * out_stride_d,
        value);
}

__global__ void fill_decode_all_indices_kernel(
    int32_t* __restrict__ out, const int32_t* __restrict__ seq_lens,
    int num_rows, int next_n, int topk, int64_t out_stride0,
    int64_t out_stride1, bool seq_lens_is_2d) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y;
    if (row >= num_rows || col >= topk)
        return;

    const int batch_idx = row / next_n;
    const int next_idx = row - batch_idx * next_n;
    const int seq_len =
        seq_lens_is_2d
            ? seq_lens[row]
            : seq_lens[batch_idx] - next_n + next_idx + 1;
    const int row_end = max(0, seq_len);
    out[static_cast<int64_t>(row) * out_stride0 +
        static_cast<int64_t>(col) * out_stride1] =
        col < row_end ? col : -1;
}

template <typename q_t, typename out_t, typename index_t>
__global__ void sparse_mla_decode_kernel(
    const q_t* __restrict__ q, const uint8_t* __restrict__ k_cache,
    const void* __restrict__ indices, const int64_t* __restrict__ topk_length,
    const float* __restrict__ attn_sink,
    const uint8_t* __restrict__ extra_k_cache,
    const void* __restrict__ extra_indices,
    const int64_t* __restrict__ extra_topk_length, out_t* __restrict__ out,
    float* __restrict__ lse, int batch_size, int active_heads, int num_heads,
    int block_size, int64_t cache_blocks, int64_t extra_cache_blocks,
    int main_topk, int extra_topk, int64_t q_stride_b, int64_t q_stride_s,
    int64_t q_stride_h, int64_t q_stride_d,
    int64_t out_stride_b, int64_t out_stride_s, int64_t out_stride_h,
    int64_t out_stride_d, int64_t index_stride_b, int64_t index_stride_s,
    int64_t index_stride_k, int64_t extra_index_stride_b,
    int64_t extra_index_stride_s, int64_t extra_index_stride_k,
    int64_t cache_stride0_bytes, int64_t extra_cache_stride0_bytes,
    float softmax_scale) {
    extern __shared__ float scores[];
    const int bh = blockIdx.x;
    const int b = bh / active_heads;
    const int h = bh - b * active_heads;
    if (b >= batch_size)
        return;

    const int main_limit = topk_length == nullptr
                               ? main_topk
                               : max(0, min(main_topk, static_cast<int>(topk_length[b])));
    const int extra_limit =
        (extra_k_cache == nullptr || extra_indices == nullptr)
            ? 0
            : (extra_topk_length == nullptr
                   ? extra_topk
                   : max(0, min(extra_topk,
                                static_cast<int>(extra_topk_length[b]))));
    const int candidate_count = main_limit + extra_limit;
    const int64_t max_main_linear = cache_blocks * block_size;

    float local_max = -INFINITY;
    for (int j = threadIdx.x; j < candidate_count; j += blockDim.x) {
        const bool is_extra = j >= main_limit;
        const int local_j = is_extra ? j - main_limit : j;
        const void* index_ptr = is_extra ? extra_indices : indices;
        const int64_t index_offset =
            is_extra ? (static_cast<int64_t>(b) * extra_index_stride_b +
                        static_cast<int64_t>(local_j) * extra_index_stride_k)
                     : (static_cast<int64_t>(b) * index_stride_b +
                        static_cast<int64_t>(local_j) * index_stride_k);
        const int64_t linear = load_index<index_t>(index_ptr, index_offset);
        const int64_t max_linear =
            is_extra ? extra_cache_blocks * block_size : max_main_linear;
        float score = -INFINITY;
        if (linear >= 0 && linear < max_linear) {
            const uint8_t* cache_ptr = is_extra ? extra_k_cache : k_cache;
            const int64_t cache_stride =
                is_extra ? extra_cache_stride0_bytes : cache_stride0_bytes;
            float partial = 0.0f;
            for (int d = 0; d < kHeadDim; ++d) {
                const float qv = load_q_value<q_t>(
                    q, static_cast<int64_t>(b) * q_stride_b +
                           static_cast<int64_t>(h) * q_stride_h +
                           static_cast<int64_t>(d) * q_stride_d);
                partial += qv *
                           load_cache_value(cache_ptr, cache_stride, block_size,
                                            linear, d);
            }
            score = partial * softmax_scale;
        }
        scores[j] = score;
        local_max = fmaxf(local_max, score);
    }

    __shared__ float max_buf[kThreads];
    max_buf[threadIdx.x] = local_max;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset)
            max_buf[threadIdx.x] = fmaxf(max_buf[threadIdx.x],
                                         max_buf[threadIdx.x + offset]);
        __syncthreads();
    }
    const float row_max = max_buf[0];

    float local_sum = 0.0f;
    for (int j = threadIdx.x; j < candidate_count; j += blockDim.x) {
        const float p = isfinite(row_max) ? expf(scores[j] - row_max) : 0.0f;
        scores[j] = p;
        local_sum += p;
    }
    __shared__ float sum_buf[kThreads];
    sum_buf[threadIdx.x] = local_sum;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset)
            sum_buf[threadIdx.x] += sum_buf[threadIdx.x + offset];
        __syncthreads();
    }
    const float row_sum = sum_buf[0];
    const float row_lse = row_sum > 0.0f ? logf(row_sum) + row_max : -INFINITY;

    const float sink =
        attn_sink == nullptr ? 0.0f : attn_sink[h];
    const float sink_gate =
        attn_sink == nullptr ? 1.0f : 1.0f / (1.0f + expf(-(row_lse - sink)));

    for (int d = threadIdx.x; d < kHeadDim; d += blockDim.x) {
        float accum = 0.0f;
        if (row_sum > 0.0f) {
            for (int j = 0; j < candidate_count; ++j) {
                const float p = scores[j] / row_sum;
                const bool is_extra = j >= main_limit;
                const int local_j = is_extra ? j - main_limit : j;
                const void* index_ptr = is_extra ? extra_indices : indices;
                const int64_t index_offset =
                    is_extra ? (static_cast<int64_t>(b) * extra_index_stride_b +
                                static_cast<int64_t>(local_j) *
                                    extra_index_stride_k)
                             : (static_cast<int64_t>(b) * index_stride_b +
                                static_cast<int64_t>(local_j) * index_stride_k);
                const int64_t linear = load_index<index_t>(index_ptr, index_offset);
                const int64_t max_linear =
                    is_extra ? extra_cache_blocks * block_size : max_main_linear;
                if (linear < 0 || linear >= max_linear)
                    continue;
                const uint8_t* cache_ptr = is_extra ? extra_k_cache : k_cache;
                const int64_t cache_stride =
                    is_extra ? extra_cache_stride0_bytes : cache_stride0_bytes;
                accum += p * load_cache_value(cache_ptr, cache_stride, block_size,
                                              linear, d);
            }
        }
        store_out_value<out_t>(
            out, static_cast<int64_t>(b) * out_stride_b +
                     static_cast<int64_t>(h) * out_stride_h +
                     static_cast<int64_t>(d) * out_stride_d,
            accum * sink_gate);
    }

    if (threadIdx.x == 0)
        lse[static_cast<int64_t>(b) * num_heads + h] = row_lse;
}

template <typename q_t, typename out_t, typename index_t>
__global__ void sparse_mla_decode_grouped_kernel(
    const q_t* __restrict__ q, const uint8_t* __restrict__ k_cache,
    const void* __restrict__ indices, const int64_t* __restrict__ topk_length,
    const float* __restrict__ attn_sink,
    const uint8_t* __restrict__ extra_k_cache,
    const void* __restrict__ extra_indices,
    const int64_t* __restrict__ extra_topk_length, out_t* __restrict__ out,
    float* __restrict__ lse, int batch_size, int active_heads, int num_heads,
    int block_size, int64_t cache_blocks, int64_t extra_cache_blocks,
    int main_topk, int extra_topk, int candidate_slots, int64_t q_stride_b,
    int64_t q_stride_s, int64_t q_stride_h, int64_t q_stride_d,
    int64_t out_stride_b, int64_t out_stride_s, int64_t out_stride_h,
    int64_t out_stride_d, int64_t index_stride_b, int64_t index_stride_s,
    int64_t index_stride_k, int64_t extra_index_stride_b,
    int64_t extra_index_stride_s, int64_t extra_index_stride_k,
    int64_t cache_stride0_bytes, int64_t extra_cache_stride0_bytes,
    float softmax_scale) {
    extern __shared__ unsigned char smem[];
    unsigned char* cursor = smem;
    float* q_s = reinterpret_cast<float*>(cursor);
    cursor += static_cast<size_t>(kHeadDim) * sizeof(float);
    float* scores = reinterpret_cast<float*>(cursor);
    cursor += static_cast<size_t>(candidate_slots) * sizeof(float);
    int64_t* linear_s = align_shared_ptr<int64_t>(cursor);
    cursor += static_cast<size_t>(candidate_slots) * sizeof(int64_t);
    int* source_s = align_shared_ptr<int>(cursor);
    cursor += static_cast<size_t>(candidate_slots) * sizeof(int);
    float* scale_s = align_shared_ptr<float>(cursor);

    const int bh = blockIdx.x;
    const int b = bh / active_heads;
    const int h = bh - b * active_heads;
    if (b >= batch_size)
        return;

    for (int d = threadIdx.x; d < kHeadDim; d += blockDim.x) {
        q_s[d] = load_q_value<q_t>(
            q, static_cast<int64_t>(b) * q_stride_b +
                   static_cast<int64_t>(h) * q_stride_h +
                   static_cast<int64_t>(d) * q_stride_d);
    }

    const int main_limit = topk_length == nullptr
                               ? main_topk
                               : max(0, min(main_topk, static_cast<int>(topk_length[b])));
    const int extra_limit =
        (extra_k_cache == nullptr || extra_indices == nullptr)
            ? 0
            : (extra_topk_length == nullptr
                   ? extra_topk
                   : max(0, min(extra_topk,
                                static_cast<int>(extra_topk_length[b]))));
    const int candidate_count = main_limit + extra_limit;
    const int64_t max_main_linear = cache_blocks * block_size;

    for (int j = threadIdx.x; j < candidate_count; j += blockDim.x) {
        const bool is_extra = j >= main_limit;
        const int local_j = is_extra ? j - main_limit : j;
        const void* index_ptr = is_extra ? extra_indices : indices;
        const int64_t index_offset =
            is_extra ? (static_cast<int64_t>(b) * extra_index_stride_b +
                        static_cast<int64_t>(local_j) * extra_index_stride_k)
                     : (static_cast<int64_t>(b) * index_stride_b +
                        static_cast<int64_t>(local_j) * index_stride_k);
        const int64_t linear = load_index<index_t>(index_ptr, index_offset);
        const int64_t max_linear =
            is_extra ? extra_cache_blocks * block_size : max_main_linear;
        const bool valid = linear >= 0 && linear < max_linear;
        linear_s[j] = valid ? linear : -1;
        source_s[j] = is_extra ? 1 : 0;

        float* cand_scales = scale_s + static_cast<int64_t>(j) * kNumQuantBlocks;
        #pragma unroll
        for (int s = 0; s < kNumQuantBlocks; ++s)
            cand_scales[s] = 0.0f;
        if (valid) {
            const uint8_t* cache_ptr = is_extra ? extra_k_cache : k_cache;
            const int64_t cache_stride =
                is_extra ? extra_cache_stride0_bytes : cache_stride0_bytes;
            const uint8_t* scales = scale_ptr_from_linear(
                cache_ptr, cache_stride, block_size, linear);
            #pragma unroll
            for (int s = 0; s < kNumQuantBlocks; ++s)
                cand_scales[s] = decode_ue8m0_scale(scales[s]);
        }
    }
    __syncthreads();

    const int group_id = threadIdx.x / kGroupedScoreGroupSize;
    const int lane = threadIdx.x - group_id * kGroupedScoreGroupSize;
    for (int base = 0; base < candidate_count; base += kGroupedScoreGroups) {
        const int j = base + group_id;
        float partial = 0.0f;
        bool valid = false;
        if (j < candidate_count && linear_s[j] >= 0) {
            valid = true;
            const bool is_extra = source_s[j] != 0;
            const uint8_t* cache_ptr = is_extra ? extra_k_cache : k_cache;
            const int64_t cache_stride =
                is_extra ? extra_cache_stride0_bytes : cache_stride0_bytes;
            const uint8_t* token = token_ptr_from_linear(
                cache_ptr, cache_stride, block_size, linear_s[j]);
            const float* cand_scales =
                scale_s + static_cast<int64_t>(j) * kNumQuantBlocks;
            for (int d = lane; d < kHeadDim; d += kGroupedScoreGroupSize) {
                partial += q_s[d] *
                           load_cache_value_from_token(token, cand_scales, d);
            }
        }

        for (int offset = kGroupedScoreGroupSize / 2; offset > 0; offset >>= 1)
            partial += __shfl_down_sync(0xffffffffu, partial, offset,
                                        kGroupedScoreGroupSize);
        if (lane == 0 && j < candidate_count)
            scores[j] = valid ? partial * softmax_scale : -INFINITY;
    }
    __syncthreads();

    __shared__ float reduce_buf[kThreads];
    float local_max = -INFINITY;
    for (int j = threadIdx.x; j < candidate_count; j += blockDim.x)
        local_max = fmaxf(local_max, scores[j]);
    reduce_buf[threadIdx.x] = local_max;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset)
            reduce_buf[threadIdx.x] =
                fmaxf(reduce_buf[threadIdx.x], reduce_buf[threadIdx.x + offset]);
        __syncthreads();
    }
    const float row_max = reduce_buf[0];

    float local_sum = 0.0f;
    for (int j = threadIdx.x; j < candidate_count; j += blockDim.x) {
        const float p = isfinite(row_max) ? expf(scores[j] - row_max) : 0.0f;
        scores[j] = p;
        local_sum += p;
    }
    reduce_buf[threadIdx.x] = local_sum;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset)
            reduce_buf[threadIdx.x] += reduce_buf[threadIdx.x + offset];
        __syncthreads();
    }
    const float row_sum = reduce_buf[0];
    const float row_lse = row_sum > 0.0f ? logf(row_sum) + row_max : -INFINITY;
    for (int j = threadIdx.x; j < candidate_count; j += blockDim.x)
        scores[j] = row_sum > 0.0f ? scores[j] / row_sum : 0.0f;
    __syncthreads();

    const float sink = attn_sink == nullptr ? 0.0f : attn_sink[h];
    const float sink_gate =
        attn_sink == nullptr ? 1.0f : 1.0f / (1.0f + expf(-(row_lse - sink)));

    for (int d = threadIdx.x; d < kHeadDim; d += blockDim.x) {
        float accum = 0.0f;
        if (row_sum > 0.0f) {
            for (int j = 0; j < candidate_count; ++j) {
                const float p = scores[j];
                if (p == 0.0f || linear_s[j] < 0)
                    continue;
                const bool is_extra = source_s[j] != 0;
                const uint8_t* cache_ptr = is_extra ? extra_k_cache : k_cache;
                const int64_t cache_stride =
                    is_extra ? extra_cache_stride0_bytes : cache_stride0_bytes;
                const uint8_t* token = token_ptr_from_linear(
                    cache_ptr, cache_stride, block_size, linear_s[j]);
                const float* cand_scales =
                    scale_s + static_cast<int64_t>(j) * kNumQuantBlocks;
                accum += p * load_cache_value_from_token(token, cand_scales, d);
            }
        }
        store_out_value<out_t>(
            out, static_cast<int64_t>(b) * out_stride_b +
                     static_cast<int64_t>(h) * out_stride_h +
                     static_cast<int64_t>(d) * out_stride_d,
            accum * sink_gate);
    }

    if (threadIdx.x == 0)
        lse[static_cast<int64_t>(b) * num_heads + h] = row_lse;
}

template <typename q_t, typename kv_t, typename out_t>
__global__ void sparse_mla_decode_workspace_kernel(
    const q_t* __restrict__ q, const kv_t* __restrict__ kv_workspace,
    const void* __restrict__ topk_length,
    const void* __restrict__ extra_topk_length,
    const float* __restrict__ attn_sink, out_t* __restrict__ out,
    float* __restrict__ lse, int batch_size, int active_heads, int num_heads,
    int main_topk, int extra_topk, int candidate_slots, int64_t q_stride_b,
    int64_t q_stride_s, int64_t q_stride_h, int64_t q_stride_d,
    int64_t kv_stride_b, int64_t kv_stride_s, int64_t kv_stride_d,
    int64_t out_stride_b, int64_t out_stride_s, int64_t out_stride_h,
    int64_t out_stride_d, int topk_length_kind, int extra_topk_length_kind,
    float softmax_scale) {
    extern __shared__ unsigned char smem[];
    auto* q_s = reinterpret_cast<float*>(smem);
    auto* scores = reinterpret_cast<float*>(
        smem + static_cast<size_t>(kHeadDim) * sizeof(float));

    const int bh = blockIdx.x;
    const int b = bh / num_heads;
    const int h = bh - b * num_heads;
    if (b >= batch_size)
        return;

    if (h >= active_heads) {
        for (int d = threadIdx.x; d < kHeadDim; d += blockDim.x) {
            store_out_value<out_t>(
                out, static_cast<int64_t>(b) * out_stride_b +
                         static_cast<int64_t>(h) * out_stride_h +
                         static_cast<int64_t>(d) * out_stride_d,
                0.0f);
        }
        if (threadIdx.x == 0)
            lse[static_cast<int64_t>(b) * num_heads + h] = -INFINITY;
        return;
    }

    for (int d = threadIdx.x; d < kHeadDim; d += blockDim.x) {
        q_s[d] = load_q_value<q_t>(
            q, static_cast<int64_t>(b) * q_stride_b +
                   static_cast<int64_t>(0) * q_stride_s +
                   static_cast<int64_t>(h) * q_stride_h +
                   static_cast<int64_t>(d) * q_stride_d);
    }
    __syncthreads();

    const int main_limit = max(
        0, min(main_topk,
               load_length_value(topk_length, topk_length_kind, b, main_topk)));
    const int extra_limit =
        max(0, min(extra_topk,
                   load_length_value(extra_topk_length, extra_topk_length_kind,
                                     b, extra_topk)));
    const int valid_count = main_limit + extra_limit;

    const int group_id = threadIdx.x / kGroupedScoreGroupSize;
    const int lane = threadIdx.x - group_id * kGroupedScoreGroupSize;
    for (int base = 0; base < candidate_slots; base += kGroupedScoreGroups) {
        const int j = base + group_id;
        float partial = 0.0f;
        const bool valid =
            j < main_limit ||
            (j >= main_topk && j < main_topk + extra_limit);
        if (j < candidate_slots && valid) {
            for (int d = lane; d < kHeadDim; d += kGroupedScoreGroupSize) {
                partial += q_s[d] *
                           load_workspace_value<kv_t>(
                               kv_workspace,
                               static_cast<int64_t>(b) * kv_stride_b +
                                   static_cast<int64_t>(j) * kv_stride_s +
                                   static_cast<int64_t>(d) * kv_stride_d);
            }
        }

        for (int offset = kGroupedScoreGroupSize / 2; offset > 0; offset >>= 1)
            partial += __shfl_down_sync(0xffffffffu, partial, offset,
                                        kGroupedScoreGroupSize);
        if (lane == 0 && j < candidate_slots)
            scores[j] = valid ? partial * softmax_scale : -INFINITY;
    }
    __syncthreads();

    __shared__ float reduce_buf[kThreads];
    float local_max = -INFINITY;
    for (int j = threadIdx.x; j < candidate_slots; j += blockDim.x)
        local_max = fmaxf(local_max, scores[j]);
    reduce_buf[threadIdx.x] = local_max;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset)
            reduce_buf[threadIdx.x] =
                fmaxf(reduce_buf[threadIdx.x], reduce_buf[threadIdx.x + offset]);
        __syncthreads();
    }
    const float row_max = reduce_buf[0];

    float local_sum = 0.0f;
    for (int j = threadIdx.x; j < candidate_slots; j += blockDim.x) {
        const float p = isfinite(row_max) ? expf(scores[j] - row_max) : 0.0f;
        scores[j] = p;
        local_sum += p;
    }
    reduce_buf[threadIdx.x] = local_sum;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset)
            reduce_buf[threadIdx.x] += reduce_buf[threadIdx.x + offset];
        __syncthreads();
    }
    const float row_sum = reduce_buf[0];
    const float row_lse = row_sum > 0.0f ? logf(row_sum) + row_max : -INFINITY;
    for (int j = threadIdx.x; j < candidate_slots; j += blockDim.x)
        scores[j] = row_sum > 0.0f ? scores[j] / row_sum : 0.0f;
    __syncthreads();

    const float sink = attn_sink == nullptr ? 0.0f : attn_sink[h];
    const float sink_gate =
        attn_sink == nullptr ? 1.0f : 1.0f / (1.0f + expf(-(row_lse - sink)));

    for (int d = threadIdx.x; d < kHeadDim; d += blockDim.x) {
        float accum = 0.0f;
        if (valid_count > 0 && row_sum > 0.0f) {
            for (int j = 0; j < main_limit; ++j) {
                accum += scores[j] *
                         load_workspace_value<kv_t>(
                             kv_workspace,
                             static_cast<int64_t>(b) * kv_stride_b +
                                 static_cast<int64_t>(j) * kv_stride_s +
                                 static_cast<int64_t>(d) * kv_stride_d);
            }
            for (int j = 0; j < extra_limit; ++j) {
                const int workspace_j = main_topk + j;
                accum += scores[workspace_j] *
                         load_workspace_value<kv_t>(
                             kv_workspace,
                             static_cast<int64_t>(b) * kv_stride_b +
                                 static_cast<int64_t>(workspace_j) *
                                     kv_stride_s +
                                 static_cast<int64_t>(d) * kv_stride_d);
            }
        }
        store_out_value<out_t>(
            out, static_cast<int64_t>(b) * out_stride_b +
                     static_cast<int64_t>(0) * out_stride_s +
                     static_cast<int64_t>(h) * out_stride_h +
                     static_cast<int64_t>(d) * out_stride_d,
            accum * sink_gate);
    }

    if (threadIdx.x == 0)
        lse[static_cast<int64_t>(b) * num_heads + h] = row_lse;
}

template <typename q_t, typename out_t, typename block_t, typename seq_t,
          typename req_t>
__global__ void sparse_mla_decode_full_context_kernel(
    const q_t* __restrict__ q, const uint8_t* __restrict__ k_cache,
    const block_t* __restrict__ block_table,
    const seq_t* __restrict__ seq_lens,
    const req_t* __restrict__ req_id_per_token,
    const float* __restrict__ attn_sink, out_t* __restrict__ out,
    float* __restrict__ lse, int batch_size, int num_heads, int block_size,
    int64_t cache_blocks, int block_table_rows, int block_table_width,
    int candidate_slots, int64_t q_stride_b, int64_t q_stride_s,
    int64_t q_stride_h, int64_t q_stride_d, int64_t out_stride_b,
    int64_t out_stride_s, int64_t out_stride_h, int64_t out_stride_d,
    int64_t block_stride_b, int64_t block_stride_s, int64_t seq_lens_stride,
    int64_t req_stride, int64_t cache_stride0_bytes, float softmax_scale) {
    extern __shared__ unsigned char smem[];
    unsigned char* cursor = smem;
    float* q_s = reinterpret_cast<float*>(cursor);
    cursor += static_cast<size_t>(kHeadDim) * sizeof(float);
    float* scores = reinterpret_cast<float*>(cursor);
    cursor += static_cast<size_t>(candidate_slots) * sizeof(float);
    int64_t* linear_s = align_shared_ptr<int64_t>(cursor);
    cursor += static_cast<size_t>(candidate_slots) * sizeof(int64_t);
    float* scale_s = align_shared_ptr<float>(cursor);

    const int bh = blockIdx.x;
    const int b = bh / num_heads;
    const int h = bh - b * num_heads;
    if (b >= batch_size)
        return;

    for (int d = threadIdx.x; d < kHeadDim; d += blockDim.x) {
        q_s[d] = load_q_value<q_t>(
            q, static_cast<int64_t>(b) * q_stride_b +
                   static_cast<int64_t>(h) * q_stride_h +
                   static_cast<int64_t>(d) * q_stride_d);
    }

    const int req = static_cast<int>(
        req_id_per_token[static_cast<int64_t>(b) * req_stride]);
    const bool req_valid = req >= 0 && req < block_table_rows;
    const int unclamped_seq_len =
        req_valid ? static_cast<int>(
                        seq_lens[static_cast<int64_t>(req) * seq_lens_stride])
                  : 0;
    const int seq_len = max(0, min(candidate_slots, unclamped_seq_len));
    const int64_t max_linear = cache_blocks * static_cast<int64_t>(block_size);

    for (int j = threadIdx.x; j < candidate_slots; j += blockDim.x) {
        int64_t linear = -1;
        if (j < seq_len) {
            const int block_in_seq = j / block_size;
            const int block_offset = j - block_in_seq * block_size;
            if (block_in_seq >= 0 && block_in_seq < block_table_width) {
                const int64_t physical_block = static_cast<int64_t>(
                    block_table[static_cast<int64_t>(req) * block_stride_b +
                                static_cast<int64_t>(block_in_seq) *
                                    block_stride_s]);
                const int64_t candidate_linear =
                    physical_block * static_cast<int64_t>(block_size) +
                    block_offset;
                if (physical_block >= 0 && candidate_linear >= 0 &&
                    candidate_linear < max_linear) {
                    linear = candidate_linear;
                }
            }
        }
        linear_s[j] = linear;

        float* cand_scales = scale_s + static_cast<int64_t>(j) * kNumQuantBlocks;
        #pragma unroll
        for (int s = 0; s < kNumQuantBlocks; ++s)
            cand_scales[s] = 0.0f;
        if (linear >= 0) {
            const uint8_t* scales = scale_ptr_from_linear(
                k_cache, cache_stride0_bytes, block_size, linear);
            #pragma unroll
            for (int s = 0; s < kNumQuantBlocks; ++s)
                cand_scales[s] = decode_ue8m0_scale(scales[s]);
        }
    }
    __syncthreads();

    const int group_id = threadIdx.x / kGroupedScoreGroupSize;
    const int lane = threadIdx.x - group_id * kGroupedScoreGroupSize;
    for (int base = 0; base < candidate_slots; base += kGroupedScoreGroups) {
        const int j = base + group_id;
        float partial = 0.0f;
        bool valid = false;
        if (j < candidate_slots && linear_s[j] >= 0) {
            valid = true;
            const uint8_t* token = token_ptr_from_linear(
                k_cache, cache_stride0_bytes, block_size, linear_s[j]);
            const float* cand_scales =
                scale_s + static_cast<int64_t>(j) * kNumQuantBlocks;
            for (int d = lane; d < kHeadDim; d += kGroupedScoreGroupSize) {
                partial += q_s[d] *
                           load_cache_value_from_token(token, cand_scales, d);
            }
        }

        for (int offset = kGroupedScoreGroupSize / 2; offset > 0; offset >>= 1)
            partial += __shfl_down_sync(0xffffffffu, partial, offset,
                                        kGroupedScoreGroupSize);
        if (lane == 0 && j < candidate_slots)
            scores[j] = valid ? partial * softmax_scale : -INFINITY;
    }
    __syncthreads();

    __shared__ float reduce_buf[kThreads];
    float local_max = -INFINITY;
    for (int j = threadIdx.x; j < candidate_slots; j += blockDim.x)
        local_max = fmaxf(local_max, scores[j]);
    reduce_buf[threadIdx.x] = local_max;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset)
            reduce_buf[threadIdx.x] =
                fmaxf(reduce_buf[threadIdx.x], reduce_buf[threadIdx.x + offset]);
        __syncthreads();
    }
    const float row_max = reduce_buf[0];

    float local_sum = 0.0f;
    for (int j = threadIdx.x; j < candidate_slots; j += blockDim.x) {
        const float p = isfinite(row_max) ? expf(scores[j] - row_max) : 0.0f;
        scores[j] = p;
        local_sum += p;
    }
    reduce_buf[threadIdx.x] = local_sum;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset)
            reduce_buf[threadIdx.x] += reduce_buf[threadIdx.x + offset];
        __syncthreads();
    }
    const float row_sum = reduce_buf[0];
    const float row_lse = row_sum > 0.0f ? logf(row_sum) + row_max : -INFINITY;
    for (int j = threadIdx.x; j < candidate_slots; j += blockDim.x)
        scores[j] = row_sum > 0.0f ? scores[j] / row_sum : 0.0f;
    __syncthreads();

    const float sink = attn_sink == nullptr ? 0.0f : attn_sink[h];
    const float sink_gate =
        attn_sink == nullptr ? 1.0f : 1.0f / (1.0f + expf(-(row_lse - sink)));

    for (int d = threadIdx.x; d < kHeadDim; d += blockDim.x) {
        float accum = 0.0f;
        if (row_sum > 0.0f) {
            for (int j = 0; j < candidate_slots; ++j) {
                const float p = scores[j];
                if (p == 0.0f || linear_s[j] < 0)
                    continue;
                const uint8_t* token = token_ptr_from_linear(
                    k_cache, cache_stride0_bytes, block_size, linear_s[j]);
                const float* cand_scales =
                    scale_s + static_cast<int64_t>(j) * kNumQuantBlocks;
                accum += p * load_cache_value_from_token(token, cand_scales, d);
            }
        }
        store_out_value<out_t>(
            out, static_cast<int64_t>(b) * out_stride_b +
                     static_cast<int64_t>(h) * out_stride_h +
                     static_cast<int64_t>(d) * out_stride_d,
            accum * sink_gate);
    }

    if (threadIdx.x == 0)
        lse[static_cast<int64_t>(b) * num_heads + h] = row_lse;
}

template <typename q_t, typename index_t>
__global__ void sparse_mla_score_kernel(
    const q_t* __restrict__ q, const uint8_t* __restrict__ k_cache,
    const void* __restrict__ indices, const int64_t* __restrict__ topk_length,
    float* __restrict__ scores, int batch_size, int num_heads, int block_size,
    int64_t cache_blocks, int topk, int64_t q_stride_b, int64_t q_stride_h,
    int64_t q_stride_d, int64_t index_stride_b, int64_t index_stride_k,
    int64_t cache_stride0_bytes, float softmax_scale) {
    __shared__ float reductions[kThreads];
    const int j = blockIdx.x;
    const int h = blockIdx.y;
    const int b = blockIdx.z;
    if (b >= batch_size || h >= num_heads || j >= topk)
        return;

    const int limit =
        topk_length == nullptr ? topk
                               : max(0, min(topk, static_cast<int>(topk_length[b])));
    const int64_t score_offset =
        (static_cast<int64_t>(b) * num_heads + h) * topk + j;
    if (j >= limit) {
        if (threadIdx.x == 0)
            scores[score_offset] = -INFINITY;
        return;
    }

    const int64_t linear = load_index<index_t>(
        indices, static_cast<int64_t>(b) * index_stride_b +
                     static_cast<int64_t>(j) * index_stride_k);
    if (linear < 0 || linear >= cache_blocks * block_size) {
        if (threadIdx.x == 0)
            scores[score_offset] = -INFINITY;
        return;
    }

    float partial = 0.0f;
    for (int d = threadIdx.x; d < kHeadDim; d += blockDim.x) {
        const float qv = load_q_value<q_t>(
            q, static_cast<int64_t>(b) * q_stride_b +
                   static_cast<int64_t>(h) * q_stride_h +
                   static_cast<int64_t>(d) * q_stride_d);
        partial += qv * load_cache_value(k_cache, cache_stride0_bytes,
                                         block_size, linear, d);
    }
    reductions[threadIdx.x] = partial;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset)
            reductions[threadIdx.x] += reductions[threadIdx.x + offset];
        __syncthreads();
    }
    if (threadIdx.x == 0)
        scores[score_offset] = reductions[0] * softmax_scale;
}

template <typename q_t, typename index_t>
__global__ void sparse_mla_score_tiled_kernel(
    const q_t* __restrict__ q, const uint8_t* __restrict__ k_cache,
    const void* __restrict__ indices, const int64_t* __restrict__ topk_length,
    float* __restrict__ scores, int batch_size, int num_heads, int block_size,
    int64_t cache_blocks, int topk, int64_t q_stride_b, int64_t q_stride_h,
    int64_t q_stride_d, int64_t index_stride_b, int64_t index_stride_k,
    int64_t cache_stride0_bytes, float softmax_scale) {
    const int group_id = threadIdx.x / kScoreGroupSize;
    const int lane = threadIdx.x - group_id * kScoreGroupSize;
    const int j = blockIdx.x * kScoreCandidatesPerBlock + group_id;
    const int h = blockIdx.y;
    const int b = blockIdx.z;
    if (b >= batch_size || h >= num_heads || j >= topk)
        return;

    const int limit =
        topk_length == nullptr ? topk
                               : max(0, min(topk, static_cast<int>(topk_length[b])));
    const int64_t score_offset =
        (static_cast<int64_t>(b) * num_heads + h) * topk + j;
    if (j >= limit) {
        if (lane == 0)
            scores[score_offset] = -INFINITY;
        return;
    }

    const int64_t linear = load_index<index_t>(
        indices, static_cast<int64_t>(b) * index_stride_b +
                     static_cast<int64_t>(j) * index_stride_k);
    if (linear < 0 || linear >= cache_blocks * block_size) {
        if (lane == 0)
            scores[score_offset] = -INFINITY;
        return;
    }

    float partial = 0.0f;
    for (int d = lane; d < kHeadDim; d += kScoreGroupSize) {
        const float qv = load_q_value<q_t>(
            q, static_cast<int64_t>(b) * q_stride_b +
                   static_cast<int64_t>(h) * q_stride_h +
                   static_cast<int64_t>(d) * q_stride_d);
        partial += qv * load_cache_value(k_cache, cache_stride0_bytes,
                                         block_size, linear, d);
    }

    unsigned mask = 0xffffffffu;
    for (int offset = kScoreGroupSize / 2; offset > 0; offset >>= 1)
        partial += __shfl_down_sync(mask, partial, offset, kScoreGroupSize);

    if (lane == 0)
        scores[score_offset] = partial * softmax_scale;
}

template <typename q_t, typename index_t>
__global__ void sparse_mla_score_tiled_scaled_kernel(
    const q_t* __restrict__ q, const uint8_t* __restrict__ k_cache,
    const void* __restrict__ indices, const int64_t* __restrict__ topk_length,
    float* __restrict__ scores, int batch_size, int num_heads, int block_size,
    int64_t cache_blocks, int topk, int64_t q_stride_b, int64_t q_stride_h,
    int64_t q_stride_d, int64_t index_stride_b, int64_t index_stride_k,
    int64_t cache_stride0_bytes, float softmax_scale) {
    const int group_id = threadIdx.x / kScoreGroupSize;
    const int lane = threadIdx.x - group_id * kScoreGroupSize;
    const int j = blockIdx.x * kScoreCandidatesPerBlock + group_id;
    const int h = blockIdx.y;
    const int b = blockIdx.z;
    if (b >= batch_size || h >= num_heads || j >= topk)
        return;

    const int limit =
        topk_length == nullptr ? topk
                               : max(0, min(topk, static_cast<int>(topk_length[b])));
    const int64_t score_offset =
        (static_cast<int64_t>(b) * num_heads + h) * topk + j;
    if (j >= limit) {
        if (lane == 0)
            scores[score_offset] = -INFINITY;
        return;
    }

    const int64_t linear = load_index<index_t>(
        indices, static_cast<int64_t>(b) * index_stride_b +
                     static_cast<int64_t>(j) * index_stride_k);
    if (linear < 0 || linear >= cache_blocks * block_size) {
        if (lane == 0)
            scores[score_offset] = -INFINITY;
        return;
    }

    const uint8_t* token = token_ptr_from_linear(
        k_cache, cache_stride0_bytes, block_size, linear);
    const uint8_t* scale_bytes = scale_ptr_from_linear(
        k_cache, cache_stride0_bytes, block_size, linear);
    float scales[kNumQuantBlocks];
    #pragma unroll
    for (int s = 0; s < kNumQuantBlocks; ++s)
        scales[s] = decode_ue8m0_scale(scale_bytes[s]);

    float partial = 0.0f;
    for (int d = lane; d < kHeadDim; d += kScoreGroupSize) {
        const float qv = load_q_value<q_t>(
            q, static_cast<int64_t>(b) * q_stride_b +
                   static_cast<int64_t>(h) * q_stride_h +
                   static_cast<int64_t>(d) * q_stride_d);
        partial += qv * load_cache_value_from_token(token, scales, d);
    }

    unsigned mask = 0xffffffffu;
    for (int offset = kScoreGroupSize / 2; offset > 0; offset >>= 1)
        partial += __shfl_down_sync(mask, partial, offset, kScoreGroupSize);

    if (lane == 0)
        scores[score_offset] = partial * softmax_scale;
}

__global__ void sparse_mla_softmax_kernel(
    float* __restrict__ scores, float* __restrict__ lse,
    const float* __restrict__ attn_sink, int batch_size, int num_heads,
    int active_heads, int topk) {
    __shared__ float reduce_buf[kThreads];
    const int bh = blockIdx.x;
    const int b = bh / active_heads;
    const int h = bh - b * active_heads;
    if (b >= batch_size)
        return;

    float local_max = -INFINITY;
    const int64_t base = static_cast<int64_t>(bh) * topk;
    for (int j = threadIdx.x; j < topk; j += blockDim.x)
        local_max = fmaxf(local_max, scores[base + j]);
    reduce_buf[threadIdx.x] = local_max;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset)
            reduce_buf[threadIdx.x] =
                fmaxf(reduce_buf[threadIdx.x], reduce_buf[threadIdx.x + offset]);
        __syncthreads();
    }
    const float row_max = reduce_buf[0];

    float local_sum = 0.0f;
    for (int j = threadIdx.x; j < topk; j += blockDim.x) {
        const float p = isfinite(row_max) ? expf(scores[base + j] - row_max) : 0.0f;
        scores[base + j] = p;
        local_sum += p;
    }
    reduce_buf[threadIdx.x] = local_sum;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset)
            reduce_buf[threadIdx.x] += reduce_buf[threadIdx.x + offset];
        __syncthreads();
    }
    const float row_sum = reduce_buf[0];
    const float row_lse = row_sum > 0.0f ? logf(row_sum) + row_max : -INFINITY;
    for (int j = threadIdx.x; j < topk; j += blockDim.x)
        scores[base + j] = row_sum > 0.0f ? scores[base + j] / row_sum : 0.0f;
    if (threadIdx.x == 0)
        lse[static_cast<int64_t>(b) * num_heads + h] = row_lse;
}

template <typename out_t, typename index_t>
__global__ void sparse_mla_output_kernel(
    const uint8_t* __restrict__ k_cache, const void* __restrict__ indices,
    const int64_t* __restrict__ topk_length, const float* __restrict__ scores,
    const float* __restrict__ lse, const float* __restrict__ attn_sink,
    out_t* __restrict__ out, int batch_size, int active_heads, int num_heads,
    int block_size, int64_t cache_blocks, int topk, int64_t out_stride_b,
    int64_t out_stride_h, int64_t out_stride_d, int64_t index_stride_b,
    int64_t index_stride_k, int64_t cache_stride0_bytes) {
    const int bh = blockIdx.x;
    const int tile = blockIdx.y;
    const int b = bh / active_heads;
    const int h = bh - b * active_heads;
    const int d = tile * blockDim.x + threadIdx.x;
    if (b >= batch_size || d >= kHeadDim)
        return;

    const int limit =
        topk_length == nullptr ? topk
                               : max(0, min(topk, static_cast<int>(topk_length[b])));
    float accum = 0.0f;
    const int64_t score_base = static_cast<int64_t>(bh) * topk;
    for (int j = 0; j < limit; ++j) {
        const float p = scores[score_base + j];
        if (p == 0.0f)
            continue;
        const int64_t linear = load_index<index_t>(
            indices, static_cast<int64_t>(b) * index_stride_b +
                         static_cast<int64_t>(j) * index_stride_k);
        if (linear < 0 || linear >= cache_blocks * block_size)
            continue;
        accum += p * load_cache_value(k_cache, cache_stride0_bytes, block_size,
                                      linear, d);
    }

    const float row_lse = lse[static_cast<int64_t>(b) * num_heads + h];
    const float sink = attn_sink == nullptr ? 0.0f : attn_sink[h];
    const float gate =
        attn_sink == nullptr ? 1.0f : 1.0f / (1.0f + expf(-(row_lse - sink)));
    store_out_value<out_t>(
        out, static_cast<int64_t>(b) * out_stride_b +
                 static_cast<int64_t>(h) * out_stride_h +
                 static_cast<int64_t>(d) * out_stride_d,
        accum * gate);
}

template <typename out_t, typename index_t>
__global__ void sparse_mla_output_scaled_kernel(
    const uint8_t* __restrict__ k_cache, const void* __restrict__ indices,
    const int64_t* __restrict__ topk_length, const float* __restrict__ scores,
    const float* __restrict__ lse, const float* __restrict__ attn_sink,
    out_t* __restrict__ out, int batch_size, int active_heads, int num_heads,
    int block_size, int64_t cache_blocks, int topk, int64_t out_stride_b,
    int64_t out_stride_h, int64_t out_stride_d, int64_t index_stride_b,
    int64_t index_stride_k, int64_t cache_stride0_bytes) {
    extern __shared__ float scale_s[];
    const int bh = blockIdx.x;
    const int tile = blockIdx.y;
    const int b = bh / active_heads;
    const int h = bh - b * active_heads;
    const int lane = threadIdx.x;
    const int d = tile * kQuantBlock + lane;
    if (b >= batch_size)
        return;

    const int limit =
        topk_length == nullptr ? topk
                               : max(0, min(topk, static_cast<int>(topk_length[b])));
    const bool fp8_tile = tile < kNumQuantBlocks;
    for (int j = threadIdx.x; j < limit; j += blockDim.x) {
        float scale = 1.0f;
        if (fp8_tile) {
            const int64_t linear = load_index<index_t>(
                indices, static_cast<int64_t>(b) * index_stride_b +
                             static_cast<int64_t>(j) * index_stride_k);
            if (linear >= 0 && linear < cache_blocks * block_size) {
                const uint8_t* scales = scale_ptr_from_linear(
                    k_cache, cache_stride0_bytes, block_size, linear);
                scale = decode_ue8m0_scale(scales[tile]);
            } else {
                scale = 0.0f;
            }
        }
        scale_s[j] = scale;
    }
    __syncthreads();

    if (lane >= kQuantBlock || d >= kHeadDim)
        return;

    float accum = 0.0f;
    const int64_t score_base = static_cast<int64_t>(bh) * topk;
    for (int j = 0; j < limit; ++j) {
        const float p = scores[score_base + j];
        if (p == 0.0f)
            continue;
        const int64_t linear = load_index<index_t>(
            indices, static_cast<int64_t>(b) * index_stride_b +
                         static_cast<int64_t>(j) * index_stride_k);
        if (linear < 0 || linear >= cache_blocks * block_size)
            continue;
        const uint8_t* token = token_ptr_from_linear(
            k_cache, cache_stride0_bytes, block_size, linear);
        float value;
        if (d < kFp8Dim) {
            value = fp8_e4m3fn_to_float(token[d]) * scale_s[j];
        } else {
            const auto* rope =
                reinterpret_cast<const __nv_bfloat16*>(token + kFp8Dim);
            value = __bfloat162float(rope[d - kFp8Dim]);
        }
        accum += p * value;
    }

    const float row_lse = lse[static_cast<int64_t>(b) * num_heads + h];
    const float sink = attn_sink == nullptr ? 0.0f : attn_sink[h];
    const float gate =
        attn_sink == nullptr ? 1.0f : 1.0f / (1.0f + expf(-(row_lse - sink)));
    store_out_value<out_t>(
        out, static_cast<int64_t>(b) * out_stride_b +
                 static_cast<int64_t>(h) * out_stride_h +
                 static_cast<int64_t>(d) * out_stride_d,
        accum * gate);
}

template <typename q_t, typename kv_t>
__global__ void sparse_mla_workspace_score_tiled_kernel(
    const q_t* __restrict__ q, const kv_t* __restrict__ kv_workspace,
    const void* __restrict__ topk_length,
    const void* __restrict__ extra_topk_length, float* __restrict__ scores,
    int batch_size, int active_heads, int num_heads, int main_topk,
    int extra_topk, int candidate_slots, int64_t q_stride_b,
    int64_t q_stride_h, int64_t q_stride_d, int64_t kv_stride_b,
    int64_t kv_stride_s, int64_t kv_stride_d, int topk_length_kind,
    int extra_topk_length_kind, float softmax_scale) {
    const int group_id = threadIdx.x / kScoreGroupSize;
    const int lane = threadIdx.x - group_id * kScoreGroupSize;
    const int j = blockIdx.x * kScoreCandidatesPerBlock + group_id;
    const int h = blockIdx.y;
    const int b = blockIdx.z;
    if (b >= batch_size || h >= active_heads || j >= candidate_slots)
        return;

    const int main_limit = max(
        0, min(main_topk,
               load_length_value(topk_length, topk_length_kind, b, main_topk)));
    const int extra_limit =
        max(0, min(extra_topk,
                   load_length_value(extra_topk_length, extra_topk_length_kind,
                                     b, extra_topk)));
    const bool valid =
        j < main_limit || (j >= main_topk && j < main_topk + extra_limit);
    const int64_t score_offset =
        (static_cast<int64_t>(b) * active_heads + h) * candidate_slots + j;
    if (!valid) {
        if (lane == 0)
            scores[score_offset] = -INFINITY;
        return;
    }

    float partial = 0.0f;
    for (int d = lane; d < kHeadDim; d += kScoreGroupSize) {
        const float qv = load_q_value<q_t>(
            q, static_cast<int64_t>(b) * q_stride_b +
                   static_cast<int64_t>(h) * q_stride_h +
                   static_cast<int64_t>(d) * q_stride_d);
        partial += qv *
                   load_workspace_value<kv_t>(
                       kv_workspace,
                       static_cast<int64_t>(b) * kv_stride_b +
                           static_cast<int64_t>(j) * kv_stride_s +
                           static_cast<int64_t>(d) * kv_stride_d);
    }

    for (int offset = kScoreGroupSize / 2; offset > 0; offset >>= 1)
        partial += __shfl_down_sync(0xffffffffu, partial, offset,
                                    kScoreGroupSize);
    if (lane == 0)
        scores[score_offset] = partial * softmax_scale;
}

template <typename kv_t, typename out_t>
__global__ void sparse_mla_workspace_output_kernel(
    const kv_t* __restrict__ kv_workspace, const void* __restrict__ topk_length,
    const void* __restrict__ extra_topk_length,
    const float* __restrict__ scores, const float* __restrict__ lse,
    const float* __restrict__ attn_sink, out_t* __restrict__ out,
    int batch_size, int active_heads, int num_heads, int main_topk,
    int extra_topk, int candidate_slots, int64_t kv_stride_b,
    int64_t kv_stride_s, int64_t kv_stride_d, int64_t out_stride_b,
    int64_t out_stride_h, int64_t out_stride_d, int topk_length_kind,
    int extra_topk_length_kind) {
    const int bh = blockIdx.x;
    const int tile = blockIdx.y;
    const int b = bh / active_heads;
    const int h = bh - b * active_heads;
    const int d = tile * blockDim.x + threadIdx.x;
    if (b >= batch_size || d >= kHeadDim)
        return;

    const int main_limit = max(
        0, min(main_topk,
               load_length_value(topk_length, topk_length_kind, b, main_topk)));
    const int extra_limit =
        max(0, min(extra_topk,
                   load_length_value(extra_topk_length, extra_topk_length_kind,
                                     b, extra_topk)));
    const int64_t score_base =
        (static_cast<int64_t>(b) * active_heads + h) * candidate_slots;

    float accum = 0.0f;
    for (int j = 0; j < main_limit; ++j) {
        accum += scores[score_base + j] *
                 load_workspace_value<kv_t>(
                     kv_workspace,
                     static_cast<int64_t>(b) * kv_stride_b +
                         static_cast<int64_t>(j) * kv_stride_s +
                         static_cast<int64_t>(d) * kv_stride_d);
    }
    for (int j = 0; j < extra_limit; ++j) {
        const int workspace_j = main_topk + j;
        accum += scores[score_base + workspace_j] *
                 load_workspace_value<kv_t>(
                     kv_workspace,
                     static_cast<int64_t>(b) * kv_stride_b +
                         static_cast<int64_t>(workspace_j) * kv_stride_s +
                         static_cast<int64_t>(d) * kv_stride_d);
    }

    const float row_lse = lse[static_cast<int64_t>(b) * num_heads + h];
    const float sink = attn_sink == nullptr ? 0.0f : attn_sink[h];
    const float gate =
        attn_sink == nullptr ? 1.0f : 1.0f / (1.0f + expf(-(row_lse - sink)));
    store_out_value<out_t>(
        out, static_cast<int64_t>(b) * out_stride_b +
                 static_cast<int64_t>(h) * out_stride_h +
                 static_cast<int64_t>(d) * out_stride_d,
        accum * gate);
}

template <typename q_t, typename kv_t, typename out_t, typename index_t>
__global__ void sparse_mla_prefill_workspace_kernel(
    const q_t* __restrict__ q, const kv_t* __restrict__ kv,
    const index_t* __restrict__ indices, const void* __restrict__ topk_length,
    int topk_length_kind, const float* __restrict__ attn_sink,
    out_t* __restrict__ out, float* __restrict__ max_logits,
    float* __restrict__ lse, int num_heads, int kv_tokens, int topk,
    int64_t q_stride_t, int64_t q_stride_h, int64_t q_stride_d,
    int64_t kv_stride_t, int64_t kv_stride_d, int64_t indices_stride_t,
    int64_t indices_stride_k, int64_t out_stride_t, int64_t out_stride_h,
    int64_t out_stride_d, double softmax_scale) {
    extern __shared__ unsigned char smem[];
    float* scores = reinterpret_cast<float*>(smem);
    float* reduce = scores + topk;

    const int token = blockIdx.x;
    const int head = blockIdx.y;
    const int tid = threadIdx.x;

    int limit = load_length_value(topk_length, topk_length_kind, token, topk);
    limit = max(0, min(limit, topk));

    float local_max = -INFINITY;
    for (int k_idx = tid; k_idx < limit; k_idx += blockDim.x) {
        const int64_t selected = load_index<index_t>(
            indices, static_cast<int64_t>(token) * indices_stride_t +
                         static_cast<int64_t>(k_idx) * indices_stride_k);
        float score = -INFINITY;
        if (selected >= 0 && selected < kv_tokens) {
            float dot = 0.0f;
#pragma unroll 4
            for (int d = 0; d < kHeadDim; ++d) {
                const float qv = load_q_value<q_t>(
                    q, static_cast<int64_t>(token) * q_stride_t +
                           static_cast<int64_t>(head) * q_stride_h +
                           static_cast<int64_t>(d) * q_stride_d);
                const float kvv = load_workspace_value<kv_t>(
                    kv, selected * kv_stride_t +
                            static_cast<int64_t>(d) * kv_stride_d);
                dot += qv * kvv;
            }
            score = dot * static_cast<float>(softmax_scale);
        }
        scores[k_idx] = score;
        local_max = fmaxf(local_max, score);
    }

    reduce[tid] = local_max;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride)
            reduce[tid] = fmaxf(reduce[tid], reduce[tid + stride]);
        __syncthreads();
    }
    const float row_max = reduce[0];

    float local_sum = 0.0f;
    if (isfinite(row_max)) {
        for (int k_idx = tid; k_idx < limit; k_idx += blockDim.x) {
            const float p = isfinite(scores[k_idx])
                                ? expf(scores[k_idx] - row_max)
                                : 0.0f;
            scores[k_idx] = p;
            local_sum += p;
        }
    }
    reduce[tid] = local_sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride)
            reduce[tid] += reduce[tid + stride];
        __syncthreads();
    }

    const float denom = reduce[0];
    const float row_lse = denom > 0.0f ? logf(denom) + row_max : -INFINITY;
    const float inv_denom = denom > 0.0f ? 1.0f / denom : 0.0f;
    const float sink = attn_sink == nullptr ? 0.0f : attn_sink[head];
    const float gate =
        attn_sink == nullptr ? 1.0f : 1.0f / (1.0f + expf(-(row_lse - sink)));

    for (int d = tid; d < kHeadDim; d += blockDim.x) {
        float accum = 0.0f;
        if (denom > 0.0f) {
            for (int k_idx = 0; k_idx < limit; ++k_idx) {
                const float p = scores[k_idx] * inv_denom;
                if (p == 0.0f)
                    continue;
                const int64_t selected = load_index<index_t>(
                    indices, static_cast<int64_t>(token) * indices_stride_t +
                                 static_cast<int64_t>(k_idx) *
                                     indices_stride_k);
                if (selected >= 0 && selected < kv_tokens) {
                    accum += p * load_workspace_value<kv_t>(
                                     kv, selected * kv_stride_t +
                                             static_cast<int64_t>(d) *
                                                 kv_stride_d);
                }
            }
        }
        store_out_value<out_t>(
            out, static_cast<int64_t>(token) * out_stride_t +
                     static_cast<int64_t>(head) * out_stride_h +
                     static_cast<int64_t>(d) * out_stride_d,
            accum * gate);
    }

    if (tid == 0) {
        max_logits[static_cast<int64_t>(token) * num_heads + head] = row_max;
        lse[static_cast<int64_t>(token) * num_heads + head] = row_lse;
    }
}

bool is_none(const pybind11::object& obj) {
    return obj.is_none();
}

torch::Tensor tensor_or_empty(const pybind11::object& obj) {
    return is_none(obj) ? torch::Tensor() : obj.cast<torch::Tensor>();
}

int64_t byte_stride(const torch::Tensor& tensor, int dim) {
    return tensor.stride(dim) * tensor.element_size();
}

int length_tensor_kind(const torch::Tensor& tensor) {
    if (!tensor.defined())
        return 1;
    if (tensor.scalar_type() == torch::kInt)
        return 0;
    DG_HOST_ASSERT(tensor.scalar_type() == torch::kInt64);
    return 1;
}

template <typename out_t, typename seq_t, typename block_t, typename gather_t>
void launch_dequantize_gather_k_cache(
    const torch::Tensor& out, const torch::Tensor& k_cache,
    const torch::Tensor& seq_lens, const torch::Tensor& gather_lens,
    const torch::Tensor& block_table, int block_size, int offset) {
    const int num_reqs = static_cast<int>(seq_lens.size(0));
    const int max_out_rows = static_cast<int>(out.size(1)) - offset;
    const int head_dim = min(static_cast<int>(out.size(2)), kHeadDim);
    if (num_reqs <= 0 || max_out_rows <= 0 || head_dim <= 0)
        return;

    const auto stream = at::cuda::getCurrentCUDAStream();
    const dim3 block(kThreads);
    const dim3 grid((head_dim + kThreads - 1) / kThreads, max_out_rows,
                    num_reqs);
    dequantize_gather_k_cache_kernel<out_t, seq_t, block_t, gather_t>
        <<<grid, block, 0, stream>>>(
            reinterpret_cast<out_t*>(out.data_ptr()),
            reinterpret_cast<const uint8_t*>(k_cache.data_ptr()),
            seq_lens.data_ptr<seq_t>(),
            gather_lens.defined() ? gather_lens.data_ptr<gather_t>() : nullptr,
            block_table.data_ptr<block_t>(), num_reqs, max_out_rows, head_dim,
            block_size, offset, k_cache.size(0), byte_stride(k_cache, 0),
            out.stride(0), out.stride(1), out.stride(2), block_table.stride(0),
            block_table.stride(1));
    DG_CUDA_RUNTIME_CHECK(cudaGetLastError());
}

template <typename out_t, typename index_t, typename len_t>
void launch_dequantize_gather_indexed_k_cache(
    const torch::Tensor& out, const torch::Tensor& k_cache,
    const torch::Tensor& indices, const torch::Tensor& topk_length,
    int block_size, int offset) {
    const bool indices_3d = indices.dim() == 3;
    const int batch_size = static_cast<int>(indices.size(0));
    const int topk = static_cast<int>(indices.size(indices_3d ? 2 : 1));
    const int head_dim = min(static_cast<int>(out.size(2)), kHeadDim);
    if (batch_size <= 0 || topk <= 0 || head_dim <= 0)
        return;

    const auto stream = at::cuda::getCurrentCUDAStream();
    const dim3 block(kThreads);
    const dim3 grid((head_dim + kThreads - 1) / kThreads, topk, batch_size);
    dequantize_gather_indexed_k_cache_kernel<out_t, index_t, len_t>
        <<<grid, block, 0, stream>>>(
            reinterpret_cast<out_t*>(out.data_ptr()),
            reinterpret_cast<const uint8_t*>(k_cache.data_ptr()),
            indices.data_ptr(),
            topk_length.defined() ? topk_length.data_ptr<len_t>() : nullptr,
            batch_size, topk, head_dim, block_size, offset, k_cache.size(0),
            byte_stride(k_cache, 0), out.stride(0), out.stride(1),
            out.stride(2), indices.stride(0),
            indices_3d ? indices.stride(1) : 0,
            indices.stride(indices_3d ? 2 : 1), indices_3d);
    DG_CUDA_RUNTIME_CHECK(cudaGetLastError());
}

template <typename q_t, typename out_t, typename index_t>
void launch_sparse_mla_decode(
    const torch::Tensor& q, const torch::Tensor& k_cache,
    const torch::Tensor& indices, const torch::Tensor& topk_length,
    const torch::Tensor& attn_sink, const torch::Tensor& extra_k_cache,
    const torch::Tensor& extra_indices, const torch::Tensor& extra_topk_length,
    const torch::Tensor& out, const torch::Tensor& lse, double softmax_scale) {
    const int batch_size = static_cast<int>(q.size(0));
    const int num_heads = static_cast<int>(q.size(2));
    const int active_heads = sm120_active_heads(num_heads);
    const int block_size = static_cast<int>(k_cache.size(1));
    const int main_topk = static_cast<int>(indices.size(2));
    const int extra_topk = extra_indices.defined() ? static_cast<int>(extra_indices.size(2)) : 0;
    const int candidate_count = main_topk + extra_topk;
    DG_HOST_ASSERT(candidate_count > 0);
    DG_HOST_ASSERT(candidate_count <= 8192);

    const auto stream = at::cuda::getCurrentCUDAStream();
    const int grid = batch_size * active_heads;

    if (candidate_count <= kMaxGroupedCandidateSlots) {
        const size_t grouped_shared_bytes =
            grouped_decode_shared_bytes(candidate_count);
        sparse_mla_decode_grouped_kernel<q_t, out_t, index_t>
            <<<grid, kThreads, grouped_shared_bytes, stream>>>(
                reinterpret_cast<const q_t*>(q.data_ptr()),
                reinterpret_cast<const uint8_t*>(k_cache.data_ptr()),
                indices.data_ptr(),
                topk_length.defined() ? topk_length.data_ptr<int64_t>() : nullptr,
                attn_sink.defined() ? attn_sink.data_ptr<float>() : nullptr,
                extra_k_cache.defined()
                    ? reinterpret_cast<const uint8_t*>(extra_k_cache.data_ptr())
                    : nullptr,
                extra_indices.defined() ? extra_indices.data_ptr() : nullptr,
                extra_topk_length.defined()
                    ? extra_topk_length.data_ptr<int64_t>()
                    : nullptr,
                reinterpret_cast<out_t*>(out.data_ptr()), lse.data_ptr<float>(),
                batch_size, active_heads, num_heads, block_size, k_cache.size(0),
                extra_k_cache.defined() ? extra_k_cache.size(0) : 0, main_topk,
                extra_topk, candidate_count, q.stride(0), q.stride(1),
                q.stride(2), q.stride(3), out.stride(0), out.stride(1),
                out.stride(2), out.stride(3), indices.stride(0),
                indices.stride(1), indices.stride(2),
                extra_indices.defined() ? extra_indices.stride(0) : 0,
                extra_indices.defined() ? extra_indices.stride(1) : 0,
                extra_indices.defined() ? extra_indices.stride(2) : 0,
                byte_stride(k_cache, 0),
                extra_k_cache.defined() ? byte_stride(extra_k_cache, 0) : 0,
                static_cast<float>(softmax_scale));
        DG_CUDA_RUNTIME_CHECK(cudaGetLastError());
        return;
    }

    const size_t shared_bytes = static_cast<size_t>(candidate_count) * sizeof(float);

    sparse_mla_decode_kernel<q_t, out_t, index_t><<<grid, kThreads, shared_bytes, stream>>>(
        reinterpret_cast<const q_t*>(q.data_ptr()), reinterpret_cast<const uint8_t*>(k_cache.data_ptr()),
        indices.data_ptr(), topk_length.defined() ? topk_length.data_ptr<int64_t>() : nullptr,
        attn_sink.defined() ? attn_sink.data_ptr<float>() : nullptr,
        extra_k_cache.defined() ? reinterpret_cast<const uint8_t*>(extra_k_cache.data_ptr()) : nullptr,
        extra_indices.defined() ? extra_indices.data_ptr() : nullptr,
        extra_topk_length.defined() ? extra_topk_length.data_ptr<int64_t>() : nullptr,
        reinterpret_cast<out_t*>(out.data_ptr()), lse.data_ptr<float>(), batch_size,
        active_heads, num_heads, block_size, k_cache.size(0),
        extra_k_cache.defined() ? extra_k_cache.size(0) : 0, main_topk, extra_topk,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3), out.stride(0),
        out.stride(1), out.stride(2), out.stride(3), indices.stride(0),
        indices.stride(1), indices.stride(2),
        extra_indices.defined() ? extra_indices.stride(0) : 0,
        extra_indices.defined() ? extra_indices.stride(1) : 0,
        extra_indices.defined() ? extra_indices.stride(2) : 0,
        byte_stride(k_cache, 0),
        extra_k_cache.defined() ? byte_stride(extra_k_cache, 0) : 0,
        static_cast<float>(softmax_scale));
    DG_CUDA_RUNTIME_CHECK(cudaGetLastError());
}

template <typename q_t, typename kv_t, typename out_t>
void launch_sparse_mla_decode_from_workspace(
    const torch::Tensor& q, const torch::Tensor& kv_workspace,
    const torch::Tensor& topk_length, const torch::Tensor& extra_topk_length,
    const torch::Tensor& attn_sink, const torch::Tensor& out,
    const torch::Tensor& lse, int main_topk, int extra_topk,
    double softmax_scale) {
    const int batch_size = static_cast<int>(q.size(0));
    const int num_heads = static_cast<int>(q.size(2));
    const int active_heads = sm120_active_heads(num_heads);
    const int candidate_slots = main_topk + extra_topk;
    DG_HOST_ASSERT(batch_size > 0);
    DG_HOST_ASSERT(num_heads > 0);
    DG_HOST_ASSERT(candidate_slots > 0);
    DG_HOST_ASSERT(candidate_slots <= kMaxGroupedCandidateSlots);
    DG_HOST_ASSERT(kv_workspace.size(0) >= batch_size);
    DG_HOST_ASSERT(kv_workspace.size(1) >= candidate_slots);
    DG_HOST_ASSERT(kv_workspace.size(2) >= kHeadDim);

    const auto stream = at::cuda::getCurrentCUDAStream();
    const int grid = batch_size * num_heads;
    const size_t shared_bytes =
        static_cast<size_t>(kHeadDim + candidate_slots) * sizeof(float);
    sparse_mla_decode_workspace_kernel<q_t, kv_t, out_t>
        <<<grid, kThreads, shared_bytes, stream>>>(
            reinterpret_cast<const q_t*>(q.data_ptr()),
            reinterpret_cast<const kv_t*>(kv_workspace.data_ptr()),
            topk_length.defined() ? topk_length.data_ptr() : nullptr,
            extra_topk_length.defined() ? extra_topk_length.data_ptr()
                                        : nullptr,
            attn_sink.defined() ? attn_sink.data_ptr<float>() : nullptr,
            reinterpret_cast<out_t*>(out.data_ptr()), lse.data_ptr<float>(),
            batch_size, active_heads, num_heads, main_topk, extra_topk,
            candidate_slots, q.stride(0), q.stride(1), q.stride(2),
            q.stride(3), kv_workspace.stride(0), kv_workspace.stride(1),
            kv_workspace.stride(2), out.stride(0), out.stride(1),
            out.stride(2), out.stride(3), length_tensor_kind(topk_length),
            length_tensor_kind(extra_topk_length),
            static_cast<float>(softmax_scale));
    DG_CUDA_RUNTIME_CHECK(cudaGetLastError());
}

template <typename q_t, typename kv_t, typename out_t>
void launch_sparse_mla_decode_from_workspace_split(
    const torch::Tensor& q, const torch::Tensor& kv_workspace,
    const torch::Tensor& topk_length, const torch::Tensor& extra_topk_length,
    const torch::Tensor& attn_sink, const torch::Tensor& out,
    const torch::Tensor& lse, int main_topk, int extra_topk,
    double softmax_scale) {
    const int batch_size = static_cast<int>(q.size(0));
    const int num_heads = static_cast<int>(q.size(2));
    const int active_heads = sm120_active_heads(num_heads);
    const int candidate_slots = main_topk + extra_topk;
    DG_HOST_ASSERT(batch_size > 0);
    DG_HOST_ASSERT(num_heads > 0);
    DG_HOST_ASSERT(candidate_slots > 0);
    DG_HOST_ASSERT(candidate_slots <= 4096);

    auto scores = torch::empty({batch_size, active_heads, candidate_slots},
                               q.options().dtype(torch::kFloat32));
    const auto stream = at::cuda::getCurrentCUDAStream();
    const dim3 score_grid(
        (candidate_slots + kScoreCandidatesPerBlock - 1) /
            kScoreCandidatesPerBlock,
        active_heads, batch_size);
    sparse_mla_workspace_score_tiled_kernel<q_t, kv_t>
        <<<score_grid, kThreads, 0, stream>>>(
            reinterpret_cast<const q_t*>(q.data_ptr()),
            reinterpret_cast<const kv_t*>(kv_workspace.data_ptr()),
            topk_length.defined() ? topk_length.data_ptr() : nullptr,
            extra_topk_length.defined() ? extra_topk_length.data_ptr()
                                        : nullptr,
            scores.data_ptr<float>(), batch_size, active_heads, num_heads,
            main_topk, extra_topk, candidate_slots, q.stride(0), q.stride(2),
            q.stride(3), kv_workspace.stride(0), kv_workspace.stride(1),
            kv_workspace.stride(2), length_tensor_kind(topk_length),
            length_tensor_kind(extra_topk_length),
            static_cast<float>(softmax_scale));
    DG_CUDA_RUNTIME_CHECK(cudaGetLastError());

    sparse_mla_softmax_kernel<<<batch_size * active_heads, kThreads, 0, stream>>>(
        scores.data_ptr<float>(), lse.data_ptr<float>(),
        attn_sink.defined() ? attn_sink.data_ptr<float>() : nullptr, batch_size,
        num_heads, active_heads, candidate_slots);
    DG_CUDA_RUNTIME_CHECK(cudaGetLastError());

    const dim3 out_grid(batch_size * active_heads,
                        (kHeadDim + kThreads - 1) / kThreads);
    sparse_mla_workspace_output_kernel<kv_t, out_t>
        <<<out_grid, kThreads, 0, stream>>>(
            reinterpret_cast<const kv_t*>(kv_workspace.data_ptr()),
            topk_length.defined() ? topk_length.data_ptr() : nullptr,
            extra_topk_length.defined() ? extra_topk_length.data_ptr()
                                        : nullptr,
            scores.data_ptr<float>(), lse.data_ptr<float>(),
            attn_sink.defined() ? attn_sink.data_ptr<float>() : nullptr,
            reinterpret_cast<out_t*>(out.data_ptr()), batch_size, active_heads,
            num_heads, main_topk, extra_topk, candidate_slots,
            kv_workspace.stride(0), kv_workspace.stride(1),
            kv_workspace.stride(2), out.stride(0), out.stride(2),
            out.stride(3), length_tensor_kind(topk_length),
            length_tensor_kind(extra_topk_length));
    DG_CUDA_RUNTIME_CHECK(cudaGetLastError());
}

template <typename q_t, typename out_t, typename index_t>
void launch_sparse_mla_decode_fast(
    const torch::Tensor& q, const torch::Tensor& k_cache,
    const torch::Tensor& indices, const torch::Tensor& topk_length,
    const torch::Tensor& attn_sink, const torch::Tensor& out,
    const torch::Tensor& lse, double softmax_scale) {
    const int batch_size = static_cast<int>(q.size(0));
    const int num_heads = static_cast<int>(q.size(2));
    const int active_heads = sm120_active_heads(num_heads);
    const int block_size = static_cast<int>(k_cache.size(1));
    const int topk = static_cast<int>(indices.size(2));
    DG_HOST_ASSERT(topk > 0);
    DG_HOST_ASSERT(topk <= 4096);

    auto scores = torch::empty({batch_size, num_heads, topk},
                               q.options().dtype(torch::kFloat32));
    const auto stream = at::cuda::getCurrentCUDAStream();
    const dim3 score_grid(
        (topk + kScoreCandidatesPerBlock - 1) / kScoreCandidatesPerBlock,
        active_heads, batch_size);
    sparse_mla_score_tiled_kernel<q_t, index_t><<<score_grid, kThreads, 0, stream>>>(
        reinterpret_cast<const q_t*>(q.data_ptr()),
        reinterpret_cast<const uint8_t*>(k_cache.data_ptr()), indices.data_ptr(),
        topk_length.defined() ? topk_length.data_ptr<int64_t>() : nullptr,
        scores.data_ptr<float>(), batch_size, num_heads, block_size,
        k_cache.size(0), topk, q.stride(0), q.stride(2), q.stride(3),
        indices.stride(0), indices.stride(2), byte_stride(k_cache, 0),
        static_cast<float>(softmax_scale));
    DG_CUDA_RUNTIME_CHECK(cudaGetLastError());

    sparse_mla_softmax_kernel<<<batch_size * active_heads, kThreads, 0, stream>>>(
        scores.data_ptr<float>(), lse.data_ptr<float>(),
        attn_sink.defined() ? attn_sink.data_ptr<float>() : nullptr, batch_size,
        num_heads, active_heads, topk);
    DG_CUDA_RUNTIME_CHECK(cudaGetLastError());

    const dim3 out_grid(batch_size * active_heads,
                        (kHeadDim + kThreads - 1) / kThreads);
    sparse_mla_output_kernel<out_t, index_t><<<out_grid, kThreads, 0, stream>>>(
        reinterpret_cast<const uint8_t*>(k_cache.data_ptr()), indices.data_ptr(),
        topk_length.defined() ? topk_length.data_ptr<int64_t>() : nullptr,
        scores.data_ptr<float>(), lse.data_ptr<float>(),
        attn_sink.defined() ? attn_sink.data_ptr<float>() : nullptr,
        reinterpret_cast<out_t*>(out.data_ptr()), batch_size, active_heads,
        num_heads, block_size, k_cache.size(0), topk, out.stride(0),
        out.stride(2), out.stride(3), indices.stride(0), indices.stride(2),
        byte_stride(k_cache, 0));
    DG_CUDA_RUNTIME_CHECK(cudaGetLastError());
}

template <typename q_t, typename out_t, typename index_t>
void launch_sparse_mla_decode_scaled(
    const torch::Tensor& q, const torch::Tensor& k_cache,
    const torch::Tensor& indices, const torch::Tensor& topk_length,
    const torch::Tensor& attn_sink, const torch::Tensor& out,
    const torch::Tensor& lse, double softmax_scale) {
    const int batch_size = static_cast<int>(q.size(0));
    const int num_heads = static_cast<int>(q.size(2));
    const int active_heads = sm120_active_heads(num_heads);
    const int block_size = static_cast<int>(k_cache.size(1));
    const int topk = static_cast<int>(indices.size(2));
    DG_HOST_ASSERT(topk > 0);
    DG_HOST_ASSERT(topk <= 4096);

    auto scores = torch::empty({batch_size, num_heads, topk},
                               q.options().dtype(torch::kFloat32));
    const auto stream = at::cuda::getCurrentCUDAStream();
    const dim3 score_grid(
        (topk + kScoreCandidatesPerBlock - 1) / kScoreCandidatesPerBlock,
        active_heads, batch_size);
    sparse_mla_score_tiled_scaled_kernel<q_t, index_t>
        <<<score_grid, kThreads, 0, stream>>>(
            reinterpret_cast<const q_t*>(q.data_ptr()),
            reinterpret_cast<const uint8_t*>(k_cache.data_ptr()),
            indices.data_ptr(),
            topk_length.defined() ? topk_length.data_ptr<int64_t>() : nullptr,
            scores.data_ptr<float>(), batch_size, num_heads, block_size,
            k_cache.size(0), topk, q.stride(0), q.stride(2), q.stride(3),
            indices.stride(0), indices.stride(2), byte_stride(k_cache, 0),
            static_cast<float>(softmax_scale));
    DG_CUDA_RUNTIME_CHECK(cudaGetLastError());

    sparse_mla_softmax_kernel<<<batch_size * active_heads, kThreads, 0, stream>>>(
        scores.data_ptr<float>(), lse.data_ptr<float>(),
        attn_sink.defined() ? attn_sink.data_ptr<float>() : nullptr, batch_size,
        num_heads, active_heads, topk);
    DG_CUDA_RUNTIME_CHECK(cudaGetLastError());

    const dim3 out_grid(batch_size * active_heads,
                        (kHeadDim + kQuantBlock - 1) / kQuantBlock);
    const size_t shared_bytes = static_cast<size_t>(topk) * sizeof(float);
    sparse_mla_output_scaled_kernel<out_t, index_t>
        <<<out_grid, kThreads, shared_bytes, stream>>>(
            reinterpret_cast<const uint8_t*>(k_cache.data_ptr()),
            indices.data_ptr(),
            topk_length.defined() ? topk_length.data_ptr<int64_t>() : nullptr,
            scores.data_ptr<float>(), lse.data_ptr<float>(),
            attn_sink.defined() ? attn_sink.data_ptr<float>() : nullptr,
            reinterpret_cast<out_t*>(out.data_ptr()), batch_size, active_heads,
            num_heads, block_size, k_cache.size(0), topk, out.stride(0),
            out.stride(2), out.stride(3), indices.stride(0),
            indices.stride(2), byte_stride(k_cache, 0));
    DG_CUDA_RUNTIME_CHECK(cudaGetLastError());
}

template <typename q_t, typename out_t, typename block_t, typename seq_t,
          typename req_t>
void launch_sparse_mla_decode_full_context(
    const torch::Tensor& q, const torch::Tensor& k_cache,
    const torch::Tensor& block_table, const torch::Tensor& seq_lens,
    const torch::Tensor& req_id_per_token, const torch::Tensor& attn_sink,
    const torch::Tensor& out, const torch::Tensor& lse,
    double softmax_scale) {
    const int batch_size = static_cast<int>(q.size(0));
    const int num_heads = static_cast<int>(q.size(2));
    const int block_size = static_cast<int>(k_cache.size(1));
    const int block_table_rows = static_cast<int>(block_table.size(0));
    const int block_table_width = static_cast<int>(block_table.size(1));
    const int candidate_slots = block_table_width * block_size;
    DG_HOST_ASSERT(batch_size > 0);
    DG_HOST_ASSERT(num_heads > 0);
    DG_HOST_ASSERT(block_size > 0);
    DG_HOST_ASSERT(block_table_rows > 0 && block_table_width > 0);
    DG_HOST_ASSERT(candidate_slots > 0);
    DG_HOST_ASSERT(candidate_slots <= kMaxGroupedCandidateSlots);

    const auto stream = at::cuda::getCurrentCUDAStream();
    const int grid = batch_size * num_heads;
    const size_t shared_bytes = full_context_decode_shared_bytes(candidate_slots);
    sparse_mla_decode_full_context_kernel<q_t, out_t, block_t, seq_t, req_t>
        <<<grid, kThreads, shared_bytes, stream>>>(
            reinterpret_cast<const q_t*>(q.data_ptr()),
            reinterpret_cast<const uint8_t*>(k_cache.data_ptr()),
            block_table.data_ptr<block_t>(), seq_lens.data_ptr<seq_t>(),
            req_id_per_token.data_ptr<req_t>(),
            attn_sink.defined() ? attn_sink.data_ptr<float>() : nullptr,
            reinterpret_cast<out_t*>(out.data_ptr()), lse.data_ptr<float>(),
            batch_size, num_heads, block_size, k_cache.size(0),
            block_table_rows, block_table_width, candidate_slots, q.stride(0),
            q.stride(1), q.stride(2), q.stride(3), out.stride(0),
            out.stride(1), out.stride(2), out.stride(3),
            block_table.stride(0), block_table.stride(1), seq_lens.stride(0),
            req_id_per_token.stride(0), byte_stride(k_cache, 0),
            static_cast<float>(softmax_scale));
    DG_CUDA_RUNTIME_CHECK(cudaGetLastError());
}

} // namespace

void dequantize_and_gather_k_cache(
    const torch::Tensor& out, const torch::Tensor& k_cache,
    const torch::Tensor& seq_lens, const pybind11::object& gather_lens_obj,
    const torch::Tensor& block_table, int block_size, int offset) {
    DG_HOST_ASSERT(out.is_cuda() && k_cache.is_cuda() && seq_lens.is_cuda() &&
                   block_table.is_cuda());
    DG_HOST_ASSERT(out.dim() == 3 && out.size(2) >= kHeadDim);
    DG_HOST_ASSERT(k_cache.dim() >= 2);
    DG_HOST_ASSERT(block_table.dim() >= 2);
    DG_HOST_ASSERT(block_size > 0);
    DG_HOST_ASSERT(offset >= 0 && offset <= out.size(1));
    DG_HOST_ASSERT(k_cache.size(k_cache.dim() - 1) >= kTokenDataBytes + kScaleBytes);

    auto gather_lens = tensor_or_empty(gather_lens_obj);
    if (gather_lens.defined())
        DG_HOST_ASSERT(gather_lens.is_cuda() && gather_lens.numel() >= seq_lens.numel());

    auto launch_for_block = [&](auto out_tag, auto seq_tag) {
        using out_t = decltype(out_tag);
        using seq_t = decltype(seq_tag);
        if (block_table.scalar_type() == torch::kInt64) {
            if (gather_lens.defined() && gather_lens.scalar_type() == torch::kInt) {
                launch_dequantize_gather_k_cache<out_t, seq_t, int64_t, int32_t>(
                    out, k_cache, seq_lens, gather_lens, block_table, block_size,
                    offset);
            } else {
                if (gather_lens.defined())
                    DG_HOST_ASSERT(gather_lens.scalar_type() == torch::kInt64);
                launch_dequantize_gather_k_cache<out_t, seq_t, int64_t, int64_t>(
                    out, k_cache, seq_lens, gather_lens, block_table, block_size,
                    offset);
            }
        } else {
            DG_HOST_ASSERT(block_table.scalar_type() == torch::kInt);
            if (gather_lens.defined() && gather_lens.scalar_type() == torch::kInt) {
                launch_dequantize_gather_k_cache<out_t, seq_t, int32_t, int32_t>(
                    out, k_cache, seq_lens, gather_lens, block_table, block_size,
                    offset);
            } else {
                if (gather_lens.defined())
                    DG_HOST_ASSERT(gather_lens.scalar_type() == torch::kInt64);
                launch_dequantize_gather_k_cache<out_t, seq_t, int32_t, int64_t>(
                    out, k_cache, seq_lens, gather_lens, block_table, block_size,
                    offset);
            }
        }
    };

    if (out.scalar_type() == torch::kBFloat16) {
        if (seq_lens.scalar_type() == torch::kInt64)
            launch_for_block(__nv_bfloat16{}, int64_t{});
        else {
            DG_HOST_ASSERT(seq_lens.scalar_type() == torch::kInt);
            launch_for_block(__nv_bfloat16{}, int32_t{});
        }
    } else if (out.scalar_type() == torch::kFloat16) {
        if (seq_lens.scalar_type() == torch::kInt64)
            launch_for_block(half{}, int64_t{});
        else {
            DG_HOST_ASSERT(seq_lens.scalar_type() == torch::kInt);
            launch_for_block(half{}, int32_t{});
        }
    } else if (out.scalar_type() == torch::kFloat32) {
        if (seq_lens.scalar_type() == torch::kInt64)
            launch_for_block(float{}, int64_t{});
        else {
            DG_HOST_ASSERT(seq_lens.scalar_type() == torch::kInt);
            launch_for_block(float{}, int32_t{});
        }
    } else {
        DG_HOST_UNREACHABLE("SM120 gather currently supports bf16, fp16, or fp32 output");
    }
}

void dequantize_and_gather_indexed_k_cache(
    const torch::Tensor& out, const torch::Tensor& k_cache,
    const torch::Tensor& indices, const pybind11::object& topk_length_obj,
    int block_size, int offset) {
    DG_HOST_ASSERT(out.is_cuda() && k_cache.is_cuda() && indices.is_cuda());
    DG_HOST_ASSERT(out.dim() == 3 && out.size(2) >= kHeadDim);
    DG_HOST_ASSERT(indices.dim() == 2 || indices.dim() == 3);
    DG_HOST_ASSERT(indices.size(0) <= out.size(0));
    DG_HOST_ASSERT(block_size > 0);
    DG_HOST_ASSERT(offset >= 0);
    DG_HOST_ASSERT(offset + indices.size(indices.dim() == 3 ? 2 : 1) <= out.size(1));
    DG_HOST_ASSERT(k_cache.dim() >= 4 && k_cache.size(2) == 1);
    DG_HOST_ASSERT(k_cache.size(k_cache.dim() - 1) >= kTokenDataBytes + kScaleBytes);

    auto topk_length = tensor_or_empty(topk_length_obj);
    if (topk_length.defined())
        DG_HOST_ASSERT(topk_length.is_cuda() && topk_length.numel() >= indices.size(0));

    auto launch_for_len = [&](auto out_tag, auto index_tag) {
        using out_t = decltype(out_tag);
        using index_t = decltype(index_tag);
        if (topk_length.defined() && topk_length.scalar_type() == torch::kInt) {
            launch_dequantize_gather_indexed_k_cache<out_t, index_t, int32_t>(
                out, k_cache, indices, topk_length, block_size, offset);
        } else {
            if (topk_length.defined())
                DG_HOST_ASSERT(topk_length.scalar_type() == torch::kInt64);
            launch_dequantize_gather_indexed_k_cache<out_t, index_t, int64_t>(
                out, k_cache, indices, topk_length, block_size, offset);
        }
    };

    auto launch_for_index = [&](auto out_tag) {
        if (indices.scalar_type() == torch::kInt64) {
            launch_for_len(out_tag, int64_t{});
        } else {
            DG_HOST_ASSERT(indices.scalar_type() == torch::kInt);
            launch_for_len(out_tag, int32_t{});
        }
    };

    if (out.scalar_type() == torch::kBFloat16) {
        launch_for_index(__nv_bfloat16{});
    } else if (out.scalar_type() == torch::kFloat16) {
        launch_for_index(half{});
    } else if (out.scalar_type() == torch::kFloat32) {
        launch_for_index(float{});
    } else {
        DG_HOST_UNREACHABLE("SM120 indexed gather currently supports bf16, fp16, or fp32 output");
    }
}

void fill_decode_all_indices(const torch::Tensor& topk_indices,
                             const torch::Tensor& seq_lens, int num_rows,
                             int next_n, int topk) {
    DG_HOST_ASSERT(topk_indices.is_cuda() && seq_lens.is_cuda());
    DG_HOST_ASSERT(topk_indices.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(seq_lens.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(topk_indices.dim() == 2);
    DG_HOST_ASSERT(seq_lens.dim() == 1 || seq_lens.dim() == 2);
    DG_HOST_ASSERT(seq_lens.is_contiguous());
    DG_HOST_ASSERT(num_rows >= 0 && next_n > 0 && topk >= 0);
    DG_HOST_ASSERT(topk_indices.size(0) >= num_rows);
    DG_HOST_ASSERT(topk_indices.size(1) >= topk);
    DG_HOST_ASSERT(seq_lens.dim() == 2
                       ? seq_lens.numel() >= num_rows
                       : seq_lens.size(0) * next_n >= num_rows);
    if (num_rows == 0 || topk == 0)
        return;

    const auto stream = at::cuda::getCurrentCUDAStream();
    constexpr int threads = 256;
    const dim3 grid((topk + threads - 1) / threads, num_rows);
    fill_decode_all_indices_kernel<<<grid, threads, 0, stream>>>(
        topk_indices.data_ptr<int32_t>(), seq_lens.data_ptr<int32_t>(),
        num_rows, next_n, topk, topk_indices.stride(0),
        topk_indices.stride(1), seq_lens.dim() == 2);
    DG_CUDA_RUNTIME_CHECK(cudaGetLastError());
}

std::tuple<torch::Tensor, torch::Tensor> sparse_mla_decode_from_bf16_workspace(
    const torch::Tensor& q, const torch::Tensor& kv_workspace,
    const pybind11::object& topk_length_obj,
    const pybind11::object& extra_topk_length_obj,
    const pybind11::object& attn_sink_obj, int main_topk, int extra_topk,
    int head_dim_v, double softmax_scale, const pybind11::object& out_obj) {
    DG_HOST_ASSERT(q.is_cuda() && kv_workspace.is_cuda());
    DG_HOST_ASSERT(q.dim() == 4 && q.size(1) == 1 && q.size(3) == kHeadDim);
    DG_HOST_ASSERT(kv_workspace.dim() == 3 && kv_workspace.size(2) >= kHeadDim);
    DG_HOST_ASSERT(head_dim_v == kHeadDim);
    DG_HOST_ASSERT(main_topk >= 0 && extra_topk >= 0);
    DG_HOST_ASSERT(main_topk + extra_topk > 0);
    DG_HOST_ASSERT(main_topk + extra_topk <= kv_workspace.size(1));

    auto topk_length = tensor_or_empty(topk_length_obj);
    auto extra_topk_length = tensor_or_empty(extra_topk_length_obj);
    auto attn_sink = tensor_or_empty(attn_sink_obj);
    if (topk_length.defined()) {
        DG_HOST_ASSERT(topk_length.is_cuda() && topk_length.numel() >= q.size(0));
        DG_HOST_ASSERT(topk_length.scalar_type() == torch::kInt ||
                       topk_length.scalar_type() == torch::kInt64);
    }
    if (extra_topk_length.defined()) {
        DG_HOST_ASSERT(extra_topk_length.is_cuda() &&
                       extra_topk_length.numel() >= q.size(0));
        DG_HOST_ASSERT(extra_topk_length.scalar_type() == torch::kInt ||
                       extra_topk_length.scalar_type() == torch::kInt64);
    }
    if (attn_sink.defined() && attn_sink.scalar_type() != torch::kFloat32)
        attn_sink = attn_sink.to(torch::kFloat32);

    torch::Tensor out;
    if (is_none(out_obj)) {
        out = torch::empty({q.size(0), q.size(1), q.size(2), head_dim_v},
                           q.options());
    } else {
        out = out_obj.cast<torch::Tensor>();
    }
    auto lse = torch::empty({q.size(0), q.size(2), q.size(1)},
                            q.options().dtype(torch::kFloat32));
    const auto stream = at::cuda::getCurrentCUDAStream();
    static sm120_profile::KernelProfileCounter profile_counter(
        "sm120_sparse_mla_workspace");
    sm120_profile::ScopedTimer profile_timer(
        profile_counter, stream, static_cast<int>(q.size(0)),
        static_cast<int>(q.size(2)), main_topk + extra_topk, 1);

    if (q.scalar_type() == torch::kBFloat16 &&
        kv_workspace.scalar_type() == torch::kBFloat16 &&
        out.scalar_type() == torch::kBFloat16) {
        launch_sparse_mla_decode_from_workspace<__nv_bfloat16, __nv_bfloat16,
                                                __nv_bfloat16>(
            q, kv_workspace, topk_length, extra_topk_length, attn_sink, out,
            lse, main_topk, extra_topk, softmax_scale);
    } else if (q.scalar_type() == torch::kFloat16 &&
               kv_workspace.scalar_type() == torch::kFloat16 &&
               out.scalar_type() == torch::kFloat16) {
        launch_sparse_mla_decode_from_workspace<half, half, half>(
            q, kv_workspace, topk_length, extra_topk_length, attn_sink, out,
            lse, main_topk, extra_topk, softmax_scale);
    } else {
        DG_HOST_UNREACHABLE(
            "SM120 workspace sparse MLA decode requires matching bf16 or fp16 tensors");
    }

    return std::make_tuple(out, lse);
}

std::tuple<torch::Tensor, torch::Tensor>
sparse_mla_decode_from_bf16_workspace_split(
    const torch::Tensor& q, const torch::Tensor& kv_workspace,
    const pybind11::object& topk_length_obj,
    const pybind11::object& extra_topk_length_obj,
    const pybind11::object& attn_sink_obj, int main_topk, int extra_topk,
    int head_dim_v, double softmax_scale, const pybind11::object& out_obj) {
    DG_HOST_ASSERT(q.is_cuda() && kv_workspace.is_cuda());
    DG_HOST_ASSERT(q.dim() == 4 && q.size(1) == 1 && q.size(3) == kHeadDim);
    DG_HOST_ASSERT(kv_workspace.dim() == 3 && kv_workspace.size(2) >= kHeadDim);
    DG_HOST_ASSERT(head_dim_v == kHeadDim);
    DG_HOST_ASSERT(main_topk >= 0 && extra_topk >= 0);
    DG_HOST_ASSERT(main_topk + extra_topk > 0);
    DG_HOST_ASSERT(main_topk + extra_topk <= kv_workspace.size(1));

    auto topk_length = tensor_or_empty(topk_length_obj);
    auto extra_topk_length = tensor_or_empty(extra_topk_length_obj);
    auto attn_sink = tensor_or_empty(attn_sink_obj);
    if (topk_length.defined()) {
        DG_HOST_ASSERT(topk_length.is_cuda() && topk_length.numel() >= q.size(0));
        DG_HOST_ASSERT(topk_length.scalar_type() == torch::kInt ||
                       topk_length.scalar_type() == torch::kInt64);
    }
    if (extra_topk_length.defined()) {
        DG_HOST_ASSERT(extra_topk_length.is_cuda() &&
                       extra_topk_length.numel() >= q.size(0));
        DG_HOST_ASSERT(extra_topk_length.scalar_type() == torch::kInt ||
                       extra_topk_length.scalar_type() == torch::kInt64);
    }
    if (attn_sink.defined() && attn_sink.scalar_type() != torch::kFloat32)
        attn_sink = attn_sink.to(torch::kFloat32);

    torch::Tensor out;
    if (is_none(out_obj)) {
        out = torch::empty({q.size(0), q.size(1), q.size(2), head_dim_v},
                           q.options());
    } else {
        out = out_obj.cast<torch::Tensor>();
    }
    auto lse = torch::empty({q.size(0), q.size(2), q.size(1)},
                            q.options().dtype(torch::kFloat32));
    const auto stream = at::cuda::getCurrentCUDAStream();
    static sm120_profile::KernelProfileCounter profile_counter(
        "sm120_sparse_mla_workspace_split");
    sm120_profile::ScopedTimer profile_timer(
        profile_counter, stream, static_cast<int>(q.size(0)),
        static_cast<int>(q.size(2)), main_topk + extra_topk, 1);

    if (q.scalar_type() == torch::kBFloat16 &&
        kv_workspace.scalar_type() == torch::kBFloat16 &&
        out.scalar_type() == torch::kBFloat16) {
        launch_sparse_mla_decode_from_workspace_split<__nv_bfloat16,
                                                      __nv_bfloat16,
                                                      __nv_bfloat16>(
            q, kv_workspace, topk_length, extra_topk_length, attn_sink, out,
            lse, main_topk, extra_topk, softmax_scale);
    } else if (q.scalar_type() == torch::kFloat16 &&
               kv_workspace.scalar_type() == torch::kFloat16 &&
               out.scalar_type() == torch::kFloat16) {
        launch_sparse_mla_decode_from_workspace_split<half, half, half>(
            q, kv_workspace, topk_length, extra_topk_length, attn_sink, out,
            lse, main_topk, extra_topk, softmax_scale);
    } else {
        DG_HOST_UNREACHABLE(
            "SM120 split workspace sparse MLA decode requires matching bf16 or fp16 tensors");
    }

    return std::make_tuple(out, lse);
}

std::tuple<torch::Tensor, torch::Tensor> sparse_mla_decode(
    const torch::Tensor& q, const torch::Tensor& k_cache,
    const torch::Tensor& indices, const pybind11::object& topk_length_obj,
    const pybind11::object& attn_sink_obj,
    const pybind11::object& extra_k_cache_obj,
    const pybind11::object& extra_indices_obj,
    const pybind11::object& extra_topk_length_obj, int head_dim_v,
    double softmax_scale, const pybind11::object& out_obj) {
    DG_HOST_ASSERT(q.is_cuda() && k_cache.is_cuda() && indices.is_cuda());
    DG_HOST_ASSERT(q.dim() == 4 && q.size(1) == 1 && q.size(3) == kHeadDim);
    DG_HOST_ASSERT(k_cache.dim() >= 4 && k_cache.size(2) == 1);
    DG_HOST_ASSERT(head_dim_v == kHeadDim);
    DG_HOST_ASSERT(k_cache.size(k_cache.dim() - 1) >= kTokenDataBytes + kScaleBytes);

    auto topk_length = tensor_or_empty(topk_length_obj);
    auto attn_sink = tensor_or_empty(attn_sink_obj);
    auto extra_k_cache = tensor_or_empty(extra_k_cache_obj);
    auto extra_indices = tensor_or_empty(extra_indices_obj);
    auto extra_topk_length = tensor_or_empty(extra_topk_length_obj);

    if (topk_length.defined() && topk_length.scalar_type() != torch::kInt64)
        topk_length = topk_length.to(torch::kInt64);
    if (extra_topk_length.defined() &&
        extra_topk_length.scalar_type() != torch::kInt64)
        extra_topk_length = extra_topk_length.to(torch::kInt64);
    if (attn_sink.defined() && attn_sink.scalar_type() != torch::kFloat32)
        attn_sink = attn_sink.to(torch::kFloat32);

    torch::Tensor out;
    if (is_none(out_obj)) {
        out = torch::empty({q.size(0), q.size(1), q.size(2), head_dim_v},
                           q.options());
    } else {
        out = out_obj.cast<torch::Tensor>();
    }
    auto lse = torch::empty({q.size(0), q.size(2), q.size(1)},
                            q.options().dtype(torch::kFloat32));
    const auto stream = at::cuda::getCurrentCUDAStream();
    static sm120_profile::KernelProfileCounter profile_counter(
        "sm120_sparse_mla_decode");
    sm120_profile::ScopedTimer profile_timer(
        profile_counter, stream, static_cast<int>(q.size(0)),
        static_cast<int>(q.size(2)), static_cast<int>(indices.size(-1)), 1);

    if (q.scalar_type() == torch::kBFloat16 && out.scalar_type() == torch::kBFloat16) {
        if (indices.scalar_type() == torch::kInt64) {
            if (!extra_k_cache.defined() && !extra_indices.defined()) {
                launch_sparse_mla_decode_fast<__nv_bfloat16, __nv_bfloat16, int64_t>(
                    q, k_cache, indices, topk_length, attn_sink, out, lse,
                    softmax_scale);
            } else {
                launch_sparse_mla_decode<__nv_bfloat16, __nv_bfloat16, int64_t>(
                    q, k_cache, indices, topk_length, attn_sink, extra_k_cache,
                    extra_indices, extra_topk_length, out, lse, softmax_scale);
            }
        } else if (indices.scalar_type() == torch::kInt32) {
            if (!extra_k_cache.defined() && !extra_indices.defined()) {
                launch_sparse_mla_decode_fast<__nv_bfloat16, __nv_bfloat16, int32_t>(
                    q, k_cache, indices, topk_length, attn_sink, out, lse,
                    softmax_scale);
            } else {
                launch_sparse_mla_decode<__nv_bfloat16, __nv_bfloat16, int32_t>(
                    q, k_cache, indices, topk_length, attn_sink, extra_k_cache,
                    extra_indices, extra_topk_length, out, lse, softmax_scale);
            }
        } else {
            DG_HOST_UNREACHABLE("SM120 sparse MLA decode indices must be int32 or int64");
        }
    } else if (q.scalar_type() == torch::kFloat16 && out.scalar_type() == torch::kFloat16) {
        if (indices.scalar_type() == torch::kInt64) {
            if (!extra_k_cache.defined() && !extra_indices.defined()) {
                launch_sparse_mla_decode_fast<half, half, int64_t>(
                    q, k_cache, indices, topk_length, attn_sink, out, lse,
                    softmax_scale);
            } else {
                launch_sparse_mla_decode<half, half, int64_t>(
                    q, k_cache, indices, topk_length, attn_sink, extra_k_cache,
                    extra_indices, extra_topk_length, out, lse, softmax_scale);
            }
        } else if (indices.scalar_type() == torch::kInt32) {
            if (!extra_k_cache.defined() && !extra_indices.defined()) {
                launch_sparse_mla_decode_fast<half, half, int32_t>(
                    q, k_cache, indices, topk_length, attn_sink, out, lse,
                    softmax_scale);
            } else {
                launch_sparse_mla_decode<half, half, int32_t>(
                    q, k_cache, indices, topk_length, attn_sink, extra_k_cache,
                    extra_indices, extra_topk_length, out, lse, softmax_scale);
            }
        } else {
            DG_HOST_UNREACHABLE("SM120 sparse MLA decode indices must be int32 or int64");
        }
    } else {
        DG_HOST_UNREACHABLE("SM120 sparse MLA decode currently supports bf16 or fp16 q/out");
    }

    return std::make_tuple(out, lse);
}

std::tuple<torch::Tensor, torch::Tensor> sparse_mla_decode_fused(
    const torch::Tensor& q, const torch::Tensor& k_cache,
    const torch::Tensor& indices, const pybind11::object& topk_length_obj,
    const pybind11::object& attn_sink_obj, int head_dim_v,
    double softmax_scale, const pybind11::object& out_obj) {
    DG_HOST_ASSERT(q.is_cuda() && k_cache.is_cuda() && indices.is_cuda());
    DG_HOST_ASSERT(q.dim() == 4 && q.size(1) == 1 && q.size(3) == kHeadDim);
    DG_HOST_ASSERT(k_cache.dim() >= 4 && k_cache.size(2) == 1);
    DG_HOST_ASSERT(indices.dim() == 3);
    DG_HOST_ASSERT(head_dim_v == kHeadDim);
    DG_HOST_ASSERT(k_cache.size(k_cache.dim() - 1) >=
                   kTokenDataBytes + kScaleBytes);
    DG_HOST_ASSERT(indices.size(2) <= kMaxGroupedCandidateSlots);

    auto topk_length = tensor_or_empty(topk_length_obj);
    auto attn_sink = tensor_or_empty(attn_sink_obj);
    if (topk_length.defined() && topk_length.scalar_type() != torch::kInt64)
        topk_length = topk_length.to(torch::kInt64);
    if (attn_sink.defined() && attn_sink.scalar_type() != torch::kFloat32)
        attn_sink = attn_sink.to(torch::kFloat32);

    torch::Tensor out;
    if (is_none(out_obj)) {
        out = torch::empty({q.size(0), q.size(1), q.size(2), head_dim_v},
                           q.options());
    } else {
        out = out_obj.cast<torch::Tensor>();
    }
    auto lse = torch::empty({q.size(0), q.size(2), q.size(1)},
                            q.options().dtype(torch::kFloat32));
    const auto stream = at::cuda::getCurrentCUDAStream();
    static sm120_profile::KernelProfileCounter profile_counter(
        "sm120_sparse_mla_decode_fused");
    sm120_profile::ScopedTimer profile_timer(
        profile_counter, stream, static_cast<int>(q.size(0)),
        static_cast<int>(q.size(2)), static_cast<int>(indices.size(2)), 1);

    if (q.scalar_type() == torch::kBFloat16 &&
        out.scalar_type() == torch::kBFloat16) {
        if (indices.scalar_type() == torch::kInt64) {
            launch_sparse_mla_decode_scaled<__nv_bfloat16, __nv_bfloat16,
                                            int64_t>(
                q, k_cache, indices, topk_length, attn_sink, out, lse,
                softmax_scale);
        } else if (indices.scalar_type() == torch::kInt32) {
            launch_sparse_mla_decode_scaled<__nv_bfloat16, __nv_bfloat16,
                                            int32_t>(
                q, k_cache, indices, topk_length, attn_sink, out, lse,
                softmax_scale);
        } else {
            DG_HOST_UNREACHABLE(
                "SM120 fused sparse MLA decode indices must be int32 or int64");
        }
    } else if (q.scalar_type() == torch::kFloat16 &&
               out.scalar_type() == torch::kFloat16) {
        if (indices.scalar_type() == torch::kInt64) {
            launch_sparse_mla_decode_scaled<half, half, int64_t>(
                q, k_cache, indices, topk_length, attn_sink, out, lse,
                softmax_scale);
        } else if (indices.scalar_type() == torch::kInt32) {
            launch_sparse_mla_decode_scaled<half, half, int32_t>(
                q, k_cache, indices, topk_length, attn_sink, out, lse,
                softmax_scale);
        } else {
            DG_HOST_UNREACHABLE(
                "SM120 fused sparse MLA decode indices must be int32 or int64");
        }
    } else {
        DG_HOST_UNREACHABLE(
            "SM120 fused sparse MLA decode currently supports bf16 or fp16 q/out");
    }

    return std::make_tuple(out, lse);
}

std::tuple<torch::Tensor, torch::Tensor> sparse_mla_decode_full_context(
    const torch::Tensor& q, const torch::Tensor& k_cache,
    const torch::Tensor& block_table, const torch::Tensor& seq_lens,
    const torch::Tensor& req_id_per_token,
    const pybind11::object& attn_sink_obj, int head_dim_v,
    double softmax_scale, const pybind11::object& out_obj) {
    DG_HOST_ASSERT(q.is_cuda() && k_cache.is_cuda() && block_table.is_cuda() &&
                   seq_lens.is_cuda() && req_id_per_token.is_cuda());
    DG_HOST_ASSERT(q.dim() == 4 && q.size(1) == 1 && q.size(3) == kHeadDim);
    DG_HOST_ASSERT(k_cache.dim() >= 4 && k_cache.size(2) == 1);
    DG_HOST_ASSERT(block_table.dim() == 2);
    DG_HOST_ASSERT(seq_lens.dim() == 1);
    DG_HOST_ASSERT(req_id_per_token.dim() == 1);
    DG_HOST_ASSERT(req_id_per_token.numel() >= q.size(0));
    DG_HOST_ASSERT(head_dim_v == kHeadDim);
    DG_HOST_ASSERT(k_cache.size(k_cache.dim() - 1) >= kTokenDataBytes + kScaleBytes);

    auto attn_sink = tensor_or_empty(attn_sink_obj);
    if (attn_sink.defined() && attn_sink.scalar_type() != torch::kFloat32)
        attn_sink = attn_sink.to(torch::kFloat32);

    torch::Tensor out;
    if (is_none(out_obj)) {
        out = torch::empty({q.size(0), q.size(1), q.size(2), head_dim_v},
                           q.options());
    } else {
        out = out_obj.cast<torch::Tensor>();
    }
    auto lse = torch::empty({q.size(0), q.size(2), q.size(1)},
                            q.options().dtype(torch::kFloat32));
    const auto stream = at::cuda::getCurrentCUDAStream();
    static sm120_profile::KernelProfileCounter profile_counter(
        "sm120_sparse_mla_full_context");
    sm120_profile::ScopedTimer profile_timer(
        profile_counter, stream, static_cast<int>(q.size(0)),
        static_cast<int>(q.size(2)), static_cast<int>(seq_lens.numel()), 1);

    auto launch_for_req = [&](auto q_tag, auto out_tag, auto block_tag,
                              auto seq_tag, auto req_tag) {
        using q_t = decltype(q_tag);
        using out_t = decltype(out_tag);
        using block_t = decltype(block_tag);
        using seq_t = decltype(seq_tag);
        using req_t = decltype(req_tag);
        launch_sparse_mla_decode_full_context<q_t, out_t, block_t, seq_t,
                                              req_t>(
            q, k_cache, block_table, seq_lens, req_id_per_token, attn_sink, out,
            lse, softmax_scale);
    };

    auto dispatch_req = [&](auto q_tag, auto out_tag, auto block_tag,
                            auto seq_tag) {
        if (req_id_per_token.scalar_type() == torch::kInt64) {
            launch_for_req(q_tag, out_tag, block_tag, seq_tag, int64_t{});
        } else {
            DG_HOST_ASSERT(req_id_per_token.scalar_type() == torch::kInt);
            launch_for_req(q_tag, out_tag, block_tag, seq_tag, int32_t{});
        }
    };

    auto dispatch_seq = [&](auto q_tag, auto out_tag, auto block_tag) {
        if (seq_lens.scalar_type() == torch::kInt64) {
            dispatch_req(q_tag, out_tag, block_tag, int64_t{});
        } else {
            DG_HOST_ASSERT(seq_lens.scalar_type() == torch::kInt);
            dispatch_req(q_tag, out_tag, block_tag, int32_t{});
        }
    };

    auto dispatch_block = [&](auto q_tag, auto out_tag) {
        if (block_table.scalar_type() == torch::kInt64) {
            dispatch_seq(q_tag, out_tag, int64_t{});
        } else {
            DG_HOST_ASSERT(block_table.scalar_type() == torch::kInt);
            dispatch_seq(q_tag, out_tag, int32_t{});
        }
    };

    if (q.scalar_type() == torch::kBFloat16 &&
        out.scalar_type() == torch::kBFloat16) {
        dispatch_block(__nv_bfloat16{}, __nv_bfloat16{});
    } else if (q.scalar_type() == torch::kFloat16 &&
               out.scalar_type() == torch::kFloat16) {
        dispatch_block(half{}, half{});
    } else {
        DG_HOST_UNREACHABLE(
            "SM120 full-context sparse MLA decode currently supports bf16 or fp16 q/out");
    }

    return std::make_tuple(out, lse);
}

template <typename q_t, typename kv_t, typename out_t, typename index_t>
void launch_sparse_mla_prefill_from_workspace(
    const torch::Tensor& q, const torch::Tensor& kv,
    const torch::Tensor& indices, const torch::Tensor& topk_length,
    const torch::Tensor& attn_sink, const torch::Tensor& out,
    const torch::Tensor& max_logits, const torch::Tensor& lse,
    double softmax_scale) {
    const int num_tokens = static_cast<int>(q.size(0));
    const int num_heads = static_cast<int>(q.size(1));
    const int active_heads = sm120_active_heads(num_heads);
    const int kv_tokens = static_cast<int>(kv.size(0));
    const int topk = static_cast<int>(indices.size(2));
    const int topk_kind = length_tensor_kind(topk_length);
    const void* topk_ptr =
        topk_length.defined() ? topk_length.data_ptr() : nullptr;
    const float* sink_ptr =
        attn_sink.defined() ? attn_sink.data_ptr<float>() : nullptr;
    const auto stream = at::cuda::getCurrentCUDAStream();
    const dim3 grid(num_tokens, active_heads);
    const size_t shared_bytes =
        (static_cast<size_t>(topk) + kThreads) * sizeof(float);

    DG_CUDA_RUNTIME_CHECK(cudaFuncSetAttribute(
        sparse_mla_prefill_workspace_kernel<q_t, kv_t, out_t, index_t>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(shared_bytes)));
    sparse_mla_prefill_workspace_kernel<q_t, kv_t, out_t, index_t>
        <<<grid, kThreads, shared_bytes, stream>>>(
            reinterpret_cast<const q_t*>(q.data_ptr()),
            reinterpret_cast<const kv_t*>(kv.data_ptr()),
            reinterpret_cast<const index_t*>(indices.data_ptr()), topk_ptr,
            topk_kind, sink_ptr, reinterpret_cast<out_t*>(out.data_ptr()),
            max_logits.data_ptr<float>(), lse.data_ptr<float>(), num_heads,
            kv_tokens, topk, q.stride(0), q.stride(1), q.stride(2),
            kv.stride(0), kv.stride(2), indices.stride(0), indices.stride(2),
            out.stride(0), out.stride(1), out.stride(2), softmax_scale);
    DG_CUDA_RUNTIME_CHECK(cudaGetLastError());
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
sparse_mla_prefill_from_bf16_workspace(
    const torch::Tensor& q, const torch::Tensor& kv,
    const torch::Tensor& indices, const pybind11::object& topk_length_obj,
    const pybind11::object& attn_sink_obj, int head_dim_v, double softmax_scale,
    const pybind11::object& out_obj) {
    DG_HOST_ASSERT(q.is_cuda() && kv.is_cuda() && indices.is_cuda());
    DG_HOST_ASSERT(q.dim() == 3 && q.size(2) == kHeadDim);
    DG_HOST_ASSERT(kv.dim() == 3 && kv.size(1) == 1 && kv.size(2) >= kHeadDim);
    DG_HOST_ASSERT(indices.dim() == 3 && indices.size(0) == q.size(0) &&
                   indices.size(1) == 1);
    DG_HOST_ASSERT(head_dim_v == kHeadDim);

    auto topk_length = tensor_or_empty(topk_length_obj);
    auto attn_sink = tensor_or_empty(attn_sink_obj);
    if (topk_length.defined()) {
        DG_HOST_ASSERT(topk_length.is_cuda() && topk_length.numel() >= q.size(0));
        DG_HOST_ASSERT(topk_length.scalar_type() == torch::kInt ||
                       topk_length.scalar_type() == torch::kInt64);
    }
    if (attn_sink.defined() && attn_sink.scalar_type() != torch::kFloat32)
        attn_sink = attn_sink.to(torch::kFloat32);

    torch::Tensor out;
    if (is_none(out_obj)) {
        out = torch::empty({q.size(0), q.size(1), head_dim_v}, q.options());
    } else {
        out = out_obj.cast<torch::Tensor>();
    }
    auto max_logits =
        torch::empty({q.size(0), q.size(1)}, q.options().dtype(torch::kFloat32));
    auto lse =
        torch::empty({q.size(0), q.size(1)}, q.options().dtype(torch::kFloat32));

    const auto stream = at::cuda::getCurrentCUDAStream();
    static sm120_profile::KernelProfileCounter profile_counter(
        "sm120_sparse_mla_prefill_workspace");
    sm120_profile::ScopedTimer profile_timer(
        profile_counter, stream, static_cast<int>(q.size(0)),
        static_cast<int>(q.size(1)), static_cast<int>(indices.size(2)), 1);

    if (q.scalar_type() == torch::kBFloat16 &&
        kv.scalar_type() == torch::kBFloat16 &&
        out.scalar_type() == torch::kBFloat16) {
        if (indices.scalar_type() == torch::kInt64) {
            launch_sparse_mla_prefill_from_workspace<__nv_bfloat16,
                                                     __nv_bfloat16,
                                                     __nv_bfloat16, int64_t>(
                q, kv, indices, topk_length, attn_sink, out, max_logits, lse,
                softmax_scale);
        } else if (indices.scalar_type() == torch::kInt32) {
            launch_sparse_mla_prefill_from_workspace<__nv_bfloat16,
                                                     __nv_bfloat16,
                                                     __nv_bfloat16, int32_t>(
                q, kv, indices, topk_length, attn_sink, out, max_logits, lse,
                softmax_scale);
        } else {
            DG_HOST_UNREACHABLE(
                "SM120 sparse MLA prefill indices must be int32 or int64");
        }
    } else if (q.scalar_type() == torch::kFloat16 &&
               kv.scalar_type() == torch::kFloat16 &&
               out.scalar_type() == torch::kFloat16) {
        if (indices.scalar_type() == torch::kInt64) {
            launch_sparse_mla_prefill_from_workspace<half, half, half, int64_t>(
                q, kv, indices, topk_length, attn_sink, out, max_logits, lse,
                softmax_scale);
        } else if (indices.scalar_type() == torch::kInt32) {
            launch_sparse_mla_prefill_from_workspace<half, half, half, int32_t>(
                q, kv, indices, topk_length, attn_sink, out, max_logits, lse,
                softmax_scale);
        } else {
            DG_HOST_UNREACHABLE(
                "SM120 sparse MLA prefill indices must be int32 or int64");
        }
    } else {
        DG_HOST_UNREACHABLE(
            "SM120 sparse MLA prefill currently supports bf16 or fp16 q/kv/out");
    }

    return std::make_tuple(out, max_logits, lse);
}

void register_apis(pybind11::module& m) {
    m.def("sm120_sparse_mla_decode", &sparse_mla_decode,
          "SM120 sparse MLA decode for DeepSeek-V4 Flash");
    m.def("sm120_sparse_mla_decode_fused", &sparse_mla_decode_fused,
          "SM120 fused indexed sparse MLA decode for DeepSeek-V4 Flash");
    m.def("sm120_sparse_mla_decode_full_context",
          &sparse_mla_decode_full_context,
          "SM120 full-context sparse MLA decode for DeepSeek-V4 Flash");
    m.def("sm120_dequantize_and_gather_k_cache", &dequantize_and_gather_k_cache,
          "SM120 DeepSeek-V4 fp8_ds_mla KV cache gather");
    m.def("sm120_dequantize_and_gather_indexed_k_cache",
          &dequantize_and_gather_indexed_k_cache,
          "SM120 DeepSeek-V4 fp8_ds_mla indexed KV cache gather");
    m.def("sm120_sparse_mla_decode_from_bf16_workspace",
          &sparse_mla_decode_from_bf16_workspace,
          "SM120 sparse MLA decode from gathered BF16/FP16 KV workspace");
    m.def("sm120_sparse_mla_decode_from_bf16_workspace_split",
          &sparse_mla_decode_from_bf16_workspace_split,
          "SM120 split sparse MLA decode from gathered BF16/FP16 KV workspace");
    m.def("sm120_sparse_mla_prefill_from_bf16_workspace",
          &sparse_mla_prefill_from_bf16_workspace,
          "SM120 sparse MLA prefill from gathered BF16/FP16 KV workspace");
    m.def("sm120_fill_decode_all_indices", &fill_decode_all_indices,
          "SM120 fill full-context sparse decode indices");
}

} // namespace sm120_mla
} // namespace deep_gemm
