#include <algorithm>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cstdint>
#include <limits>
#include <torch/python.h>

#include "jit_kernels/impls/sm120_metadata.hpp"
#include "utils/exception.hpp"

namespace deep_gemm::sm120_metadata {
namespace {

template <typename StartT>
__global__ void fill_token_to_req_indices_kernel(
    int32_t* __restrict__ out,
    const StartT* __restrict__ query_start_loc,
    int num_reqs,
    int max_tokens) {
    const int req = static_cast<int>(blockIdx.x);
    if (req >= num_reqs)
        return;

    int64_t begin = static_cast<int64_t>(query_start_loc[req]);
    int64_t end = static_cast<int64_t>(query_start_loc[req + 1]);
    const int64_t max_token_i64 = static_cast<int64_t>(max_tokens);
    begin = begin < 0 ? 0 : (begin > max_token_i64 ? max_token_i64 : begin);
    end = end < begin ? begin : (end > max_token_i64 ? max_token_i64 : end);

    for (int64_t idx = begin + threadIdx.x; idx < end; idx += blockDim.x) {
        out[idx] = static_cast<int32_t>(req);
    }
}

template <typename T>
__device__ __forceinline__ void clamp_block_table_nonnegative(T* data,
                                                              int64_t idx) {
    const T value = data[idx];
    data[idx] = value < static_cast<T>(0) ? static_cast<T>(0) : value;
}

template <typename StartT, typename BlockT>
__global__ void build_compressor_metadata_kernel(
    int32_t* __restrict__ token_to_req,
    const StartT* __restrict__ query_start_loc,
    BlockT* __restrict__ block_table,
    int num_reqs,
    int max_tokens,
    int64_t block_table_numel) {
    constexpr int kThreads = 256;
    if (blockIdx.y == 0) {
        const int req = static_cast<int>(blockIdx.x);
        if (req >= num_reqs)
            return;
        int64_t begin = static_cast<int64_t>(query_start_loc[req]);
        int64_t end = static_cast<int64_t>(query_start_loc[req + 1]);
        const int64_t max_token_i64 = static_cast<int64_t>(max_tokens);
        begin = begin < 0 ? 0 : (begin > max_token_i64 ? max_token_i64 : begin);
        end = end < begin ? begin : (end > max_token_i64 ? max_token_i64 : end);
        for (int64_t idx = begin + threadIdx.x; idx < end; idx += blockDim.x)
            token_to_req[idx] = static_cast<int32_t>(req);
        return;
    }

    const int64_t idx = static_cast<int64_t>(blockIdx.x) * kThreads + threadIdx.x;
    if (idx < block_table_numel)
        clamp_block_table_nonnegative<BlockT>(block_table, idx);
}

template <typename StartT, typename SlotT>
__global__ void build_sparse_swa_metadata_kernel(
    int32_t* __restrict__ token_to_req,
    bool* __restrict__ is_valid_token,
    const StartT* __restrict__ query_start_loc,
    const SlotT* __restrict__ slot_mapping,
    int32_t* __restrict__ decode_swa_lens,
    int num_reqs,
    int max_tokens,
    int64_t slot_mapping_numel,
    int decode_swa_lens_numel,
    int num_decode_tokens) {
    constexpr int kThreads = 256;
    if (blockIdx.y == 0) {
        const int req = static_cast<int>(blockIdx.x);
        if (req >= num_reqs)
            return;
        int64_t begin = static_cast<int64_t>(query_start_loc[req]);
        int64_t end = static_cast<int64_t>(query_start_loc[req + 1]);
        const int64_t max_token_i64 = static_cast<int64_t>(max_tokens);
        begin = begin < 0 ? 0 : (begin > max_token_i64 ? max_token_i64 : begin);
        end = end < begin ? begin : (end > max_token_i64 ? max_token_i64 : end);
        for (int64_t idx = begin + threadIdx.x; idx < end; idx += blockDim.x)
            token_to_req[idx] = static_cast<int32_t>(req);
        return;
    }

    const int64_t idx = static_cast<int64_t>(blockIdx.x) * kThreads + threadIdx.x;
    if (blockIdx.y == 1) {
        if (idx < slot_mapping_numel)
            is_valid_token[idx] = slot_mapping[idx] >= static_cast<SlotT>(0);
        return;
    }

    const int64_t decode_idx = static_cast<int64_t>(num_decode_tokens) + idx;
    if (decode_idx < decode_swa_lens_numel)
        decode_swa_lens[decode_idx] = 0;
}

template <typename StartT>
__device__ __forceinline__ int find_req_for_token(const StartT* __restrict__ query_start_loc,
                                                  int num_reqs,
                                                  int64_t token_idx) {
    int lo = 0;
    int hi = num_reqs;
    while (lo + 1 < hi) {
        const int mid = (lo + hi) >> 1;
        if (static_cast<int64_t>(query_start_loc[mid]) <= token_idx) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    return lo;
}

template <typename StartT, typename SlotT, typename SeqT, typename BlockT, typename OutT>
__global__ void build_sparse_swa_decode_metadata_kernel(
    int32_t* __restrict__ token_to_req,
    bool* __restrict__ is_valid_token,
    const StartT* __restrict__ query_start_loc,
    const SlotT* __restrict__ slot_mapping,
    int32_t* __restrict__ decode_swa_lens,
    OutT* __restrict__ decode_swa_indices,
    int64_t decode_swa_indices_stride,
    const SeqT* __restrict__ seq_lens,
    const BlockT* __restrict__ block_table,
    int64_t block_table_stride,
    int num_reqs,
    int max_tokens,
    int64_t slot_mapping_numel,
    int decode_swa_lens_numel,
    int num_decode_tokens,
    int window_size,
    int block_size) {
    constexpr int kThreads = 256;

    if (blockIdx.y == 0) {
        const int req = static_cast<int>(blockIdx.x);
        if (req >= num_reqs)
            return;
        int64_t begin = static_cast<int64_t>(query_start_loc[req]);
        int64_t end = static_cast<int64_t>(query_start_loc[req + 1]);
        const int64_t max_token_i64 = static_cast<int64_t>(max_tokens);
        begin = begin < 0 ? 0 : (begin > max_token_i64 ? max_token_i64 : begin);
        end = end < begin ? begin : (end > max_token_i64 ? max_token_i64 : end);
        for (int64_t idx = begin + threadIdx.x; idx < end; idx += blockDim.x)
            token_to_req[idx] = static_cast<int32_t>(req);
        return;
    }

    if (blockIdx.y == 1) {
        const int64_t idx = static_cast<int64_t>(blockIdx.x) * kThreads + threadIdx.x;
        if (idx < slot_mapping_numel)
            is_valid_token[idx] = slot_mapping[idx] >= static_cast<SlotT>(0);
        return;
    }

    if (blockIdx.y == 2) {
        const int64_t idx = static_cast<int64_t>(blockIdx.x) * kThreads + threadIdx.x;
        const int64_t decode_idx = static_cast<int64_t>(num_decode_tokens) + idx;
        if (decode_idx < decode_swa_lens_numel)
            decode_swa_lens[decode_idx] = 0;
        return;
    }

    const int token_idx = static_cast<int>(blockIdx.x);
    if (token_idx >= num_decode_tokens)
        return;

    const bool valid_slot =
        static_cast<int64_t>(token_idx) < slot_mapping_numel &&
        slot_mapping[token_idx] >= static_cast<SlotT>(0);
    if (!valid_slot) {
        if (threadIdx.x == 0)
            decode_swa_lens[token_idx] = 0;
        for (int offset = threadIdx.x; offset < window_size; offset += blockDim.x) {
            decode_swa_indices[static_cast<int64_t>(token_idx) *
                                   decode_swa_indices_stride +
                               offset] = static_cast<OutT>(-1);
        }
        return;
    }

    const int req = find_req_for_token(query_start_loc, num_reqs, token_idx);
    const int64_t query_start = static_cast<int64_t>(query_start_loc[req]);
    const int64_t query_end = static_cast<int64_t>(query_start_loc[req + 1]);
    if (token_idx < query_start || token_idx >= query_end) {
        if (threadIdx.x == 0)
            decode_swa_lens[token_idx] = 0;
        for (int offset = threadIdx.x; offset < window_size; offset += blockDim.x) {
            decode_swa_indices[static_cast<int64_t>(token_idx) *
                                   decode_swa_indices_stride +
                               offset] = static_cast<OutT>(-1);
        }
        return;
    }

    const int64_t query_len = query_end - query_start;
    const int64_t seq_len = static_cast<int64_t>(seq_lens[req]);
    const int64_t prefix_len = seq_len - query_len;
    const int64_t pos = prefix_len + static_cast<int64_t>(token_idx) - query_start;
    const int64_t raw_start_pos = pos - static_cast<int64_t>(window_size) + 1;
    const int64_t start_pos = raw_start_pos > 0 ? raw_start_pos : 0;
    const int64_t end_pos = pos + 1;
    const int64_t swa_len_i64 = end_pos - start_pos;
    const int swa_len = static_cast<int>(
        swa_len_i64 < static_cast<int64_t>(window_size)
            ? swa_len_i64
            : static_cast<int64_t>(window_size));

    if (threadIdx.x == 0)
        decode_swa_lens[token_idx] = swa_len;

    for (int offset = threadIdx.x; offset < window_size; offset += blockDim.x) {
        OutT slot_id = static_cast<OutT>(-1);
        if (offset < swa_len) {
            const int64_t pos_offset = start_pos + offset;
            const int64_t block_in_seq = pos_offset / block_size;
            const int64_t block_offset = pos_offset - block_in_seq * block_size;
            const int64_t block_number = static_cast<int64_t>(
                block_table[static_cast<int64_t>(req) * block_table_stride +
                            block_in_seq]);
            slot_id = static_cast<OutT>(block_number * block_size + block_offset);
        }
        decode_swa_indices[static_cast<int64_t>(token_idx) *
                               decode_swa_indices_stride +
                           offset] = slot_id;
    }
}

template <typename SeqT, typename StartT>
__global__ void build_sparse_swa_prefill_metadata_kernel(
    int32_t* __restrict__ prefill_gather_lens,
    const SeqT* __restrict__ seq_lens,
    const StartT* __restrict__ query_start_loc,
    int num_prefills,
    int num_decodes,
    int window_size) {
    const int prefill_idx = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (prefill_idx >= num_prefills)
        return;

    const int req_idx = num_decodes + prefill_idx;
    const int64_t qsl_start = static_cast<int64_t>(query_start_loc[req_idx]);
    const int64_t qsl_end = static_cast<int64_t>(query_start_loc[req_idx + 1]);
    const int64_t query_len = qsl_end - qsl_start;
    const int64_t seq_len = static_cast<int64_t>(seq_lens[req_idx]);
    const int64_t prefix_len = seq_len - query_len;
    const int64_t max_prefix =
        static_cast<int64_t>(window_size) > 0
            ? static_cast<int64_t>(window_size) - 1
            : 0;
    const int64_t clipped_prefix =
        prefix_len < 0 ? 0 : (prefix_len < max_prefix ? prefix_len : max_prefix);
    prefill_gather_lens[prefill_idx] = static_cast<int32_t>(query_len + clipped_prefix);
}

template <typename StartT>
void launch_fill_token_to_req_indices(const torch::Tensor& out,
                                      const torch::Tensor& query_start_loc,
                                      int64_t num_reqs,
                                      cudaStream_t stream) {
    constexpr int kThreads = 256;
    if (num_reqs <= 0 || out.numel() == 0)
        return;
    fill_token_to_req_indices_kernel<StartT>
        <<<static_cast<unsigned>(num_reqs), kThreads, 0, stream>>>(
            out.data_ptr<int32_t>(), query_start_loc.data_ptr<StartT>(),
            static_cast<int>(num_reqs), static_cast<int>(out.numel()));
    DG_CUDA_RUNTIME_CHECK(cudaGetLastError());
}

template <typename StartT, typename BlockT>
void launch_build_compressor_metadata(const torch::Tensor& token_to_req,
                                      const torch::Tensor& query_start_loc,
                                      const torch::Tensor& block_table,
                                      int64_t num_reqs,
                                      cudaStream_t stream) {
    constexpr int kThreads = 256;
    if (num_reqs < 0 || token_to_req.numel() < 0 || block_table.numel() < 0)
        return;
    const int64_t clamp_blocks = (block_table.numel() + kThreads - 1) / kThreads;
    const int64_t grid_x_i64 = std::max<int64_t>(num_reqs, clamp_blocks);
    if (grid_x_i64 <= 0)
        return;
    DG_HOST_ASSERT(grid_x_i64 <= std::numeric_limits<unsigned>::max());
    const dim3 grid(static_cast<unsigned>(grid_x_i64), 2);
    build_compressor_metadata_kernel<StartT, BlockT><<<grid, kThreads, 0, stream>>>(
        token_to_req.data_ptr<int32_t>(), query_start_loc.data_ptr<StartT>(),
        block_table.data_ptr<BlockT>(), static_cast<int>(num_reqs),
        static_cast<int>(token_to_req.numel()), block_table.numel());
    DG_CUDA_RUNTIME_CHECK(cudaGetLastError());
}

template <typename StartT, typename SlotT>
void launch_build_sparse_swa_metadata(const torch::Tensor& token_to_req,
                                      const torch::Tensor& is_valid_token,
                                      const torch::Tensor& query_start_loc,
                                      const torch::Tensor& slot_mapping,
                                      const torch::Tensor& decode_swa_lens,
                                      int64_t num_reqs,
                                      int64_t num_decode_tokens,
                                      cudaStream_t stream) {
    constexpr int kThreads = 256;
    const int64_t valid_blocks = (slot_mapping.numel() + kThreads - 1) / kThreads;
    const int64_t zero_count =
        std::max<int64_t>(0, decode_swa_lens.numel() - num_decode_tokens);
    const int64_t zero_blocks = (zero_count + kThreads - 1) / kThreads;
    const int64_t grid_x_i64 =
        std::max<int64_t>(num_reqs, std::max<int64_t>(valid_blocks, zero_blocks));
    if (grid_x_i64 <= 0)
        return;
    DG_HOST_ASSERT(grid_x_i64 <= std::numeric_limits<unsigned>::max());
    const dim3 grid(static_cast<unsigned>(grid_x_i64), 3);
    build_sparse_swa_metadata_kernel<StartT, SlotT>
        <<<grid, kThreads, 0, stream>>>(
            token_to_req.data_ptr<int32_t>(), is_valid_token.data_ptr<bool>(),
            query_start_loc.data_ptr<StartT>(), slot_mapping.data_ptr<SlotT>(),
            decode_swa_lens.data_ptr<int32_t>(), static_cast<int>(num_reqs),
            static_cast<int>(token_to_req.numel()), slot_mapping.numel(),
            static_cast<int>(decode_swa_lens.numel()),
            static_cast<int>(num_decode_tokens));
    DG_CUDA_RUNTIME_CHECK(cudaGetLastError());
}

template <typename StartT, typename SlotT, typename SeqT, typename BlockT, typename OutT>
void launch_build_sparse_swa_decode_metadata(
    const torch::Tensor& token_to_req,
    const torch::Tensor& is_valid_token,
    const torch::Tensor& query_start_loc,
    const torch::Tensor& slot_mapping,
    const torch::Tensor& decode_swa_lens,
    const torch::Tensor& decode_swa_indices,
    const torch::Tensor& seq_lens,
    const torch::Tensor& block_table,
    int64_t num_reqs,
    int64_t num_decode_tokens,
    int64_t window_size,
    int64_t block_size,
    cudaStream_t stream) {
    constexpr int kThreads = 256;
    const int64_t valid_blocks = (slot_mapping.numel() + kThreads - 1) / kThreads;
    const int64_t zero_count =
        std::max<int64_t>(0, decode_swa_lens.numel() - num_decode_tokens);
    const int64_t zero_blocks = (zero_count + kThreads - 1) / kThreads;
    const int64_t grid_x_i64 = std::max<int64_t>(
        std::max<int64_t>(num_reqs, valid_blocks),
        std::max<int64_t>(zero_blocks, num_decode_tokens));
    if (grid_x_i64 <= 0)
        return;
    DG_HOST_ASSERT(grid_x_i64 <= std::numeric_limits<unsigned>::max());
    const dim3 grid(static_cast<unsigned>(grid_x_i64), 4);
    build_sparse_swa_decode_metadata_kernel<StartT, SlotT, SeqT, BlockT, OutT>
        <<<grid, kThreads, 0, stream>>>(
            token_to_req.data_ptr<int32_t>(), is_valid_token.data_ptr<bool>(),
            query_start_loc.data_ptr<StartT>(), slot_mapping.data_ptr<SlotT>(),
            decode_swa_lens.data_ptr<int32_t>(),
            decode_swa_indices.data_ptr<OutT>(), decode_swa_indices.stride(0),
            seq_lens.data_ptr<SeqT>(), block_table.data_ptr<BlockT>(),
            block_table.stride(0), static_cast<int>(num_reqs),
            static_cast<int>(token_to_req.numel()), slot_mapping.numel(),
            static_cast<int>(decode_swa_lens.numel()),
            static_cast<int>(num_decode_tokens), static_cast<int>(window_size),
            static_cast<int>(block_size));
    DG_CUDA_RUNTIME_CHECK(cudaGetLastError());
}

template <typename SeqT, typename StartT>
void launch_build_sparse_swa_prefill_metadata(
    const torch::Tensor& prefill_gather_lens,
    const torch::Tensor& seq_lens,
    const torch::Tensor& query_start_loc,
    int64_t num_prefills,
    int64_t num_decodes,
    int64_t window_size,
    cudaStream_t stream) {
    constexpr int kThreads = 256;
    if (num_prefills <= 0)
        return;
    const int64_t blocks = (num_prefills + kThreads - 1) / kThreads;
    DG_HOST_ASSERT(blocks <= std::numeric_limits<unsigned>::max());
    build_sparse_swa_prefill_metadata_kernel<SeqT, StartT>
        <<<static_cast<unsigned>(blocks), kThreads, 0, stream>>>(
            prefill_gather_lens.data_ptr<int32_t>(), seq_lens.data_ptr<SeqT>(),
            query_start_loc.data_ptr<StartT>(), static_cast<int>(num_prefills),
            static_cast<int>(num_decodes), static_cast<int>(window_size));
    DG_CUDA_RUNTIME_CHECK(cudaGetLastError());
}

}  // namespace

void sm120_fill_token_to_req_indices(const torch::Tensor& out,
                                     const torch::Tensor& query_start_loc,
                                     int64_t num_reqs) {
    DG_HOST_ASSERT(out.is_cuda());
    DG_HOST_ASSERT(query_start_loc.is_cuda());
    DG_HOST_ASSERT(out.scalar_type() == torch::kInt32);
    DG_HOST_ASSERT(query_start_loc.dim() == 1);
    DG_HOST_ASSERT(query_start_loc.numel() >= num_reqs + 1);
    DG_HOST_ASSERT(num_reqs >= 0);
    DG_HOST_ASSERT(out.numel() <= std::numeric_limits<int>::max());

    const at::cuda::OptionalCUDAGuard device_guard(device_of(out));
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    if (query_start_loc.scalar_type() == torch::kInt32) {
        launch_fill_token_to_req_indices<int32_t>(out, query_start_loc, num_reqs,
                                                  stream);
    } else {
        DG_HOST_ASSERT(query_start_loc.scalar_type() == torch::kInt64);
        launch_fill_token_to_req_indices<int64_t>(out, query_start_loc, num_reqs,
                                                  stream);
    }
}

void sm120_build_compressor_metadata(const torch::Tensor& token_to_req,
                                     const torch::Tensor& query_start_loc,
                                     const torch::Tensor& block_table,
                                     int64_t num_reqs) {
    DG_HOST_ASSERT(token_to_req.is_cuda());
    DG_HOST_ASSERT(query_start_loc.is_cuda());
    DG_HOST_ASSERT(block_table.is_cuda());
    DG_HOST_ASSERT(token_to_req.scalar_type() == torch::kInt32);
    DG_HOST_ASSERT(query_start_loc.dim() == 1);
    DG_HOST_ASSERT(query_start_loc.numel() >= num_reqs + 1);
    DG_HOST_ASSERT(num_reqs >= 0);
    DG_HOST_ASSERT(num_reqs <= std::numeric_limits<int>::max());
    DG_HOST_ASSERT(token_to_req.numel() <= std::numeric_limits<int>::max());
    DG_HOST_ASSERT(block_table.scalar_type() == torch::kInt32 ||
                   block_table.scalar_type() == torch::kInt64);

    const at::cuda::OptionalCUDAGuard device_guard(device_of(token_to_req));
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

#define DISPATCH_START(BLOCK_T)                                                    \
    do {                                                                           \
        if (query_start_loc.scalar_type() == torch::kInt32) {                      \
            launch_build_compressor_metadata<int32_t, BLOCK_T>(                    \
                token_to_req, query_start_loc, block_table, num_reqs, stream);     \
        } else {                                                                   \
            DG_HOST_ASSERT(query_start_loc.scalar_type() == torch::kInt64);        \
            launch_build_compressor_metadata<int64_t, BLOCK_T>(                    \
                token_to_req, query_start_loc, block_table, num_reqs, stream);     \
        }                                                                          \
    } while (0)

    if (block_table.scalar_type() == torch::kInt32) {
        DISPATCH_START(int32_t);
    } else {
        DISPATCH_START(int64_t);
    }

#undef DISPATCH_START
}

void sm120_build_sparse_swa_metadata(const torch::Tensor& token_to_req,
                                     const torch::Tensor& is_valid_token,
                                     const torch::Tensor& query_start_loc,
                                     const torch::Tensor& slot_mapping,
                                     const torch::Tensor& decode_swa_lens,
                                     int64_t num_reqs,
                                     int64_t num_decode_tokens) {
    DG_HOST_ASSERT(token_to_req.is_cuda());
    DG_HOST_ASSERT(is_valid_token.is_cuda());
    DG_HOST_ASSERT(query_start_loc.is_cuda());
    DG_HOST_ASSERT(slot_mapping.is_cuda());
    DG_HOST_ASSERT(decode_swa_lens.is_cuda());
    DG_HOST_ASSERT(token_to_req.scalar_type() == torch::kInt32);
    DG_HOST_ASSERT(is_valid_token.scalar_type() == torch::kBool);
    DG_HOST_ASSERT(decode_swa_lens.scalar_type() == torch::kInt32);
    DG_HOST_ASSERT(query_start_loc.dim() == 1);
    DG_HOST_ASSERT(query_start_loc.numel() >= num_reqs + 1);
    DG_HOST_ASSERT(slot_mapping.dim() == 1);
    DG_HOST_ASSERT(is_valid_token.dim() == 1);
    DG_HOST_ASSERT(is_valid_token.numel() >= slot_mapping.numel());
    DG_HOST_ASSERT(num_reqs >= 0);
    DG_HOST_ASSERT(num_reqs <= std::numeric_limits<int>::max());
    DG_HOST_ASSERT(num_decode_tokens >= 0);
    DG_HOST_ASSERT(num_decode_tokens <= decode_swa_lens.numel());
    DG_HOST_ASSERT(token_to_req.numel() <= std::numeric_limits<int>::max());
    DG_HOST_ASSERT(decode_swa_lens.numel() <= std::numeric_limits<int>::max());
    DG_HOST_ASSERT(slot_mapping.scalar_type() == torch::kInt32 ||
                   slot_mapping.scalar_type() == torch::kInt64);

    const at::cuda::OptionalCUDAGuard device_guard(device_of(token_to_req));
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

#define DISPATCH_START(SLOT_T)                                                     \
    do {                                                                           \
        if (query_start_loc.scalar_type() == torch::kInt32) {                      \
            launch_build_sparse_swa_metadata<int32_t, SLOT_T>(                     \
                token_to_req, is_valid_token, query_start_loc, slot_mapping,       \
                decode_swa_lens, num_reqs, num_decode_tokens, stream);            \
        } else {                                                                   \
            DG_HOST_ASSERT(query_start_loc.scalar_type() == torch::kInt64);        \
            launch_build_sparse_swa_metadata<int64_t, SLOT_T>(                     \
                token_to_req, is_valid_token, query_start_loc, slot_mapping,       \
                decode_swa_lens, num_reqs, num_decode_tokens, stream);            \
        }                                                                          \
    } while (0)

    if (slot_mapping.scalar_type() == torch::kInt32) {
        DISPATCH_START(int32_t);
    } else {
        DISPATCH_START(int64_t);
    }

#undef DISPATCH_START
}

void sm120_build_sparse_swa_decode_metadata(
    const torch::Tensor& token_to_req,
    const torch::Tensor& is_valid_token,
    const torch::Tensor& query_start_loc,
    const torch::Tensor& slot_mapping,
    const torch::Tensor& decode_swa_lens,
    const torch::Tensor& decode_swa_indices,
    const torch::Tensor& seq_lens,
    const torch::Tensor& block_table,
    int64_t num_reqs,
    int64_t num_decode_tokens,
    int64_t window_size,
    int64_t block_size) {
    DG_HOST_ASSERT(token_to_req.is_cuda());
    DG_HOST_ASSERT(is_valid_token.is_cuda());
    DG_HOST_ASSERT(query_start_loc.is_cuda());
    DG_HOST_ASSERT(slot_mapping.is_cuda());
    DG_HOST_ASSERT(decode_swa_lens.is_cuda());
    DG_HOST_ASSERT(decode_swa_indices.is_cuda());
    DG_HOST_ASSERT(seq_lens.is_cuda());
    DG_HOST_ASSERT(block_table.is_cuda());
    DG_HOST_ASSERT(token_to_req.scalar_type() == torch::kInt32);
    DG_HOST_ASSERT(is_valid_token.scalar_type() == torch::kBool);
    DG_HOST_ASSERT(decode_swa_lens.scalar_type() == torch::kInt32);
    DG_HOST_ASSERT(decode_swa_indices.scalar_type() == torch::kInt32 ||
                   decode_swa_indices.scalar_type() == torch::kInt64);
    DG_HOST_ASSERT(query_start_loc.dim() == 1);
    DG_HOST_ASSERT(query_start_loc.numel() >= num_reqs + 1);
    DG_HOST_ASSERT(slot_mapping.dim() == 1);
    DG_HOST_ASSERT(is_valid_token.dim() == 1);
    DG_HOST_ASSERT(is_valid_token.numel() >= slot_mapping.numel());
    DG_HOST_ASSERT(decode_swa_indices.dim() == 2 ||
                   decode_swa_indices.dim() == 3);
    DG_HOST_ASSERT(decode_swa_indices.size(decode_swa_indices.dim() - 1) >=
                   window_size);
    DG_HOST_ASSERT(decode_swa_indices.size(0) >= num_decode_tokens);
    DG_HOST_ASSERT(seq_lens.dim() == 1);
    DG_HOST_ASSERT(seq_lens.numel() >= num_reqs);
    DG_HOST_ASSERT(block_table.dim() == 2);
    DG_HOST_ASSERT(block_table.size(0) >= num_reqs);
    DG_HOST_ASSERT(num_reqs >= 0);
    DG_HOST_ASSERT(num_reqs <= std::numeric_limits<int>::max());
    DG_HOST_ASSERT(num_decode_tokens >= 0);
    DG_HOST_ASSERT(num_decode_tokens <= decode_swa_lens.numel());
    DG_HOST_ASSERT(token_to_req.numel() <= std::numeric_limits<int>::max());
    DG_HOST_ASSERT(decode_swa_lens.numel() <= std::numeric_limits<int>::max());
    DG_HOST_ASSERT(window_size > 0);
    DG_HOST_ASSERT(window_size <=
                   decode_swa_indices.size(decode_swa_indices.dim() - 1));
    DG_HOST_ASSERT(window_size <= std::numeric_limits<int>::max());
    DG_HOST_ASSERT(block_size > 0);
    DG_HOST_ASSERT(block_size <= std::numeric_limits<int>::max());
    DG_HOST_ASSERT(query_start_loc.scalar_type() == torch::kInt32 ||
                   query_start_loc.scalar_type() == torch::kInt64);
    DG_HOST_ASSERT(slot_mapping.scalar_type() == torch::kInt32 ||
                   slot_mapping.scalar_type() == torch::kInt64);
    DG_HOST_ASSERT(seq_lens.scalar_type() == torch::kInt32 ||
                   seq_lens.scalar_type() == torch::kInt64);
    DG_HOST_ASSERT(block_table.scalar_type() == torch::kInt32 ||
                   block_table.scalar_type() == torch::kInt64);

    const at::cuda::OptionalCUDAGuard device_guard(device_of(token_to_req));
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

#define DISPATCH_OUT(START_T, SLOT_T, SEQ_T, BLOCK_T)                              \
    do {                                                                           \
        if (decode_swa_indices.scalar_type() == torch::kInt32) {                   \
            launch_build_sparse_swa_decode_metadata<START_T, SLOT_T, SEQ_T,        \
                                                    BLOCK_T, int32_t>(             \
                token_to_req, is_valid_token, query_start_loc, slot_mapping,       \
                decode_swa_lens, decode_swa_indices, seq_lens, block_table,        \
                num_reqs, num_decode_tokens, window_size, block_size, stream);     \
        } else {                                                                   \
            launch_build_sparse_swa_decode_metadata<START_T, SLOT_T, SEQ_T,        \
                                                    BLOCK_T, int64_t>(             \
                token_to_req, is_valid_token, query_start_loc, slot_mapping,       \
                decode_swa_lens, decode_swa_indices, seq_lens, block_table,        \
                num_reqs, num_decode_tokens, window_size, block_size, stream);     \
        }                                                                          \
    } while (0)

#define DISPATCH_BLOCK(START_T, SLOT_T, SEQ_T)                                      \
    do {                                                                           \
        if (block_table.scalar_type() == torch::kInt32) {                          \
            DISPATCH_OUT(START_T, SLOT_T, SEQ_T, int32_t);                         \
        } else {                                                                   \
            DISPATCH_OUT(START_T, SLOT_T, SEQ_T, int64_t);                         \
        }                                                                          \
    } while (0)

#define DISPATCH_SEQ(START_T, SLOT_T)                                               \
    do {                                                                           \
        if (seq_lens.scalar_type() == torch::kInt32) {                             \
            DISPATCH_BLOCK(START_T, SLOT_T, int32_t);                              \
        } else {                                                                   \
            DISPATCH_BLOCK(START_T, SLOT_T, int64_t);                              \
        }                                                                          \
    } while (0)

#define DISPATCH_SLOT(START_T)                                                      \
    do {                                                                           \
        if (slot_mapping.scalar_type() == torch::kInt32) {                         \
            DISPATCH_SEQ(START_T, int32_t);                                        \
        } else {                                                                   \
            DISPATCH_SEQ(START_T, int64_t);                                        \
        }                                                                          \
    } while (0)

    if (query_start_loc.scalar_type() == torch::kInt32) {
        DISPATCH_SLOT(int32_t);
    } else {
        DISPATCH_SLOT(int64_t);
    }

#undef DISPATCH_SLOT
#undef DISPATCH_SEQ
#undef DISPATCH_BLOCK
#undef DISPATCH_OUT
}

void sm120_build_sparse_swa_prefill_metadata(
    const torch::Tensor& prefill_gather_lens,
    const torch::Tensor& seq_lens,
    const torch::Tensor& query_start_loc,
    int64_t num_prefills,
    int64_t num_decodes,
    int64_t window_size) {
    DG_HOST_ASSERT(prefill_gather_lens.is_cuda());
    DG_HOST_ASSERT(seq_lens.is_cuda());
    DG_HOST_ASSERT(query_start_loc.is_cuda());
    DG_HOST_ASSERT(prefill_gather_lens.scalar_type() == torch::kInt32);
    DG_HOST_ASSERT(seq_lens.scalar_type() == torch::kInt32 ||
                   seq_lens.scalar_type() == torch::kInt64);
    DG_HOST_ASSERT(query_start_loc.scalar_type() == torch::kInt32 ||
                   query_start_loc.scalar_type() == torch::kInt64);
    DG_HOST_ASSERT(prefill_gather_lens.dim() == 1);
    DG_HOST_ASSERT(seq_lens.dim() == 1);
    DG_HOST_ASSERT(query_start_loc.dim() == 1);
    DG_HOST_ASSERT(num_prefills >= 0);
    DG_HOST_ASSERT(num_decodes >= 0);
    DG_HOST_ASSERT(num_prefills <= std::numeric_limits<int>::max());
    DG_HOST_ASSERT(num_decodes <= std::numeric_limits<int>::max());
    DG_HOST_ASSERT(window_size > 0);
    DG_HOST_ASSERT(window_size <= std::numeric_limits<int>::max());
    DG_HOST_ASSERT(prefill_gather_lens.numel() >= num_prefills);
    DG_HOST_ASSERT(seq_lens.numel() >= num_decodes + num_prefills);
    DG_HOST_ASSERT(query_start_loc.numel() >= num_decodes + num_prefills + 1);

    const at::cuda::OptionalCUDAGuard device_guard(device_of(prefill_gather_lens));
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

#define DISPATCH_START(SEQ_T)                                                      \
    do {                                                                           \
        if (query_start_loc.scalar_type() == torch::kInt32) {                      \
            launch_build_sparse_swa_prefill_metadata<SEQ_T, int32_t>(              \
                prefill_gather_lens, seq_lens, query_start_loc, num_prefills,      \
                num_decodes, window_size, stream);                                \
        } else {                                                                   \
            launch_build_sparse_swa_prefill_metadata<SEQ_T, int64_t>(              \
                prefill_gather_lens, seq_lens, query_start_loc, num_prefills,      \
                num_decodes, window_size, stream);                                \
        }                                                                          \
    } while (0)

    if (seq_lens.scalar_type() == torch::kInt32) {
        DISPATCH_START(int32_t);
    } else {
        DISPATCH_START(int64_t);
    }

#undef DISPATCH_START
}

void register_apis(pybind11::module& m) {
    m.def("sm120_fill_token_to_req_indices",
          &sm120_fill_token_to_req_indices,
          pybind11::arg("out"), pybind11::arg("query_start_loc"),
          pybind11::arg("num_reqs"));
    m.def("sm120_build_compressor_metadata",
          &sm120_build_compressor_metadata,
          pybind11::arg("token_to_req"), pybind11::arg("query_start_loc"),
          pybind11::arg("block_table"), pybind11::arg("num_reqs"));
    m.def("sm120_build_sparse_swa_metadata",
          &sm120_build_sparse_swa_metadata,
          pybind11::arg("token_to_req"), pybind11::arg("is_valid_token"),
          pybind11::arg("query_start_loc"), pybind11::arg("slot_mapping"),
          pybind11::arg("decode_swa_lens"), pybind11::arg("num_reqs"),
          pybind11::arg("num_decode_tokens"));
    m.def("sm120_build_sparse_swa_decode_metadata",
          &sm120_build_sparse_swa_decode_metadata,
          pybind11::arg("token_to_req"), pybind11::arg("is_valid_token"),
          pybind11::arg("query_start_loc"), pybind11::arg("slot_mapping"),
          pybind11::arg("decode_swa_lens"), pybind11::arg("decode_swa_indices"),
          pybind11::arg("seq_lens"), pybind11::arg("block_table"),
          pybind11::arg("num_reqs"), pybind11::arg("num_decode_tokens"),
          pybind11::arg("window_size"), pybind11::arg("block_size"));
    m.def("sm120_build_sparse_swa_prefill_metadata",
          &sm120_build_sparse_swa_prefill_metadata,
          pybind11::arg("prefill_gather_lens"),
          pybind11::arg("seq_lens"), pybind11::arg("query_start_loc"),
          pybind11::arg("num_prefills"), pybind11::arg("num_decodes"),
          pybind11::arg("window_size"));
}

}  // namespace deep_gemm::sm120_metadata
