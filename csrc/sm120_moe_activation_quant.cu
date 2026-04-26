#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <torch/python.h>

#include "jit_kernels/impls/sm120_moe_activation_quant.hpp"
#include "utils/exception.hpp"

namespace deep_gemm::sm120_moe {
namespace {

constexpr float kFp8E4M3Max = 448.0f;
constexpr float kFp8E4M3Min = -448.0f;
constexpr float kScaleEps = 1.0e-10f;

__device__ __forceinline__ float silu(float x) {
    return x / (1.0f + expf(-x));
}

__device__ __forceinline__ float warp_reduce_max(float value) {
    value = fmaxf(value, __shfl_xor_sync(0xffffffffu, value, 16));
    value = fmaxf(value, __shfl_xor_sync(0xffffffffu, value, 8));
    value = fmaxf(value, __shfl_xor_sync(0xffffffffu, value, 4));
    value = fmaxf(value, __shfl_xor_sync(0xffffffffu, value, 2));
    value = fmaxf(value, __shfl_xor_sync(0xffffffffu, value, 1));
    return value;
}

template <typename T>
__device__ __forceinline__ float weight_to_float(T value) {
    return static_cast<float>(value);
}

template <>
__device__ __forceinline__ float weight_to_float<__nv_bfloat16>(
    __nv_bfloat16 value) {
    return __bfloat162float(value);
}

__global__ void silu_mul_quant_fp8_packed_kernel(
    const __nv_bfloat16* __restrict__ input,
    __nv_fp8_e4m3* __restrict__ output,
    uint32_t* __restrict__ output_scales,
    int group_size,
    int num_groups_padded,
    int groups_per_block,
    int padded_groups_per_row,
    int groups_per_row,
    int mn,
    int hidden,
    int tma_aligned_mn,
    int num_scale_elems,
    float clamp_limit,
    bool has_clamp) {
    extern __shared__ float smem[];

    const int local_group = threadIdx.y;
    const int local_tid = threadIdx.x;
    const int global_group =
        static_cast<int>(blockIdx.x) * groups_per_block + local_group;
    if (global_group >= num_groups_padded)
        return;

    const int sf_k_idx = global_group % padded_groups_per_row;
    const int mn_idx = global_group / padded_groups_per_row;
    const bool valid = mn_idx < mn && sf_k_idx < groups_per_row;

    float* values = smem + static_cast<size_t>(local_group) * group_size;
    float local_absmax = kScaleEps;

    if (valid) {
        const int col_base = sf_k_idx * group_size;
        const __nv_bfloat16* gate =
            input + static_cast<int64_t>(mn_idx) * 2 * hidden + col_base;
        const __nv_bfloat16* up = gate + hidden;
        for (int i = local_tid; i < group_size; i += blockDim.x) {
            float gate_v = __bfloat162float(gate[i]);
            float up_v = __bfloat162float(up[i]);
            if (has_clamp) {
                gate_v = fminf(gate_v, clamp_limit);
                up_v = fminf(fmaxf(up_v, -clamp_limit), clamp_limit);
            }
            const float value = __bfloat162float(
                __float2bfloat16_rn(silu(gate_v) * up_v));
            values[i] = value;
            local_absmax = fmaxf(local_absmax, fabsf(value));
        }
    }

    float absmax = warp_reduce_max(local_absmax);
    __shared__ float scale_slots[16];
    if (local_tid == 0) {
        float scale = absmax / kFp8E4M3Max;
        scale = exp2f(ceilf(log2f(fmaxf(fabsf(scale), kScaleEps))));
        scale_slots[local_group] = scale;

        const int pack = sf_k_idx >> 2;
        const int pos = sf_k_idx & 3;
        const int out_idx = pack * tma_aligned_mn + mn_idx;
        if (valid) {
            const uint32_t bits = __float_as_uint(scale);
            const uint8_t exponent = static_cast<uint8_t>((bits >> 23u) & 0xffu);
            reinterpret_cast<uint8_t*>(output_scales)[out_idx * 4 + pos] = exponent;
        } else if (out_idx < num_scale_elems) {
            reinterpret_cast<uint8_t*>(output_scales)[out_idx * 4 + pos] = 0;
        }
    }
    __syncthreads();

    if (!valid)
        return;

    const float inv_scale = 1.0f / scale_slots[local_group];
    __nv_fp8_e4m3* out =
        output + static_cast<int64_t>(mn_idx) * hidden + sf_k_idx * group_size;
    for (int i = local_tid; i < group_size; i += blockDim.x) {
        const float q = fminf(fmaxf(values[i] * inv_scale, kFp8E4M3Min), kFp8E4M3Max);
        out[i] = __nv_fp8_e4m3(q);
    }
}

template <typename TopkIdT, typename WeightT, bool kHasExpertMap, int kHiddenChunk>
__global__ void moe_unpermute_reduce_bf16_kernel(
    const __nv_bfloat16* __restrict__ input,
    const TopkIdT* __restrict__ topk_ids,
    const WeightT* __restrict__ topk_weights,
    const int32_t* __restrict__ input_index,
    const int32_t* __restrict__ expert_map,
    __nv_bfloat16* __restrict__ output,
    int num_tokens,
    int hidden,
    int topk,
    int expert_map_size,
    int64_t input_stride0,
    int64_t topk_ids_stride0,
    int64_t topk_ids_stride1,
    int64_t topk_weights_stride0,
    int64_t topk_weights_stride1,
    int64_t input_index_stride0,
    int64_t input_index_stride1,
    int64_t output_stride0) {
    constexpr int kMaxTopK = 32;
    __shared__ int32_t source_slots[kMaxTopK];
    __shared__ float weight_slots[kMaxTopK];

    const int token = static_cast<int>(blockIdx.y);
    const int col_base = static_cast<int>(blockIdx.x) * kHiddenChunk;

    if (threadIdx.x < kMaxTopK) {
        const int k = static_cast<int>(threadIdx.x);
        int32_t source = -1;
        float weight = 0.0f;
        if (k < topk) {
            const int64_t global_expert_id = static_cast<int64_t>(
                topk_ids[token * topk_ids_stride0 + k * topk_ids_stride1]);
            int64_t expert_id = global_expert_id;
            if constexpr (kHasExpertMap) {
                if (global_expert_id >= 0 && global_expert_id < expert_map_size) {
                    expert_id = static_cast<int64_t>(expert_map[global_expert_id]);
                } else {
                    expert_id = -1;
                }
            }
            if (expert_id >= 0) {
                source = input_index[token * input_index_stride0 +
                                     k * input_index_stride1];
                if (source >= 0) {
                    weight = weight_to_float<WeightT>(
                        topk_weights[token * topk_weights_stride0 +
                                     k * topk_weights_stride1]);
                }
            }
        }
        source_slots[k] = source;
        weight_slots[k] = weight;
    }
    __syncthreads();

    for (int offset = threadIdx.x; offset < kHiddenChunk; offset += blockDim.x) {
        const int col = col_base + offset;
        if (token >= num_tokens || col >= hidden)
            continue;

        float acc = 0.0f;
#pragma unroll 8
        for (int k = 0; k < kMaxTopK; ++k) {
            if (k >= topk)
                break;
            const int32_t source = source_slots[k];
            if (source >= 0) {
                acc += __bfloat162float(input[source * input_stride0 + col]) *
                       weight_slots[k];
            }
        }
        output[token * output_stride0 + col] = __float2bfloat16(acc);
    }
}

int groups_per_block_for(int64_t num_groups) {
    if (num_groups % 16 == 0)
        return 16;
    if (num_groups % 8 == 0)
        return 8;
    if (num_groups % 4 == 0)
        return 4;
    if (num_groups % 2 == 0)
        return 2;
    return 1;
}

}  // namespace

torch::Tensor sm120_silu_mul_quant_fp8_packed(const torch::Tensor& input,
                                              const torch::Tensor& output,
                                              int64_t group_size,
                                              double clamp_limit) {
    DG_HOST_ASSERT(input.is_cuda());
    DG_HOST_ASSERT(output.is_cuda());
    DG_HOST_ASSERT(input.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(output.scalar_type() == torch::kFloat8_e4m3fn);
    DG_HOST_ASSERT(input.dim() == 2);
    DG_HOST_ASSERT(output.dim() == 2);
    DG_HOST_ASSERT(input.is_contiguous());
    DG_HOST_ASSERT(output.is_contiguous());
    DG_HOST_ASSERT(input.size(1) % 2 == 0);
    const int64_t mn = input.size(0);
    const int64_t hidden = input.size(1) / 2;
    DG_HOST_ASSERT(output.size(0) == mn and output.size(1) == hidden);
    DG_HOST_ASSERT(group_size > 0 and hidden % group_size == 0);

    const int64_t groups_per_row = hidden / group_size;
    const int64_t packed_groups_per_row = (groups_per_row + 3) / 4;
    const int64_t padded_groups_per_row = packed_groups_per_row * 4;
    const int64_t tma_aligned_mn = ((mn + 3) / 4) * 4;

    auto scales = torch::empty_strided(
        {mn, packed_groups_per_row},
        {1, tma_aligned_mn},
        torch::TensorOptions().device(input.device()).dtype(torch::kInt32));

    const int64_t num_groups_padded = tma_aligned_mn * padded_groups_per_row;
    const int64_t num_scale_elems =
        mn + (packed_groups_per_row - 1) * tma_aligned_mn;
    if (num_groups_padded == 0)
        return scales;

    const int groups_per_block = groups_per_block_for(num_groups_padded);
    const dim3 block(32, groups_per_block);
    const dim3 grid(num_groups_padded / groups_per_block);
    const size_t smem_bytes =
        static_cast<size_t>(groups_per_block) * group_size * sizeof(float);

    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    silu_mul_quant_fp8_packed_kernel<<<grid, block, smem_bytes, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
        reinterpret_cast<__nv_fp8_e4m3*>(output.data_ptr()),
        reinterpret_cast<uint32_t*>(scales.data_ptr()),
        static_cast<int>(group_size),
        static_cast<int>(num_groups_padded),
        groups_per_block,
        static_cast<int>(padded_groups_per_row),
        static_cast<int>(groups_per_row),
        static_cast<int>(mn),
        static_cast<int>(hidden),
        static_cast<int>(tma_aligned_mn),
        static_cast<int>(num_scale_elems),
        static_cast<float>(clamp_limit),
        clamp_limit > 0.0);
    return scales;
}

template <typename TopkIdT, typename WeightT, bool kHasExpertMap>
void launch_moe_unpermute_reduce_bf16(const torch::Tensor& input,
                                      const torch::Tensor& topk_ids,
                                      const torch::Tensor& topk_weights,
                                      const torch::Tensor& input_index,
                                      const torch::Tensor* expert_map,
                                      const torch::Tensor& output,
                                      cudaStream_t stream) {
    constexpr int kHiddenChunk = 1024;
    constexpr int kThreads = 256;
    const int num_tokens = static_cast<int>(output.size(0));
    const int hidden = static_cast<int>(output.size(1));
    const int topk = static_cast<int>(topk_ids.size(1));
    const int expert_map_size = (kHasExpertMap && expert_map != nullptr)
                                    ? static_cast<int>(expert_map->numel())
                                    : 0;
    const dim3 block(kThreads);
    const dim3 grid((hidden + kHiddenChunk - 1) / kHiddenChunk, num_tokens);
    moe_unpermute_reduce_bf16_kernel<TopkIdT, WeightT, kHasExpertMap, kHiddenChunk>
        <<<grid, block, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
            reinterpret_cast<const TopkIdT*>(topk_ids.data_ptr()),
            reinterpret_cast<const WeightT*>(topk_weights.data_ptr()),
            reinterpret_cast<const int32_t*>(input_index.data_ptr()),
            kHasExpertMap
                ? reinterpret_cast<const int32_t*>(expert_map->data_ptr())
                : nullptr,
            reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
            num_tokens,
            hidden,
            topk,
            expert_map_size,
            input.stride(0),
            topk_ids.stride(0),
            topk_ids.stride(1),
            topk_weights.stride(0),
            topk_weights.stride(1),
            input_index.stride(0),
            input_index.stride(1),
            output.stride(0));
}

void validate_moe_unpermute_reduce_bf16(const torch::Tensor& input,
                                        const torch::Tensor& topk_ids,
                                        const torch::Tensor& topk_weights,
                                        const torch::Tensor& input_index,
                                        const torch::Tensor& output) {
    DG_HOST_ASSERT(input.is_cuda());
    DG_HOST_ASSERT(topk_ids.is_cuda());
    DG_HOST_ASSERT(topk_weights.is_cuda());
    DG_HOST_ASSERT(input_index.is_cuda());
    DG_HOST_ASSERT(output.is_cuda());
    DG_HOST_ASSERT(input.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(output.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(input_index.scalar_type() == torch::kInt32);
    DG_HOST_ASSERT(topk_ids.scalar_type() == torch::kInt32 ||
                   topk_ids.scalar_type() == torch::kInt64);
    DG_HOST_ASSERT(topk_weights.scalar_type() == torch::kFloat32 ||
                   topk_weights.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(input.dim() == 2);
    DG_HOST_ASSERT(output.dim() == 2);
    DG_HOST_ASSERT(topk_ids.dim() == 2);
    DG_HOST_ASSERT(topk_weights.dim() == 2);
    DG_HOST_ASSERT(input_index.dim() == 2);
    DG_HOST_ASSERT(output.size(1) == input.size(1));
    DG_HOST_ASSERT(topk_ids.sizes() == topk_weights.sizes());
    DG_HOST_ASSERT(topk_ids.sizes() == input_index.sizes());
    DG_HOST_ASSERT(topk_ids.size(0) == output.size(0));
    DG_HOST_ASSERT(topk_ids.size(1) > 0 && topk_ids.size(1) <= 32);
    DG_HOST_ASSERT(input.stride(1) == 1);
    DG_HOST_ASSERT(output.stride(1) == 1);
}

void sm120_moe_unpermute_reduce_bf16(const torch::Tensor& input,
                                     const torch::Tensor& topk_ids,
                                     const torch::Tensor& topk_weights,
                                     const torch::Tensor& input_index,
                                     const torch::Tensor& output) {
    validate_moe_unpermute_reduce_bf16(input, topk_ids, topk_weights,
                                       input_index, output);
    if (output.numel() == 0)
        return;

    const at::cuda::OptionalCUDAGuard device_guard(device_of(output));
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

#define DISPATCH_WEIGHT(TOPK_T)                                                     \
    do {                                                                            \
        if (topk_weights.scalar_type() == torch::kFloat32) {                        \
            launch_moe_unpermute_reduce_bf16<TOPK_T, float, false>(                 \
                input, topk_ids, topk_weights, input_index, nullptr, output,        \
                stream);                                                            \
        } else {                                                                    \
            launch_moe_unpermute_reduce_bf16<TOPK_T, __nv_bfloat16, false>(         \
                input, topk_ids, topk_weights, input_index, nullptr, output,        \
                stream);                                                            \
        }                                                                           \
    } while (0)

    if (topk_ids.scalar_type() == torch::kInt32) {
        DISPATCH_WEIGHT(int32_t);
    } else {
        DISPATCH_WEIGHT(int64_t);
    }

#undef DISPATCH_WEIGHT
}

void sm120_moe_unpermute_reduce_bf16_mapped(const torch::Tensor& input,
                                            const torch::Tensor& topk_ids,
                                            const torch::Tensor& topk_weights,
                                            const torch::Tensor& input_index,
                                            const torch::Tensor& expert_map,
                                            const torch::Tensor& output) {
    validate_moe_unpermute_reduce_bf16(input, topk_ids, topk_weights,
                                       input_index, output);
    DG_HOST_ASSERT(expert_map.is_cuda());
    DG_HOST_ASSERT(expert_map.scalar_type() == torch::kInt32);
    if (output.numel() == 0)
        return;

    const at::cuda::OptionalCUDAGuard device_guard(device_of(output));
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

#define DISPATCH_WEIGHT_MAPPED(TOPK_T)                                              \
    do {                                                                            \
        if (topk_weights.scalar_type() == torch::kFloat32) {                        \
            launch_moe_unpermute_reduce_bf16<TOPK_T, float, true>(                  \
                input, topk_ids, topk_weights, input_index, &expert_map, output,    \
                stream);                                                            \
        } else {                                                                    \
            launch_moe_unpermute_reduce_bf16<TOPK_T, __nv_bfloat16, true>(          \
                input, topk_ids, topk_weights, input_index, &expert_map, output,    \
                stream);                                                            \
        }                                                                           \
    } while (0)

    if (topk_ids.scalar_type() == torch::kInt32) {
        DISPATCH_WEIGHT_MAPPED(int32_t);
    } else {
        DISPATCH_WEIGHT_MAPPED(int64_t);
    }

#undef DISPATCH_WEIGHT_MAPPED
}

void register_apis(pybind11::module& m) {
    m.def("sm120_silu_mul_quant_fp8_packed",
          &sm120_silu_mul_quant_fp8_packed,
          pybind11::arg("input"), pybind11::arg("output"),
          pybind11::arg("group_size"),
          pybind11::arg("clamp_limit") = 0.0);
    m.def("sm120_moe_unpermute_reduce_bf16",
          &sm120_moe_unpermute_reduce_bf16,
          pybind11::arg("input"), pybind11::arg("topk_ids"),
          pybind11::arg("topk_weights"), pybind11::arg("input_index"),
          pybind11::arg("output"));
    m.def("sm120_moe_unpermute_reduce_bf16_mapped",
          &sm120_moe_unpermute_reduce_bf16_mapped,
          pybind11::arg("input"), pybind11::arg("topk_ids"),
          pybind11::arg("topk_weights"), pybind11::arg("input_index"),
          pybind11::arg("expert_map"), pybind11::arg("output"));
}

}  // namespace deep_gemm::sm120_moe
