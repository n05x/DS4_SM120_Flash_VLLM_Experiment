#pragma once

#include <pybind11/pybind11.h>
#include <torch/python.h>

namespace deep_gemm::sm120_moe {

torch::Tensor sm120_silu_mul_quant_fp8_packed(const torch::Tensor& input,
                                              const torch::Tensor& output,
                                              int64_t group_size,
                                              double clamp_limit = 0.0);

void sm120_moe_unpermute_reduce_bf16(const torch::Tensor& input,
                                     const torch::Tensor& topk_ids,
                                     const torch::Tensor& topk_weights,
                                     const torch::Tensor& input_index,
                                     const torch::Tensor& output);

void sm120_moe_unpermute_reduce_bf16_mapped(const torch::Tensor& input,
                                            const torch::Tensor& topk_ids,
                                            const torch::Tensor& topk_weights,
                                            const torch::Tensor& input_index,
                                            const torch::Tensor& expert_map,
                                            const torch::Tensor& output);

void register_apis(pybind11::module& m);

}  // namespace deep_gemm::sm120_moe
