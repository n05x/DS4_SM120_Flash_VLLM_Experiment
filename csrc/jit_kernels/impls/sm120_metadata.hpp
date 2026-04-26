#pragma once

#include <cstdint>
#include <torch/python.h>
#include <pybind11/pybind11.h>

namespace deep_gemm::sm120_metadata {

void sm120_fill_token_to_req_indices(const torch::Tensor& out,
                                     const torch::Tensor& query_start_loc,
                                     int64_t num_reqs);
void sm120_build_compressor_metadata(const torch::Tensor& token_to_req,
                                     const torch::Tensor& query_start_loc,
                                     const torch::Tensor& block_table,
                                     int64_t num_reqs);
void sm120_build_sparse_swa_metadata(const torch::Tensor& token_to_req,
                                     const torch::Tensor& is_valid_token,
                                     const torch::Tensor& query_start_loc,
                                     const torch::Tensor& slot_mapping,
                                     const torch::Tensor& decode_swa_lens,
                                     int64_t num_reqs,
                                     int64_t num_decode_tokens);
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
    int64_t block_size);
void sm120_build_sparse_swa_prefill_metadata(
    const torch::Tensor& prefill_gather_lens,
    const torch::Tensor& seq_lens,
    const torch::Tensor& query_start_loc,
    int64_t num_prefills,
    int64_t num_decodes,
    int64_t window_size);
void register_apis(pybind11::module& m);

}  // namespace deep_gemm::sm120_metadata
