#pragma once

#include <pybind11/pybind11.h>
#include <torch/python.h>

namespace deep_gemm {
namespace sm120_mla {

std::tuple<torch::Tensor, torch::Tensor> sparse_mla_decode(
    const torch::Tensor& q, const torch::Tensor& k_cache,
    const torch::Tensor& indices, const pybind11::object& topk_length,
    const pybind11::object& attn_sink, const pybind11::object& extra_k_cache,
    const pybind11::object& extra_indices_in_kvcache,
    const pybind11::object& extra_topk_length, int head_dim_v,
    double softmax_scale, const pybind11::object& out);

std::tuple<torch::Tensor, torch::Tensor> sparse_mla_decode_fused(
    const torch::Tensor& q, const torch::Tensor& k_cache,
    const torch::Tensor& indices, const pybind11::object& topk_length,
    const pybind11::object& attn_sink, int head_dim_v, double softmax_scale,
    const pybind11::object& out);

std::tuple<torch::Tensor, torch::Tensor> sparse_mla_decode_full_context(
    const torch::Tensor& q, const torch::Tensor& k_cache,
    const torch::Tensor& block_table, const torch::Tensor& seq_lens,
    const torch::Tensor& req_id_per_token,
    const pybind11::object& attn_sink, int head_dim_v, double softmax_scale,
    const pybind11::object& out);

void dequantize_and_gather_k_cache(
    const torch::Tensor& out, const torch::Tensor& k_cache,
    const torch::Tensor& seq_lens, const pybind11::object& gather_lens,
    const torch::Tensor& block_table, int block_size, int offset);

void dequantize_and_gather_indexed_k_cache(
    const torch::Tensor& out, const torch::Tensor& k_cache,
    const torch::Tensor& indices, const pybind11::object& topk_length,
    int block_size, int offset);

std::tuple<torch::Tensor, torch::Tensor> sparse_mla_decode_from_bf16_workspace(
    const torch::Tensor& q, const torch::Tensor& kv_workspace,
    const pybind11::object& topk_length,
    const pybind11::object& extra_topk_length,
    const pybind11::object& attn_sink, int main_topk, int extra_topk,
    int head_dim_v, double softmax_scale, const pybind11::object& out);

std::tuple<torch::Tensor, torch::Tensor>
sparse_mla_decode_from_bf16_workspace_split(
    const torch::Tensor& q, const torch::Tensor& kv_workspace,
    const pybind11::object& topk_length,
    const pybind11::object& extra_topk_length,
    const pybind11::object& attn_sink, int main_topk, int extra_topk,
    int head_dim_v, double softmax_scale, const pybind11::object& out);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
sparse_mla_prefill_from_bf16_workspace(
    const torch::Tensor& q, const torch::Tensor& kv,
    const torch::Tensor& indices, const pybind11::object& topk_length,
    const pybind11::object& attn_sink, int head_dim_v, double softmax_scale,
    const pybind11::object& out);

void build_prefill_workspace_map(const torch::Tensor& out,
                                 const torch::Tensor& block_table,
                                 const torch::Tensor& seq_lens,
                                 const torch::Tensor& workspace_starts,
                                 int block_size);

void build_prefill_strided_workspace_map(
    const torch::Tensor& out, const torch::Tensor& block_table,
    const torch::Tensor& seq_lens, const pybind11::object& gather_lens,
    int block_size, int64_t row_stride, int64_t offset, bool encode_negative);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
sparse_mla_prefill_from_fp8_workspace_map(
    const torch::Tensor& q, const torch::Tensor& k_cache,
    const torch::Tensor& workspace_map, const torch::Tensor& indices,
    const pybind11::object& topk_length, const pybind11::object& attn_sink,
    int block_size, int head_dim_v, double softmax_scale,
    const pybind11::object& out);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
sparse_mla_prefill_from_two_fp8_workspace_map(
    const torch::Tensor& q, const torch::Tensor& primary_k_cache,
    const torch::Tensor& extra_k_cache, const torch::Tensor& workspace_map,
    const torch::Tensor& indices, const pybind11::object& topk_length,
    const pybind11::object& attn_sink, int primary_block_size,
    int extra_block_size, int head_dim_v, double softmax_scale,
    const pybind11::object& out);

void register_apis(pybind11::module& m);

} // namespace sm120_mla
} // namespace deep_gemm
