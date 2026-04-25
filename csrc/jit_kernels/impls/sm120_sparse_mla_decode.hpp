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

void register_apis(pybind11::module& m);

} // namespace sm120_mla
} // namespace deep_gemm
