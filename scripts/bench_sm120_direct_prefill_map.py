#!/usr/bin/env python3
"""Benchmark the experimental direct-FP8 sparse-prefill workspace-map path.

This script intentionally avoids loading DeepSeek V4/vLLM.  It constructs a
synthetic FP8-MLA cache with the same packed block layout assumed by the SM120
extension, compares the direct cache-reading path against the current BF16
workspace split path, and reports both correctness deltas and latency.
"""

from __future__ import annotations

import argparse
import math
import time

import torch

import deep_gemm


HEAD_DIM = 512
FP8_DIM = 448
BF16_DIM = 64
TOKEN_DATA_BYTES = FP8_DIM + BF16_DIM * 2
SCALE_BYTES = 8


def bench(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1e6 / iters


def fill_cache_from_rope(cache: torch.Tensor, rope: torch.Tensor, block_size: int) -> None:
    """Populate the synthetic packed FP8-MLA cache.

    The direct kernels treat each block as all token payloads first, followed by
    the per-token UE8M0 scale bytes.  For correctness isolation, the first 448
    FP8 dimensions are set to zero and only the BF16 rope tail carries data.
    """

    blocks = cache.shape[0]
    flat = cache.view(blocks, -1)
    flat.zero_()
    rope_bytes = rope.contiguous().view(torch.uint8).view(-1, BF16_DIM * 2)
    for linear in range(rope.shape[0]):
        block = linear // block_size
        offset = linear - block * block_size
        token_base = offset * TOKEN_DATA_BYTES
        flat[block, token_base + FP8_DIM : token_base + TOKEN_DATA_BYTES].copy_(
            rope_bytes[linear]
        )
        scale_base = block_size * TOKEN_DATA_BYTES + offset * SCALE_BYTES
        flat[block, scale_base : scale_base + SCALE_BYTES].fill_(127)


def make_case(tokens: int, heads: int, topk: int, block_size: int, device: str):
    torch.manual_seed(20260426)
    kv_tokens = tokens * topk
    blocks = math.ceil(kv_tokens / block_size)
    cache = torch.empty(
        (blocks, 1, 1, block_size * (TOKEN_DATA_BYTES + SCALE_BYTES)),
        device=device,
        dtype=torch.uint8,
    )
    rope = (torch.randn((blocks * block_size, BF16_DIM), device=device) * 0.05).to(
        torch.bfloat16
    )
    fill_cache_from_rope(cache, rope, block_size)

    q = (torch.randn((tokens, heads, HEAD_DIM), device=device) * 0.05).to(torch.bfloat16)
    linears = torch.arange(kv_tokens, device=device, dtype=torch.int32).view(tokens, topk)
    workspace_map = linears.reshape(-1).contiguous()
    indices = torch.arange(kv_tokens, device=device, dtype=torch.int32).view(tokens, 1, topk)
    lens = torch.full((tokens,), topk, device=device, dtype=torch.int64)
    sink = torch.zeros((heads,), device=device, dtype=torch.float32)

    workspace = torch.zeros((tokens, topk, HEAD_DIM), device=device, dtype=torch.bfloat16)
    workspace[:, :, FP8_DIM:].copy_(rope[linears.to(torch.long)])
    out_workspace = torch.empty((tokens, heads, HEAD_DIM), device=device, dtype=torch.bfloat16)
    out_direct = torch.empty_like(out_workspace)
    return q, cache, workspace_map, indices, lens, sink, workspace, out_workspace, out_direct


def call_workspace(case, scale: float):
    q, _cache, _map, _indices, lens, sink, workspace, out_workspace, _out_direct = case
    return deep_gemm._C.sm120_sparse_mla_decode_from_bf16_workspace_split(
        q.unsqueeze(1),
        workspace,
        lens,
        None,
        sink,
        workspace.shape[1],
        0,
        HEAD_DIM,
        scale,
        out_workspace.unsqueeze(1),
    )


def call_direct(case, scale: float, block_size: int):
    q, cache, workspace_map, indices, lens, sink, _workspace, _out_workspace, out_direct = case
    return deep_gemm._C.sm120_sparse_mla_prefill_from_two_fp8_workspace_map(
        q,
        cache,
        cache,
        workspace_map,
        indices,
        lens,
        sink,
        block_size,
        block_size,
        HEAD_DIM,
        scale,
        out_direct,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=16)
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--topk", type=int, default=512)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    scale = HEAD_DIM ** -0.5
    case = make_case(args.tokens, args.heads, args.topk, args.block_size, args.device)
    ws_out, ws_lse = call_workspace(case, scale)
    direct_out, _direct_max, direct_lse = call_direct(case, scale, args.block_size)
    torch.cuda.synchronize()

    ws = ws_out.squeeze(1)
    out_diff = (direct_out.float() - ws.float()).abs()
    lse_diff = (direct_lse.float() - ws_lse.squeeze(-1).float()).abs()
    print(
        f"shape: tokens={args.tokens} heads={args.heads} topk={args.topk} "
        f"block_size={args.block_size}"
    )
    print(
        f"direct_out_max_abs={out_diff.max().item():.6g} "
        f"direct_out_mean_abs={out_diff.mean().item():.6g} "
        f"direct_lse_max_abs={lse_diff.max().item():.6g}"
    )
    print(
        f"workspace_us: {bench(lambda: call_workspace(case, scale), args.warmup, args.iters):.3f}"
    )
    print(
        f"direct_fp8_map_us: {bench(lambda: call_direct(case, scale, args.block_size), args.warmup, args.iters):.3f}"
    )


if __name__ == "__main__":
    main()
