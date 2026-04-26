#!/usr/bin/env python3
import argparse
import time

import torch

import deep_gemm
from vllm.model_executor.layers.fused_moe.deep_gemm_utils import ep_gather


def bench(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1e6 / iters


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=162)
    parser.add_argument("--hidden", type=int, default=7168)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=500)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--weights-dtype", choices=["fp32", "bf16"], default="fp32")
    parser.add_argument("--ids-dtype", choices=["int64", "int32"], default="int64")
    parser.add_argument("--expert-map", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(1234)
    rows = args.tokens * args.topk
    a = torch.randn((rows, args.hidden), device=args.device, dtype=torch.bfloat16)
    ids_dtype = torch.int64 if args.ids_dtype == "int64" else torch.int32
    topk_ids = torch.randint(0, 128, (args.tokens, args.topk), device=args.device,
                             dtype=ids_dtype)
    expert_map = None
    if args.expert_map:
        expert_map = torch.full((128,), -1, device=args.device, dtype=torch.int32)
        expert_map[:64] = torch.arange(64, device=args.device, dtype=torch.int32)
        topk_ids = torch.randint(0, 64, (args.tokens, args.topk),
                                 device=args.device, dtype=ids_dtype)
    weight_dtype = torch.float32 if args.weights_dtype == "fp32" else torch.bfloat16
    topk_weights = torch.rand((args.tokens, args.topk), device=args.device,
                              dtype=torch.float32)
    topk_weights = (topk_weights / topk_weights.sum(dim=1, keepdim=True)).to(weight_dtype)
    inv_perm = torch.arange(rows, device=args.device, dtype=torch.int32).view(
        args.tokens, args.topk
    )
    out_triton = torch.empty((args.tokens, args.hidden), device=args.device,
                             dtype=torch.bfloat16)
    out_sm120 = torch.empty_like(out_triton)

    ep_gather(a, topk_ids, topk_weights, inv_perm, expert_map, out_triton)
    if expert_map is None:
        deep_gemm._C.sm120_moe_unpermute_reduce_bf16(
            a, topk_ids, topk_weights, inv_perm, out_sm120
        )
        custom_call = lambda: deep_gemm._C.sm120_moe_unpermute_reduce_bf16(
            a, topk_ids, topk_weights, inv_perm, out_sm120
        )
    else:
        deep_gemm._C.sm120_moe_unpermute_reduce_bf16_mapped(
            a, topk_ids, topk_weights, inv_perm, expert_map, out_sm120
        )
        custom_call = lambda: deep_gemm._C.sm120_moe_unpermute_reduce_bf16_mapped(
            a, topk_ids, topk_weights, inv_perm, expert_map, out_sm120
        )
    torch.cuda.synchronize()
    diff = (out_triton.float() - out_sm120.float()).abs()
    ref_abs = out_triton.float().abs().clamp_min(1e-6)

    triton_us = bench(
        lambda: ep_gather(a, topk_ids, topk_weights, inv_perm, expert_map, out_triton),
        args.warmup,
        args.iters,
    )
    sm120_us = bench(custom_call, args.warmup, args.iters)

    print(
        f"shape: tokens={args.tokens} hidden={args.hidden} topk={args.topk} "
        f"weights={args.weights_dtype} ids={args.ids_dtype} "
        f"expert_map={args.expert_map}"
    )
    print(
        f"triton_ep_gather: {triton_us:.3f} us\n"
        f"sm120_cuda:       {sm120_us:.3f} us "
        f"speedup={triton_us / sm120_us:.3f}x"
    )
    print(
        f"max_abs_diff={diff.max().item():.6g} "
        f"mean_abs_diff={diff.mean().item():.6g} "
        f"max_rel_diff={(diff / ref_abs).max().item():.6g}"
    )


if __name__ == "__main__":
    main()
