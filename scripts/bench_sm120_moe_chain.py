#!/usr/bin/env python3
"""Benchmark the SM120 MoE FC1 -> SiLU/quant -> FC2 chain.

This synthetic chain uses the same DeepGEMM grouped FP8xFP4 kernels and the
SM120 fused SiLU*mul+FP8 quant helper used by the vLLM patcher.  It does not
model routing/unpermute overhead; it isolates the larger MoE boundary where a
future persistent/fused expert kernel could keep FC1 activations resident and
avoid intermediate global-memory traffic.
"""

import argparse
import os
import time

import torch

import deep_gemm


def set_mode(mode: str) -> None:
    for name in (
        "DG_SM120_ENABLE_SMALL_M_MMA",
        "DG_SM120_MOE_SKIP_SFA_FILL",
        "DG_SM120_MOE_ROW_GROUPED",
        "DG_SM120_MOE_ROW_GROUPED_SKIP_SFA_FILL",
        "DG_SM120_MOE_DIRECT_GROUPS_WHEN_NO_SHRINK",
    ):
        os.environ.pop(name, None)
    if mode == "direct_groups":
        os.environ["DG_SM120_MOE_DIRECT_GROUPS_WHEN_NO_SHRINK"] = "1"
    elif mode == "small_m":
        os.environ["DG_SM120_ENABLE_SMALL_M_MMA"] = "1"
    elif mode == "skip_fill":
        os.environ["DG_SM120_MOE_SKIP_SFA_FILL"] = "1"
    elif mode == "row_grouped":
        os.environ["DG_SM120_MOE_ROW_GROUPED"] = "1"
    elif mode == "row_grouped_skip_fill":
        os.environ["DG_SM120_MOE_ROW_GROUPED"] = "1"
        os.environ["DG_SM120_MOE_ROW_GROUPED_SKIP_SFA_FILL"] = "1"


def make_layout(m: int, groups: int, active_groups: int, device: str):
    grouped_layout = torch.full((m,), -1, device=device, dtype=torch.int32)
    starts = torch.zeros((groups,), device=device, dtype=torch.int32)
    counts = torch.zeros((groups,), device=device, dtype=torch.int32)
    base = m // active_groups
    rem = m % active_groups
    offset = 0
    for group in range(active_groups):
        count = base + (1 if group < rem else 0)
        grouped_layout[offset : offset + count] = group
        starts[group] = offset
        counts[group] = count
        offset += count
    return grouped_layout, starts, counts


def make_case(m: int, hidden: int, moe_hidden: int, groups: int, active_groups: int, device: str):
    assert hidden % 128 == 0 and moe_hidden % 128 == 0
    torch.manual_seed(20260426 + m + hidden + moe_hidden)

    a = (torch.randn((m, hidden), device=device, dtype=torch.float32) * 0.25).to(
        torch.float8_e4m3fn
    ).contiguous()
    sfa_words = (hidden + 511) // 512
    sfa = torch.full((m, sfa_words), 0x7F7F7F7F, device=device, dtype=torch.int32)

    fc1_n = moe_hidden * 2
    w1 = torch.randint(-128, 128, (groups, fc1_n, hidden // 2), device=device, dtype=torch.int8).contiguous()
    w1_sfb_raw = torch.full((groups, fc1_n, hidden // 32), 0x7F, device=device, dtype=torch.uint8)
    w1_sfb = deep_gemm._C.sm120_prepack_fp8_fp4_sfb(w1_sfb_raw, 128, fc1_n, hidden)

    w2 = torch.randint(-128, 128, (groups, hidden, moe_hidden // 2), device=device, dtype=torch.int8).contiguous()
    w2_sfb_raw = torch.full((groups, hidden, moe_hidden // 32), 0x7F, device=device, dtype=torch.uint8)
    w2_sfb = deep_gemm._C.sm120_prepack_fp8_fp4_sfb(w2_sfb_raw, 128, hidden, moe_hidden)

    layout = make_layout(m, groups, active_groups, device)
    fc1_out = torch.empty((m, fc1_n), device=device, dtype=torch.bfloat16)
    act_q = torch.empty((m, moe_hidden), device=device, dtype=torch.float8_e4m3fn)
    fc2_out = torch.empty((m, hidden), device=device, dtype=torch.bfloat16)
    return (a, sfa, w1, w1_sfb, w2, w2_sfb, *layout, fc1_out, act_q, fc2_out)


def fc1(case):
    a, sfa, w1, w1_sfb, _w2, _w2_sfb, layout, starts, counts, fc1_out, _act_q, _fc2_out = case
    deep_gemm._C.m_grouped_fp8_fp4_gemm_nt_contiguous_with_starts(
        (a, sfa), (w1, w1_sfb), fc1_out, layout, starts, counts, recipe_a=(1, 128), recipe_b=(1, 32)
    )
    return fc1_out


def act_quant(case):
    *_prefix, fc1_out, act_q, _fc2_out = case
    return deep_gemm._C.sm120_silu_mul_quant_fp8_packed(fc1_out, act_q, 128, 0.0)


def fc2(case, act_scale):
    _a, _sfa, _w1, _w1_sfb, w2, w2_sfb, layout, starts, counts, _fc1_out, act_q, fc2_out = case
    deep_gemm._C.m_grouped_fp8_fp4_gemm_nt_contiguous_with_starts(
        (act_q, act_scale), (w2, w2_sfb), fc2_out, layout, starts, counts, recipe_a=(1, 128), recipe_b=(1, 32)
    )
    return fc2_out


def chain(case):
    fc1(case)
    act_scale = act_quant(case)
    return fc2(case, act_scale)


def make_reduce_case(m: int, hidden: int, topk: int, device: str):
    assert m % topk == 0, "m must be divisible by topk to model per-token reduce"
    tokens = m // topk
    topk_ids = torch.zeros((tokens, topk), device=device, dtype=torch.int32)
    weights = torch.rand((tokens, topk), device=device, dtype=torch.float32)
    weights = (weights / weights.sum(dim=1, keepdim=True)).to(torch.bfloat16)
    inv_perm = torch.arange(m, device=device, dtype=torch.int32).view(tokens, topk)
    output = torch.empty((tokens, hidden), device=device, dtype=torch.bfloat16)
    return topk_ids, weights, inv_perm, output


def reduce_fc2(case, reduce_case):
    *_prefix, fc2_out = case
    topk_ids, weights, inv_perm, output = reduce_case
    deep_gemm._C.sm120_moe_unpermute_reduce_bf16(
        fc2_out, topk_ids, weights, inv_perm, output
    )
    return output


def chain_with_reduce(case, reduce_case):
    fc1(case)
    act_scale = act_quant(case)
    fc2(case, act_scale)
    return reduce_fc2(case, reduce_case)


def bench(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1e6 / iters


def bench_graph(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        graph.replay()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1e6 / iters


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=12)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--moe-hidden", type=int, default=2048)
    parser.add_argument("--groups", type=int, default=128)
    parser.add_argument("--active-groups", type=int, default=12)
    parser.add_argument("--mode", default="skip_fill")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--reduce-topk", type=int, default=0, help="If >0, also benchmark SM120 FC2 unpermute/reduce and full chain+reduce. Requires m divisible by reduce-topk.")
    parser.add_argument(
        "--graph",
        action="store_true",
        help="Also measure CUDA graph replay for the full FC1->activation->FC2 chain.",
    )
    args = parser.parse_args()

    set_mode(args.mode)
    case = make_case(args.m, args.hidden, args.moe_hidden, args.groups, args.active_groups, args.device)
    reduce_case = None
    if args.reduce_topk > 0:
        reduce_case = make_reduce_case(args.m, args.hidden, args.reduce_topk, args.device)
    act_scale = act_quant(case)
    torch.cuda.synchronize()

    fc1_us = bench(lambda: fc1(case), args.warmup, args.iters)
    act_us = bench(lambda: act_quant(case), args.warmup, args.iters)
    fc2_us = bench(lambda: fc2(case, act_scale), args.warmup, args.iters)
    chain_us = bench(lambda: chain(case), args.warmup, args.iters)
    reduce_us = None
    chain_reduce_us = None
    graph_chain_reduce_us = None
    if reduce_case is not None:
        reduce_us = bench(lambda: reduce_fc2(case, reduce_case), args.warmup, args.iters)
        chain_reduce_us = bench(lambda: chain_with_reduce(case, reduce_case), args.warmup, args.iters)
    graph_chain_us = None
    if args.graph:
        graph_chain_us = bench_graph(lambda: chain(case), args.warmup, args.iters)
        if reduce_case is not None:
            graph_chain_reduce_us = bench_graph(lambda: chain_with_reduce(case, reduce_case), args.warmup, args.iters)
    separate_sum = fc1_us + act_us + fc2_us

    print(
        f"shape: m={args.m} hidden={args.hidden} moe_hidden={args.moe_hidden} "
        f"groups={args.groups} active_groups={args.active_groups} mode={args.mode} "
        f"reduce_topk={args.reduce_topk}"
    )
    print(f"fc1_us: {fc1_us:.3f}")
    print(f"act_quant_us: {act_us:.3f}")
    print(f"fc2_us: {fc2_us:.3f}")
    print(f"separate_sum_us: {separate_sum:.3f}")
    if reduce_us is not None:
        print(f"reduce_us: {reduce_us:.3f}")
        print(f"separate_sum_plus_reduce_us: {separate_sum + reduce_us:.3f}")
    print(f"chain_us: {chain_us:.3f}")
    if chain_reduce_us is not None:
        print(f"chain_reduce_us: {chain_reduce_us:.3f}")
    if graph_chain_us is not None:
        print(f"graph_chain_us: {graph_chain_us:.3f}")
        print(f"graph_chain_over_eager_chain: {graph_chain_us / chain_us:.3f}")
        if graph_chain_reduce_us is not None:
            print(f"graph_chain_reduce_us: {graph_chain_reduce_us:.3f}")
            print(f"graph_chain_reduce_over_eager_chain_reduce: {graph_chain_reduce_us / chain_reduce_us:.3f}")
    print(f"chain_over_separate: {chain_us / separate_sum:.3f}")
    print(
        "fusion_target_us: "
        f"{separate_sum:.3f} minus avoidable global activation traffic / launches"
    )


if __name__ == "__main__":
    main()
