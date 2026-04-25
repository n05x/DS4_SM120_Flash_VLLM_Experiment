#!/usr/bin/env python3
import argparse
import os
import time

import torch

import deep_gemm


SHAPES = (
    ("fused_wqa_wkv", 1536, 4096),
    ("wq_b_tp2", 16384, 1024),
    ("wo_b", 4096, 4096),
    ("router_gate", 256, 4096),
)


def make_case(m: int, n: int, k: int, device: str):
    assert n % 128 == 0
    assert k % 128 == 0
    torch.manual_seed(1234 + m + n + k)
    a = (torch.randn((m, k), device=device, dtype=torch.float32) * 0.25).to(
        torch.float8_e4m3fn
    )
    b = (torch.randn((n, k), device=device, dtype=torch.float32) * 0.25).to(
        torch.float8_e4m3fn
    )
    scale_words = (k // 128 + 3) // 4
    sfa = torch.full((m, scale_words), 0x7F7F7F7F, device=device, dtype=torch.int32)
    sfb = torch.full((n, scale_words), 0x7F7F7F7F, device=device, dtype=torch.int32)
    out = torch.empty((m, n), device=device, dtype=torch.bfloat16)
    return a.contiguous(), sfa.contiguous(), b.contiguous(), sfb.contiguous(), out


def set_mode(mode: str) -> None:
    os.environ.pop("DG_SM120_ENABLE_FP8_M1_MMA", None)
    os.environ.pop("DG_SM120_ENABLE_FP8_M1_KBLOCK", None)
    os.environ.pop("DG_SM120_ENABLE_FP8_M1_REGSCALE", None)
    os.environ.pop("DG_SM120_ENABLE_FP8_M1_WARPCOL", None)
    os.environ.pop("DG_SM120_ENABLE_FP8_M1_WARPCOL_HEURISTIC", None)
    os.environ.pop("DG_SM120_FP8_M1_KBLOCK_COLS", None)
    os.environ.pop("DG_SM120_FP8_M1_KBLOCK_THREADS", None)
    if mode == "cute":
        os.environ["DG_SM120_ENABLE_FP8_M1_MMA"] = "1"
    elif mode.startswith("kblock"):
        os.environ["DG_SM120_ENABLE_FP8_M1_KBLOCK"] = "1"
        parts = mode.split(":")
        if len(parts) >= 2:
            os.environ["DG_SM120_FP8_M1_KBLOCK_COLS"] = parts[1]
        if len(parts) >= 3:
            os.environ["DG_SM120_FP8_M1_KBLOCK_THREADS"] = parts[2]
    elif mode.startswith("regscale"):
        os.environ["DG_SM120_ENABLE_FP8_M1_REGSCALE"] = "1"
        parts = mode.split(":")
        if len(parts) >= 2:
            os.environ["DG_SM120_FP8_M1_KBLOCK_COLS"] = parts[1]
        if len(parts) >= 3:
            os.environ["DG_SM120_FP8_M1_KBLOCK_THREADS"] = parts[2]
    elif mode.startswith("warpcol"):
        os.environ["DG_SM120_ENABLE_FP8_M1_WARPCOL"] = "1"
        parts = mode.split(":")
        if len(parts) >= 2:
            os.environ["DG_SM120_FP8_M1_KBLOCK_COLS"] = parts[1]
    elif mode == "hybrid":
        os.environ["DG_SM120_ENABLE_FP8_M1_KBLOCK"] = "1"
        os.environ["DG_SM120_ENABLE_FP8_M1_WARPCOL_HEURISTIC"] = "1"


def call(case):
    a, sfa, b, sfb, out = case
    deep_gemm._C.fp8_gemm_nt(
        (a, sfa),
        (b, sfb),
        out,
        recipe_a=(1, 128),
        recipe_b=(1, 128),
    )


def bench(case, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        call(case)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        call(case)
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1e6 / iters


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m-values", default="1,4,8")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--enable-m1-mma", action="store_true")
    parser.add_argument(
        "--modes",
        default=None,
        help="Comma-separated modes: default, cute, kblock[:cols[:threads]], regscale[:cols[:threads]], warpcol[:warps]",
    )
    args = parser.parse_args()

    modes = (
        [x for x in args.modes.split(",") if x]
        if args.modes
        else (["cute"] if args.enable_m1_mma else ["default"])
    )

    print(f"modes: {','.join(modes)}")
    for m in [int(x) for x in args.m_values.split(",") if x]:
        for name, n, k in SHAPES:
            try:
                case = make_case(m, n, k, args.device)
                row = [f"{name}: m={m} n={n} k={k}"]
                for mode in modes:
                    set_mode(mode)
                    us = bench(case, args.warmup, args.iters)
                    row.append(f"{mode}={us:.3f}us")
                print(" ".join(row))
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"{name}: m={m} n={n} k={k} OOM")


if __name__ == "__main__":
    main()
