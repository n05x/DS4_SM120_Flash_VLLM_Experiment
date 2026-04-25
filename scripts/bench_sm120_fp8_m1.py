#!/usr/bin/env python3
import argparse
import os
import time

import torch

import deep_gemm


def make_case(n: int, k: int, device: str):
    assert n % 128 == 0
    assert k % 128 == 0
    torch.manual_seed(1234)
    a = (torch.randn((1, k), device=device, dtype=torch.float32) * 0.25).to(
        torch.float8_e4m3fn
    )
    b = (torch.randn((n, k), device=device, dtype=torch.float32) * 0.25).to(
        torch.float8_e4m3fn
    )
    # UE8M0 1.0, packed as four 128-wide scale blocks per int32 word.
    sfa = torch.full((1, (k // 128 + 3) // 4), 0x7F7F7F7F,
                     device=device, dtype=torch.int32)
    sfb = torch.full((n, (k // 128 + 3) // 4), 0x7F7F7F7F,
                     device=device, dtype=torch.int32)
    return a.contiguous(), sfa.contiguous(), b.contiguous(), sfb.contiguous()


def call_kernel(case, out):
    a, sfa, b, sfb = case
    deep_gemm._C.fp8_gemm_nt(
        (a, sfa),
        (b, sfb),
        out,
        recipe_a=(1, 128),
        recipe_b=(1, 128),
    )


def set_mode(mode: str) -> None:
    os.environ.pop("DG_SM120_ENABLE_FP8_M1_MMA", None)
    os.environ.pop("DG_SM120_ENABLE_FP8_M1_KBLOCK", None)
    os.environ.pop("DG_SM120_ENABLE_FP8_M1_REGSCALE", None)
    os.environ.pop("DG_SM120_ENABLE_FP8_M1_WARPCOL", None)
    os.environ.pop("DG_SM120_ENABLE_FP8_M1_WARPCOL_HEURISTIC", None)
    os.environ.pop("DG_SM120_ENABLE_FP8_M1_CONTIG_UE8M0_KBLOCK", None)
    os.environ.pop("DG_SM120_DISABLE_FP8_M1_CONTIG_UE8M0_KBLOCK", None)
    os.environ.pop("DG_SM120_FP8_M1_KBLOCK_COLS", None)
    os.environ.pop("DG_SM120_FP8_M1_KBLOCK_THREADS", None)
    if mode == "cute":
        os.environ["DG_SM120_ENABLE_FP8_M1_MMA"] = "1"
    elif mode.startswith("kblock"):
        os.environ["DG_SM120_ENABLE_FP8_M1_KBLOCK"] = "1"
        os.environ["DG_SM120_DISABLE_FP8_M1_CONTIG_UE8M0_KBLOCK"] = "1"
        parts = mode.split(":")
        if len(parts) >= 2:
            os.environ["DG_SM120_FP8_M1_KBLOCK_COLS"] = parts[1]
        if len(parts) >= 3:
            os.environ["DG_SM120_FP8_M1_KBLOCK_THREADS"] = parts[2]
    elif mode.startswith("contig"):
        os.environ["DG_SM120_ENABLE_FP8_M1_KBLOCK"] = "1"
        os.environ["DG_SM120_ENABLE_FP8_M1_CONTIG_UE8M0_KBLOCK"] = "1"
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


def bench(case, out, mode: str, warmup: int, iters: int) -> float:
    set_mode(mode)
    for _ in range(warmup):
        call_kernel(case, out)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        call_kernel(case, out)
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1e6 / iters


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=7168)
    parser.add_argument("--k", type=int, default=7168)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--modes",
        default="default,kblock,hybrid,regscale:4:128,warpcol:4,cute",
        help="Comma-separated modes: default, cute, kblock[:cols[:threads]], contig[:cols[:threads]], regscale[:cols[:threads]], warpcol[:warps]",
    )
    args = parser.parse_args()

    case = make_case(args.n, args.k, args.device)
    modes = [x for x in args.modes.split(",") if x]
    outputs = {
        mode: torch.empty((1, args.n), device=args.device, dtype=torch.bfloat16)
        for mode in modes
    }

    set_mode("default")
    ref = torch.empty((1, args.n), device=args.device, dtype=torch.bfloat16)
    call_kernel(case, ref)
    for mode, out in outputs.items():
        set_mode(mode)
        call_kernel(case, out)
    torch.cuda.synchronize()

    print(f"shape: m=1 n={args.n} k={args.k}")
    ref_abs = ref.float().abs().clamp_min(1e-6)
    timings = {}
    for mode in modes:
        diff = (ref.float() - outputs[mode].float()).abs()
        timings[mode] = bench(case, outputs[mode], mode, args.warmup, args.iters)
        print(
            f"{mode}: us={timings[mode]:.3f} "
            f"speedup={timings['default'] / timings[mode]:.3f}x "
            f"max_abs_diff={diff.max().item():.6g} "
            f"mean_abs_diff={diff.mean().item():.6g} "
            f"max_rel_diff={(diff / ref_abs).max().item():.6g}"
        )


if __name__ == "__main__":
    main()
