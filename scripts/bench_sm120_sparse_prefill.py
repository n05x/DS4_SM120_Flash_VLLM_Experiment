#!/usr/bin/env python3
import argparse
import time

import torch

import deep_gemm


def make_case(tokens: int, heads: int, kv_tokens: int, topk: int, dtype, device: str):
    torch.manual_seed(20260426)
    q = (torch.randn((tokens, heads, 512), device=device, dtype=torch.float32) * 0.05).to(dtype)
    kv = (torch.randn((kv_tokens, 1, 512), device=device, dtype=torch.float32) * 0.05).to(dtype)
    indices = torch.randint(0, kv_tokens, (tokens, 1, topk), device=device, dtype=torch.int32)
    lens = torch.full((tokens,), topk, device=device, dtype=torch.int64)
    sink = torch.zeros((heads,), device=device, dtype=torch.float32)
    out = torch.empty((tokens, heads, 512), device=device, dtype=dtype)
    return q.contiguous(), kv.contiguous(), indices.contiguous(), lens, sink, out


def call_native(case, scale: float):
    q, kv, indices, lens, sink, out = case
    return deep_gemm._C.sm120_sparse_mla_prefill_from_bf16_workspace(
        q, kv, indices, lens, sink, 512, scale, out
    )


def call_split(case, scale: float):
    q, kv, indices, lens, sink, out = case
    return deep_gemm._C.sm120_sparse_mla_prefill_from_bf16_workspace_split(
        q, kv, indices, lens, sink, 512, scale, out
    )


def call_chunked_workspace(case, scale: float):
    q, kv, indices, lens, sink, out = case
    kv2d = kv[:, 0, :]
    workspace = kv2d[indices[:, 0, :].to(torch.long)]
    dec_out, dec_lse = deep_gemm._C.sm120_sparse_mla_decode_from_bf16_workspace_split(
        q.unsqueeze(1), workspace, lens, None, sink, workspace.shape[1], 0, 512, scale, out.unsqueeze(1)
    )
    return dec_out.squeeze(1), torch.empty((q.shape[0], q.shape[1]), device=q.device, dtype=torch.float32), dec_lse.squeeze(-1)


def call_chunked_workspace_gather(case, scale: float):
    q, kv, indices, lens, sink, out = case
    kv2d = kv[:, 0, :]
    workspace = torch.empty(
        (q.shape[0], indices.shape[-1], kv2d.shape[-1]),
        device=q.device,
        dtype=kv2d.dtype,
    )
    deep_gemm._C.sm120_gather_bf16_workspace(kv2d, indices[:, 0, :], workspace)
    dec_out, dec_lse = deep_gemm._C.sm120_sparse_mla_decode_from_bf16_workspace_split(
        q.unsqueeze(1), workspace, lens, None, sink, workspace.shape[1], 0, 512, scale, out.unsqueeze(1)
    )
    return dec_out.squeeze(1), torch.empty((q.shape[0], q.shape[1]), device=q.device, dtype=torch.float32), dec_lse.squeeze(-1)


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
    parser.add_argument("--tokens", type=int, default=16)
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--kv-tokens", type=int, default=2048)
    parser.add_argument("--topk", type=int, default=2048)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    scale = 512.0 ** -0.5
    case = make_case(args.tokens, args.heads, args.kv_tokens, args.topk, torch.bfloat16, args.device)
    out_native, _max_native, lse_native = call_native(case, scale)
    out_split, _max_split, lse_split = call_split(case, scale)
    out_chunk, _max_chunk, lse_chunk = call_chunked_workspace(case, scale)
    out_gather, _max_gather, lse_gather = call_chunked_workspace_gather(case, scale)
    torch.cuda.synchronize()

    print(
        f"shape: tokens={args.tokens} heads={args.heads} kv_tokens={args.kv_tokens} topk={args.topk}"
    )
    for name, out, lse in (
        ("split", out_split, lse_split),
        ("chunked_workspace", out_chunk, lse_chunk),
        ("chunked_workspace_gather", out_gather, lse_gather),
    ):
        diff = (out.float() - out_native.float()).abs()
        lse_diff = (lse.float() - lse_native.float()).abs()
        print(
            f"{name}_out_max_abs={diff.max().item():.6g} "
            f"{name}_out_mean_abs={diff.mean().item():.6g} "
            f"{name}_lse_max_abs={lse_diff.max().item():.6g}"
        )
    print(f"native_us: {bench(lambda: call_native(case, scale), args.warmup, args.iters):.3f}")
    print(f"split_us: {bench(lambda: call_split(case, scale), args.warmup, args.iters):.3f}")
    print(
        f"chunked_workspace_us: {bench(lambda: call_chunked_workspace(case, scale), args.warmup, args.iters):.3f}"
    )
    print(
        "chunked_workspace_gather_us: "
        f"{bench(lambda: call_chunked_workspace_gather(case, scale), args.warmup, args.iters):.3f}"
    )


if __name__ == "__main__":
    main()
