#!/usr/bin/env python3
"""Sweep the patched vLLM SM120 sparse-prefill helper chunk size.

This measures the actual Python helper installed by docker/patch_vllm_deepseekv4.py
rather than only DeepGEMM extension entry points.  It is intended to catch the
full bridge cost: per-chunk gather, BMMs, masking/softmax, sink gating, and
framework launch overhead.
"""

import argparse
import os
import time
from dataclasses import dataclass

import torch

from vllm.third_party.flashmla.flash_mla_interface import (  # noqa: E402
    _sm120_flash_mla_sparse_prefill_fwd,
)


@dataclass
class Case:
    q: torch.Tensor
    kv: torch.Tensor
    indices: torch.Tensor
    lens: torch.Tensor
    sink: torch.Tensor
    out: torch.Tensor


def make_case(
    tokens: int,
    heads: int,
    kv_tokens: int,
    topk: int,
    lens_value: int | None,
    device: str,
) -> Case:
    torch.manual_seed(20260426 + tokens + heads + kv_tokens + topk)
    q = (torch.randn((tokens, heads, 512), device=device, dtype=torch.float32) * 0.05).to(
        torch.bfloat16
    ).contiguous()
    kv = (torch.randn((kv_tokens, 1, 512), device=device, dtype=torch.float32) * 0.05).to(
        torch.bfloat16
    ).contiguous()
    indices = torch.randint(
        0, kv_tokens, (tokens, 1, topk), device=device, dtype=torch.int32
    ).contiguous()
    lens = torch.full(
        (tokens,),
        topk if lens_value is None else min(topk, max(0, lens_value)),
        device=device,
        dtype=torch.int64,
    )
    if lens_value is not None and lens_value < topk:
        pos = torch.arange(topk, device=device).view(1, 1, topk)
        indices = torch.where(pos < lens.view(tokens, 1, 1), indices, -1)
    sink = torch.zeros((heads,), device=device, dtype=torch.float32)
    out = torch.empty((tokens, heads, 512), device=device, dtype=torch.bfloat16)
    return Case(q, kv, indices, lens, sink, out)


def run_helper(case: Case, scale: float):
    return _sm120_flash_mla_sparse_prefill_fwd(
        case.q,
        case.kv,
        case.indices,
        scale,
        512,
        case.sink,
        case.lens,
        case.out,
    )


def bench(
    case: Case,
    chunk: int,
    warmup: int,
    iters: int,
    compile_bridge: bool,
    cudnn: bool,
    trim_topk: bool,
    trim_min_width: int,
    gather_workspace: bool,
    index_select_workspace: bool,
):
    os.environ["DG_SM120_PREFILL_WORKSPACE_CHUNK"] = str(chunk)
    os.environ["DG_SM120_PREFILL_TORCH_BMM"] = "1"
    os.environ["DG_SM120_PREFILL_TRUST_INDICES"] = "1"
    os.environ["DG_SM120_PREFILL_TORCH_COMPILE"] = "1" if compile_bridge else "0"
    os.environ["DG_SM120_PREFILL_CUDNN"] = "1" if cudnn else "0"
    os.environ["DG_SM120_PREFILL_CUDNN_UNMASKED"] = "1" if cudnn else "0"
    os.environ["DG_SM120_PREFILL_TRIM_TOPK"] = "1" if trim_topk else "0"
    os.environ["DG_SM120_PREFILL_TRIM_TOPK_MIN_WIDTH"] = str(trim_min_width)
    os.environ["DG_SM120_PREFILL_GATHER_WORKSPACE"] = "1" if gather_workspace else "0"
    os.environ["DG_SM120_PREFILL_INDEX_SELECT"] = "1" if index_select_workspace else "0"
    scale = 512.0**-0.5

    for _ in range(warmup):
        run_helper(case, scale)
    torch.cuda.synchronize()

    start = time.perf_counter()
    got = None
    for _ in range(iters):
        got = run_helper(case, scale)
    torch.cuda.synchronize()
    assert got is not None
    return (time.perf_counter() - start) * 1e6 / iters, got


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, nargs="+", default=[64, 128, 256])
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--kv-tokens", type=int, default=2048)
    parser.add_argument("--topk", type=int, default=2048)
    parser.add_argument(
        "--lens",
        type=int,
        default=None,
        help="Optional effective top-k length; padded entries are set to -1.",
    )
    parser.add_argument("--chunks", type=int, nargs="+", default=[16, 32, 64, 128, 256])
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--iters", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--compile-bridge", action="store_true")
    parser.add_argument("--no-trim-topk", action="store_true")
    parser.add_argument("--trim-topk-min-width", type=int, default=2048)
    parser.add_argument(
        "--gather-workspace",
        action="store_true",
        help="Use the DeepGEMM reusable workspace gather kernel instead of advanced indexing.",
    )
    parser.add_argument(
        "--index-select-workspace",
        action="store_true",
        help="Use torch.index_select into a reusable workspace instead of advanced indexing.",
    )
    parser.add_argument(
        "--cudnn",
        action="store_true",
        help="Use the experimental cuDNN SDPA+logsumexp prefill bridge.",
    )
    args = parser.parse_args()

    for tokens in args.tokens:
        case = make_case(
            tokens, args.heads, args.kv_tokens, args.topk, args.lens, args.device
        )
        print(
            f"tokens={tokens} heads={args.heads} kv_tokens={args.kv_tokens} "
            f"topk={args.topk} lens={args.lens} compile_bridge={args.compile_bridge} "
            f"cudnn={args.cudnn} trim_topk={not args.no_trim_topk} "
            f"trim_min_width={args.trim_topk_min_width} "
            f"gather_workspace={args.gather_workspace} "
            f"index_select_workspace={args.index_select_workspace}"
        )
        ref_out = None
        ref_lse = None
        for chunk in args.chunks:
            if chunk > tokens:
                continue
            try:
                us, got = bench(
                    case,
                    chunk,
                    args.warmup,
                    args.iters,
                    args.compile_bridge,
                    args.cudnn,
                    not args.no_trim_topk,
                    args.trim_topk_min_width,
                    args.gather_workspace,
                    args.index_select_workspace,
                )
            except RuntimeError as exc:
                print(f"  chunk={chunk} ERROR {type(exc).__name__}: {str(exc).splitlines()[0]}")
                torch.cuda.empty_cache()
                continue
            if ref_out is None:
                ref_out = got[0].detach().clone()
                ref_lse = got[2].detach().clone()
                out_diff = 0.0
                lse_diff = 0.0
            else:
                out_diff = (got[0].float() - ref_out.float()).abs().max().item()
                lse_diff = (got[2].float() - ref_lse.float()).abs().max().item()
            print(
                f"  chunk={chunk} us={us:.3f} per_token_us={us / tokens:.3f} "
                f"out_diff={out_diff:.6g} lse_diff={lse_diff:.6g}"
            )


if __name__ == "__main__":
    main()
