#!/usr/bin/env python3
import argparse
import time

import torch

import deep_gemm


_COMPILED_TORCH_BMM = None


def _prefill_torch_bmm_inner(q, workspace, valid, valid_any, sink, scale: float):
    scores = torch.bmm(q, workspace.transpose(1, 2)).float().mul_(float(scale))
    scores.masked_fill_(~valid.unsqueeze(1), float("-inf"))
    scores = torch.where(valid_any.view(-1, 1, 1), scores, torch.zeros_like(scores))
    lse = torch.logsumexp(scores, dim=-1)
    probs = torch.softmax(scores, dim=-1).to(q.dtype)
    probs.masked_fill_(~valid_any.view(-1, 1, 1), 0)
    out = torch.bmm(probs, workspace)
    gate = torch.sigmoid(lse - sink.to(lse.dtype).view(1, -1))
    out = out * gate.to(out.dtype).unsqueeze(-1)
    return out, lse


def get_compiled_torch_bmm():
    global _COMPILED_TORCH_BMM
    if _COMPILED_TORCH_BMM is None:
        _COMPILED_TORCH_BMM = torch.compile(
            _prefill_torch_bmm_inner,
            mode="reduce-overhead",
            fullgraph=True,
        )
    return _COMPILED_TORCH_BMM


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


def call_gather_torch_bmm(case, scale: float):
    q, kv, indices, lens, sink, out = case
    kv2d = kv[:, 0, :]
    workspace = torch.empty(
        (q.shape[0], indices.shape[-1], kv2d.shape[-1]),
        device=q.device,
        dtype=kv2d.dtype,
    )
    deep_gemm._C.sm120_gather_bf16_workspace(kv2d, indices[:, 0, :], workspace)
    scores = torch.bmm(q, workspace.transpose(1, 2)).float().mul_(scale)
    if lens is not None:
        pos = torch.arange(workspace.shape[1], device=q.device)
        idx = indices[:, 0, : workspace.shape[1]]
        valid = (idx >= 0) & (idx < kv2d.shape[0])
        valid = valid & (pos.view(1, -1) < lens.to(torch.long).view(-1, 1))
        valid_any = valid.any(dim=-1)
        scores.masked_fill_(~valid.unsqueeze(1), float("-inf"))
        scores = torch.where(
            valid_any.view(-1, 1, 1), scores, torch.zeros_like(scores)
        )
    else:
        valid_any = None
    lse = torch.logsumexp(scores, dim=-1)
    probs = torch.softmax(scores, dim=-1).to(q.dtype)
    if valid_any is not None:
        probs.masked_fill_(~valid_any.view(-1, 1, 1), 0)
    out_bmm = torch.bmm(probs, workspace)
    if sink is not None:
        gate = torch.sigmoid(lse - sink.to(lse.dtype).view(1, -1))
        out_bmm = out_bmm * gate.to(out_bmm.dtype).unsqueeze(-1)
    out.copy_(out_bmm)
    if valid_any is not None:
        lse = lse.masked_fill(~valid_any.unsqueeze(1), float("-inf"))
    return out, torch.empty((q.shape[0], q.shape[1]), device=q.device, dtype=torch.float32), lse


def call_trusted_index_torch_bmm(case, scale: float):
    q, kv, indices, lens, sink, out = case
    kv2d = kv[:, 0, :]
    workspace = kv2d[indices[:, 0, :]]
    scores = torch.bmm(q, workspace.transpose(1, 2)).float().mul_(scale)
    if lens is not None:
        pos = torch.arange(workspace.shape[1], device=q.device)
        idx = indices[:, 0, : workspace.shape[1]]
        valid = (idx >= 0) & (idx < kv2d.shape[0])
        valid = valid & (pos.view(1, -1) < lens.to(torch.long).view(-1, 1))
        valid_any = valid.any(dim=-1)
        scores.masked_fill_(~valid.unsqueeze(1), float("-inf"))
        scores = torch.where(
            valid_any.view(-1, 1, 1), scores, torch.zeros_like(scores)
        )
    else:
        valid_any = None
    lse = torch.logsumexp(scores, dim=-1)
    probs = torch.softmax(scores, dim=-1).to(q.dtype)
    if valid_any is not None:
        probs.masked_fill_(~valid_any.view(-1, 1, 1), 0)
    out_bmm = torch.bmm(probs, workspace)
    if sink is not None:
        gate = torch.sigmoid(lse - sink.to(lse.dtype).view(1, -1))
        out_bmm = out_bmm * gate.to(out_bmm.dtype).unsqueeze(-1)
    out.copy_(out_bmm)
    if valid_any is not None:
        lse = lse.masked_fill(~valid_any.unsqueeze(1), float("-inf"))
    return out, torch.empty((q.shape[0], q.shape[1]), device=q.device, dtype=torch.float32), lse


def call_index_select_torch_bmm(case, scale: float):
    q, kv, indices, lens, sink, out = case
    kv2d = kv[:, 0, :]
    workspace = torch.empty(
        (q.shape[0], indices.shape[-1], kv2d.shape[-1]),
        device=q.device,
        dtype=kv2d.dtype,
    )
    torch.index_select(
        kv2d,
        0,
        indices[:, 0, :].reshape(-1),
        out=workspace.reshape(-1, kv2d.shape[-1]),
    )
    scores = torch.bmm(q, workspace.transpose(1, 2)).float().mul_(scale)
    if lens is not None:
        pos = torch.arange(workspace.shape[1], device=q.device)
        idx = indices[:, 0, : workspace.shape[1]]
        valid = (idx >= 0) & (idx < kv2d.shape[0])
        valid = valid & (pos.view(1, -1) < lens.to(torch.long).view(-1, 1))
        valid_any = valid.any(dim=-1)
        scores.masked_fill_(~valid.unsqueeze(1), float("-inf"))
        scores = torch.where(
            valid_any.view(-1, 1, 1), scores, torch.zeros_like(scores)
        )
    else:
        valid_any = None
    lse = torch.logsumexp(scores, dim=-1)
    probs = torch.softmax(scores, dim=-1).to(q.dtype)
    if valid_any is not None:
        probs.masked_fill_(~valid_any.view(-1, 1, 1), 0)
    out_bmm = torch.bmm(probs, workspace)
    if sink is not None:
        gate = torch.sigmoid(lse - sink.to(lse.dtype).view(1, -1))
        out_bmm = out_bmm * gate.to(out_bmm.dtype).unsqueeze(-1)
    out.copy_(out_bmm)
    if valid_any is not None:
        lse = lse.masked_fill(~valid_any.unsqueeze(1), float("-inf"))
    return out, torch.empty((q.shape[0], q.shape[1]), device=q.device, dtype=torch.float32), lse


def call_cached_index_select_torch_bmm(case, scale: float):
    q, kv, indices, lens, sink, out = case
    kv2d = kv[:, 0, :]
    key = (q.device.index, q.shape[0], indices.shape[-1], kv2d.shape[-1], str(kv2d.dtype))
    cache = getattr(call_cached_index_select_torch_bmm, "_cache", {})
    workspace = cache.get(key)
    if workspace is None:
        workspace = torch.empty(
            (q.shape[0], indices.shape[-1], kv2d.shape[-1]),
            device=q.device,
            dtype=kv2d.dtype,
        )
        cache[key] = workspace
        setattr(call_cached_index_select_torch_bmm, "_cache", cache)
    torch.index_select(
        kv2d,
        0,
        indices[:, 0, :].reshape(-1),
        out=workspace.reshape(-1, kv2d.shape[-1]),
    )
    scores = torch.bmm(q, workspace.transpose(1, 2)).float().mul_(scale)
    if lens is not None:
        pos_cache = getattr(call_cached_index_select_torch_bmm, "_pos_cache", {})
        pos = pos_cache.get((q.device.index, workspace.shape[1]))
        if pos is None:
            pos = torch.arange(workspace.shape[1], device=q.device)
            pos_cache[(q.device.index, workspace.shape[1])] = pos
            setattr(call_cached_index_select_torch_bmm, "_pos_cache", pos_cache)
        idx = indices[:, 0, : workspace.shape[1]]
        valid = (idx >= 0) & (idx < kv2d.shape[0])
        valid = valid & (pos.view(1, -1) < lens.to(torch.long).view(-1, 1))
        valid_any = valid.any(dim=-1)
        scores.masked_fill_(~valid.unsqueeze(1), float("-inf"))
        scores = torch.where(
            valid_any.view(-1, 1, 1), scores, torch.zeros_like(scores)
        )
    else:
        valid_any = None
    lse = torch.logsumexp(scores, dim=-1)
    probs = torch.softmax(scores, dim=-1).to(q.dtype)
    if valid_any is not None:
        probs.masked_fill_(~valid_any.view(-1, 1, 1), 0)
    out_bmm = torch.bmm(probs, workspace)
    if sink is not None:
        gate = torch.sigmoid(lse - sink.to(lse.dtype).view(1, -1))
        out_bmm = out_bmm * gate.to(out_bmm.dtype).unsqueeze(-1)
    out.copy_(out_bmm)
    if valid_any is not None:
        lse = lse.masked_fill(~valid_any.unsqueeze(1), float("-inf"))
    return out, torch.empty((q.shape[0], q.shape[1]), device=q.device, dtype=torch.float32), lse


def call_compiled_cached_index_select_torch_bmm(case, scale: float):
    q, kv, indices, lens, sink, out = case
    kv2d = kv[:, 0, :]
    key = (q.device.index, q.shape[0], indices.shape[-1], kv2d.shape[-1], str(kv2d.dtype))
    cache = getattr(call_compiled_cached_index_select_torch_bmm, "_cache", {})
    workspace = cache.get(key)
    if workspace is None:
        workspace = torch.empty(
            (q.shape[0], indices.shape[-1], kv2d.shape[-1]),
            device=q.device,
            dtype=kv2d.dtype,
        )
        cache[key] = workspace
        setattr(call_compiled_cached_index_select_torch_bmm, "_cache", cache)
    torch.index_select(
        kv2d,
        0,
        indices[:, 0, :].reshape(-1),
        out=workspace.reshape(-1, kv2d.shape[-1]),
    )
    idx = indices[:, 0, : workspace.shape[1]]
    pos_cache = getattr(call_compiled_cached_index_select_torch_bmm, "_pos_cache", {})
    pos = pos_cache.get((q.device.index, workspace.shape[1]))
    if pos is None:
        pos = torch.arange(workspace.shape[1], device=q.device)
        pos_cache[(q.device.index, workspace.shape[1])] = pos
        setattr(call_compiled_cached_index_select_torch_bmm, "_pos_cache", pos_cache)
    valid = (idx >= 0) & (idx < kv2d.shape[0])
    if lens is not None:
        valid = valid & (pos.view(1, -1) < lens.to(torch.long).view(-1, 1))
    valid_any = valid.any(dim=-1)
    out_bmm, lse = get_compiled_torch_bmm()(q, workspace, valid, valid_any, sink, scale)
    out.copy_(out_bmm)
    lse = lse.masked_fill(~valid_any.unsqueeze(1), float("-inf"))
    return out, torch.empty((q.shape[0], q.shape[1]), device=q.device, dtype=torch.float32), lse


def trusted_index_torch_bmm_breakdown(case, scale: float, warmup: int, iters: int):
    """Time the architectural pieces of the current prefill BMM bridge.

    This intentionally measures the same tensor operations used by the vLLM
    patcher rather than only the aggregate wall time.  The output guides the
    next fused/native boundary: gather/materialization, QK, mask/softmax, PV,
    or sink/output staging.
    """
    q, kv, indices, lens, sink, out = case
    kv2d = kv[:, 0, :]
    pos = torch.arange(indices.shape[-1], device=q.device)

    def run_once(record=None):
        events = {}

        def mark(name):
            if record is None:
                return
            ev = torch.cuda.Event(enable_timing=True)
            ev.record()
            events[name] = ev

        mark("start")
        workspace = kv2d[indices[:, 0, :]]
        mark("gather")
        scores = torch.bmm(q, workspace.transpose(1, 2)).float().mul_(scale)
        mark("qk_bmm")
        idx = indices[:, 0, : workspace.shape[1]]
        valid = (idx >= 0) & (idx < kv2d.shape[0])
        valid = valid & (pos.view(1, -1) < lens.to(torch.long).view(-1, 1))
        valid_any = valid.any(dim=-1)
        scores.masked_fill_(~valid.unsqueeze(1), float("-inf"))
        scores = torch.where(
            valid_any.view(-1, 1, 1), scores, torch.zeros_like(scores)
        )
        mark("mask")
        lse = torch.logsumexp(scores, dim=-1)
        probs = torch.softmax(scores, dim=-1).to(q.dtype)
        probs.masked_fill_(~valid_any.view(-1, 1, 1), 0)
        mark("softmax")
        out_bmm = torch.bmm(probs, workspace)
        mark("pv_bmm")
        gate = torch.sigmoid(lse - sink.to(lse.dtype).view(1, -1))
        out_bmm = out_bmm * gate.to(out_bmm.dtype).unsqueeze(-1)
        out.copy_(out_bmm)
        lse = lse.masked_fill(~valid_any.unsqueeze(1), float("-inf"))
        mark("sink_copy")
        if record is not None:
            return events
        return out, lse

    for _ in range(warmup):
        run_once()
    torch.cuda.synchronize()

    names = ("gather", "qk_bmm", "mask", "softmax", "pv_bmm", "sink_copy")
    totals = {name: 0.0 for name in names}
    for _ in range(iters):
        events = run_once(record=True)
        torch.cuda.synchronize()
        prev = events["start"]
        for name in names:
            totals[name] += prev.elapsed_time(events[name]) * 1000.0
            prev = events[name]
    return {name: totals[name] / iters for name in names}


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
    out_bmm, _max_bmm, lse_bmm = call_gather_torch_bmm(case, scale)
    out_trusted, _max_trusted, lse_trusted = call_trusted_index_torch_bmm(case, scale)
    out_index_select, _max_index_select, lse_index_select = call_index_select_torch_bmm(case, scale)
    out_cached_index_select, _max_cached_index_select, lse_cached_index_select = call_cached_index_select_torch_bmm(case, scale)
    out_compiled_cached, _max_compiled_cached, lse_compiled_cached = call_compiled_cached_index_select_torch_bmm(case, scale)
    torch.cuda.synchronize()

    print(
        f"shape: tokens={args.tokens} heads={args.heads} kv_tokens={args.kv_tokens} topk={args.topk}"
    )
    for name, out, lse in (
        ("split", out_split, lse_split),
        ("chunked_workspace", out_chunk, lse_chunk),
        ("chunked_workspace_gather", out_gather, lse_gather),
        ("gather_torch_bmm", out_bmm, lse_bmm),
        ("trusted_index_torch_bmm", out_trusted, lse_trusted),
        ("index_select_torch_bmm", out_index_select, lse_index_select),
        ("cached_index_select_torch_bmm", out_cached_index_select, lse_cached_index_select),
        ("compiled_cached_index_select_torch_bmm", out_compiled_cached, lse_compiled_cached),
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
    print(
        "gather_torch_bmm_us: "
        f"{bench(lambda: call_gather_torch_bmm(case, scale), args.warmup, args.iters):.3f}"
    )
    print(
        "trusted_index_torch_bmm_us: "
        f"{bench(lambda: call_trusted_index_torch_bmm(case, scale), args.warmup, args.iters):.3f}"
    )
    print(
        "index_select_torch_bmm_us: "
        f"{bench(lambda: call_index_select_torch_bmm(case, scale), args.warmup, args.iters):.3f}"
    )
    print(
        "cached_index_select_torch_bmm_us: "
        f"{bench(lambda: call_cached_index_select_torch_bmm(case, scale), args.warmup, args.iters):.3f}"
    )
    print(
        "compiled_cached_index_select_torch_bmm_us: "
        f"{bench(lambda: call_compiled_cached_index_select_torch_bmm(case, scale), args.warmup, args.iters):.3f}"
    )
    breakdown = trusted_index_torch_bmm_breakdown(
        case, scale, args.warmup, args.iters
    )
    print(
        "trusted_index_breakdown_us: "
        + " ".join(f"{name}={value:.3f}" for name, value in breakdown.items())
    )


if __name__ == "__main__":
    main()
