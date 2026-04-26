#!/usr/bin/env python3
"""Benchmark SM120 graph-native DeepSeek-V4 metadata builders."""

import argparse
import time

import numpy as np
import torch

import deep_gemm


def bench(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1e6 / iters


def parse_lens(spec: str) -> list[int]:
    return [int(x) for x in spec.replace(",", " ").split() if x]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lens", default="1,1,1,1", help="comma/space separated query lengths")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=500)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--window-size", type=int, default=1024)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--prefix-len", type=int, default=2048)
    parser.add_argument(
        "--prefill-num-decodes",
        type=int,
        default=0,
        help="number of leading requests to treat as decodes for prefill metadata",
    )
    args = parser.parse_args()

    lens = parse_lens(args.lens)
    starts = [0]
    for length in lens:
        starts.append(starts[-1] + length)

    q_cpu = torch.tensor(starts, dtype=torch.int32)
    q_gpu = q_cpu.to(args.device)
    out = torch.empty((starts[-1],), device=args.device, dtype=torch.int32)
    ref = torch.repeat_interleave(
        torch.arange(len(lens), device=args.device, dtype=torch.int32),
        torch.tensor(lens, device=args.device),
    )

    block_table_seed = torch.tensor(
        [[0, -1, 3, -2], [5, 6, -7, 8]], device=args.device, dtype=torch.int32
    )
    block_table = block_table_seed.clone()
    slot_mapping = torch.arange(starts[-1], device=args.device, dtype=torch.int64)
    if slot_mapping.numel() > 1:
        slot_mapping[1::7] = -1
    is_valid = torch.empty((starts[-1],), device=args.device, dtype=torch.bool)
    decode_swa_lens = torch.ones((max(starts[-1], 1) + 32,), device=args.device, dtype=torch.int32)
    num_decode_tokens = min(len(lens), decode_swa_lens.numel())
    max_blocks = (args.prefix_len + max(lens) + args.block_size - 1) // args.block_size + 2
    seq_lens = torch.tensor(
        [args.prefix_len + length for length in lens], device=args.device, dtype=torch.int32
    )
    swa_block_table = (
        torch.arange(len(lens) * max_blocks, device=args.device, dtype=torch.int32)
        .reshape(len(lens), max_blocks)
    )
    decode_swa_indices = torch.full(
        (max(num_decode_tokens, 1), args.window_size),
        -1,
        device=args.device,
        dtype=torch.int32,
    )
    prefill_num_decodes = max(0, min(args.prefill_num_decodes, len(lens)))
    num_prefills = len(lens) - prefill_num_decodes
    prefill_gather_lens = torch.empty((max(num_prefills, 1),), device=args.device, dtype=torch.int32)

    deep_gemm._C.sm120_fill_token_to_req_indices(out, q_gpu, len(lens))
    torch.cuda.synchronize()
    if not torch.equal(out, ref):
        raise SystemExit("SM120 metadata fill mismatch")

    block_table = block_table_seed.clone()
    deep_gemm._C.sm120_build_compressor_metadata(out, q_gpu, block_table, len(lens))
    torch.cuda.synchronize()
    if not torch.equal(out, ref):
        raise SystemExit("SM120 fused metadata token fill mismatch")
    if int(block_table.min().item()) < 0:
        raise SystemExit("SM120 fused metadata block clamp mismatch")

    decode_swa_lens.fill_(1)
    deep_gemm._C.sm120_build_sparse_swa_metadata(
        out, is_valid, q_gpu, slot_mapping, decode_swa_lens, len(lens), num_decode_tokens
    )
    torch.cuda.synchronize()
    if not torch.equal(out, ref):
        raise SystemExit("SM120 sparse SWA token fill mismatch")
    if not torch.equal(is_valid, slot_mapping >= 0):
        raise SystemExit("SM120 sparse SWA valid-token mismatch")
    if int(decode_swa_lens[num_decode_tokens:].sum().item()) != 0:
        raise SystemExit("SM120 sparse SWA decode-lens tail clear mismatch")

    decode_swa_lens.fill_(1)
    decode_swa_indices.fill_(-2)
    deep_gemm._C.sm120_build_sparse_swa_decode_metadata(
        out,
        is_valid,
        q_gpu,
        slot_mapping,
        decode_swa_lens,
        decode_swa_indices,
        seq_lens,
        swa_block_table,
        len(lens),
        num_decode_tokens,
        args.window_size,
        args.block_size,
    )
    torch.cuda.synchronize()
    if not torch.equal(out, ref):
        raise SystemExit("SM120 fused sparse SWA decode token fill mismatch")
    if not torch.equal(is_valid, slot_mapping >= 0):
        raise SystemExit("SM120 fused sparse SWA decode valid-token mismatch")
    if int(decode_swa_lens[num_decode_tokens:].sum().item()) != 0:
        raise SystemExit("SM120 fused sparse SWA decode-lens tail clear mismatch")

    ref_lens = torch.zeros_like(decode_swa_lens[:num_decode_tokens]).cpu()
    ref_indices = torch.full_like(decode_swa_indices[:num_decode_tokens], -1).cpu()
    slot_cpu = slot_mapping.cpu()
    q_cpu_i64 = q_cpu.to(torch.int64)
    seq_cpu = seq_lens.cpu().to(torch.int64)
    block_cpu = swa_block_table.cpu().to(torch.int64)
    for token_idx in range(num_decode_tokens):
        if int(slot_cpu[token_idx]) < 0:
            continue
        req_idx = int(torch.searchsorted(q_cpu_i64, token_idx, right=True).item()) - 1
        query_start = int(q_cpu_i64[req_idx])
        query_end = int(q_cpu_i64[req_idx + 1])
        query_len = query_end - query_start
        prefix_len = int(seq_cpu[req_idx]) - query_len
        pos = prefix_len + token_idx - query_start
        start_pos = max(pos - args.window_size + 1, 0)
        end_pos = pos + 1
        swa_len = min(end_pos - start_pos, args.window_size)
        ref_lens[token_idx] = swa_len
        for offset in range(swa_len):
            pos_offset = start_pos + offset
            block_in_seq = pos_offset // args.block_size
            block_offset = pos_offset % args.block_size
            ref_indices[token_idx, offset] = (
                int(block_cpu[req_idx, block_in_seq]) * args.block_size + block_offset
            )
    if not torch.equal(decode_swa_lens[:num_decode_tokens].cpu(), ref_lens):
        raise SystemExit("SM120 fused sparse SWA decode lens mismatch")
    if not torch.equal(decode_swa_indices[:num_decode_tokens].cpu(), ref_indices):
        raise SystemExit("SM120 fused sparse SWA decode indices mismatch")

    if num_prefills > 0:
        deep_gemm._C.sm120_build_sparse_swa_prefill_metadata(
            prefill_gather_lens,
            seq_lens,
            q_gpu,
            num_prefills,
            prefill_num_decodes,
            args.window_size,
        )
        torch.cuda.synchronize()
        ref_prefill_lens = []
        for prefill_idx in range(num_prefills):
            req_idx = prefill_num_decodes + prefill_idx
            query_len = starts[req_idx + 1] - starts[req_idx]
            prefix_len = int(seq_lens[req_idx].item()) - query_len
            ref_prefill_lens.append(query_len + min(max(prefix_len, 0), args.window_size - 1))
        ref_prefill = torch.tensor(ref_prefill_lens, dtype=torch.int32, device=args.device)
        if not torch.equal(prefill_gather_lens[:num_prefills], ref_prefill):
            raise SystemExit("SM120 fused sparse SWA prefill metadata mismatch")

    def sm120_fill_call() -> None:
        deep_gemm._C.sm120_fill_token_to_req_indices(out, q_gpu, len(lens))

    def sm120_fused_call() -> None:
        deep_gemm._C.sm120_build_compressor_metadata(out, q_gpu, block_table, len(lens))

    def cpu_repeat_copy_clamp() -> None:
        x = torch.repeat_interleave(torch.arange(len(lens)), q_cpu[1:] - q_cpu[:-1]).pin_memory()
        out.copy_(x, non_blocking=True)
        block_table.clamp_(min=0)

    def cpu_np_repeat_copy() -> None:
        starts_np = np.asarray(starts, dtype=np.int32)
        seg_lengths = np.diff(starts_np)
        x_np = np.repeat(np.arange(len(lens), dtype=np.int32), seg_lengths)
        out.copy_(torch.from_numpy(x_np), non_blocking=True)

    def sm120_sparse_swa_call() -> None:
        deep_gemm._C.sm120_build_sparse_swa_metadata(
            out, is_valid, q_gpu, slot_mapping, decode_swa_lens, len(lens), num_decode_tokens
        )

    def sm120_sparse_swa_decode_call() -> None:
        deep_gemm._C.sm120_build_sparse_swa_decode_metadata(
            out,
            is_valid,
            q_gpu,
            slot_mapping,
            decode_swa_lens,
            decode_swa_indices,
            seq_lens,
            swa_block_table,
            len(lens),
            num_decode_tokens,
            args.window_size,
            args.block_size,
        )

    def sm120_sparse_swa_prefill_call() -> None:
        if num_prefills > 0:
            deep_gemm._C.sm120_build_sparse_swa_prefill_metadata(
                prefill_gather_lens,
                seq_lens,
                q_gpu,
                num_prefills,
                prefill_num_decodes,
                args.window_size,
            )

    def torch_sparse_swa_prefill() -> None:
        if num_prefills > 0:
            req_starts = q_gpu[prefill_num_decodes : prefill_num_decodes + num_prefills]
            req_ends = q_gpu[prefill_num_decodes + 1 : prefill_num_decodes + num_prefills + 1]
            query_lens = req_ends - req_starts
            prefix_lens = seq_lens[prefill_num_decodes : prefill_num_decodes + num_prefills] - query_lens
            prefill_gather_lens[:num_prefills].copy_(
                query_lens + torch.minimum(
                    torch.clamp(prefix_lens, min=0),
                    torch.full_like(prefix_lens, args.window_size - 1),
                )
            )

    def cpu_sparse_swa() -> None:
        x = torch.repeat_interleave(torch.arange(len(lens)), q_cpu[1:] - q_cpu[:-1]).pin_memory()
        out.copy_(x, non_blocking=True)
        is_valid.copy_(slot_mapping >= 0)
        decode_swa_lens[num_decode_tokens:] = 0

    sm120_fill_us = bench(sm120_fill_call, args.warmup, args.iters)
    sm120_fused_us = bench(sm120_fused_call, args.warmup, args.iters)
    sm120_sparse_swa_us = bench(sm120_sparse_swa_call, args.warmup, args.iters)
    sm120_sparse_swa_decode_us = bench(
        sm120_sparse_swa_decode_call, args.warmup, args.iters
    )
    sm120_sparse_swa_prefill_us = bench(
        sm120_sparse_swa_prefill_call, args.warmup, args.iters
    )
    torch_sparse_swa_prefill_us = bench(
        torch_sparse_swa_prefill, max(1, args.warmup // 5), max(1, args.iters // 5)
    )
    cpu_np_repeat_copy_us = bench(
        cpu_np_repeat_copy, max(1, args.warmup // 5), max(1, args.iters // 5)
    )
    cpu_us = bench(cpu_repeat_copy_clamp, max(1, args.warmup // 5), max(1, args.iters // 5))
    cpu_sparse_swa_us = bench(
        cpu_sparse_swa, max(1, args.warmup // 5), max(1, args.iters // 5)
    )
    print(
        f"lens={lens} tokens={starts[-1]} requests={len(lens)}\n"
        f"sm120_device_fill_us={sm120_fill_us:.3f}\n"
        f"sm120_fused_fill_clamp_us={sm120_fused_us:.3f}\n"
        f"cpu_np_repeat_copy_us={cpu_np_repeat_copy_us:.3f}\n"
        f"req_id_speedup={cpu_np_repeat_copy_us / sm120_fill_us:.3f}x\n"
        f"cpu_repeat_pin_copy_plus_clamp_us={cpu_us:.3f}\n"
        f"fused_speedup={cpu_us / sm120_fused_us:.3f}x\n"
        f"sm120_sparse_swa_metadata_us={sm120_sparse_swa_us:.3f}\n"
        f"sm120_sparse_swa_decode_metadata_us={sm120_sparse_swa_decode_us:.3f}\n"
        f"sm120_sparse_swa_prefill_metadata_us={sm120_sparse_swa_prefill_us:.3f}\n"
        f"torch_sparse_swa_prefill_metadata_us={torch_sparse_swa_prefill_us:.3f}\n"
        f"cpu_sparse_swa_metadata_us={cpu_sparse_swa_us:.3f}\n"
        f"sparse_swa_speedup={cpu_sparse_swa_us / sm120_sparse_swa_us:.3f}x"
    )


if __name__ == "__main__":
    main()
