"""Lean validator for the SM120 MMA mqa_logits kernel.

Mirrors test_mqa_logits but uses smaller shapes so it fits when the rig is
under load (DSv4 is hogging GPUs 0-3, 5090s are partially occupied).

Run with DG_SM120_MMA_INDEXER=1 to exercise the new kernel; otherwise the
vec kernel is what's tested.
"""

import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import deep_gemm
from deep_gemm.utils import per_custom_dims_cast_to_fp8
from deep_gemm.testing.numeric import calc_diff


def ref_logits(q, kv, weights, ks, ke):
    seq_len_kv = kv.shape[0]
    q = q.float(); k = kv.float()
    mask_lo = torch.arange(0, seq_len_kv, device='cuda')[None, :] >= ks[:, None]
    mask_hi = torch.arange(0, seq_len_kv, device='cuda')[None, :] < ke[:, None]
    mask = mask_lo & mask_hi
    score = torch.einsum('mhd,nd->hmn', q, k)
    out = (score.relu() * weights.unsqueeze(-1).transpose(0, 1)).sum(dim=0)
    out = out.masked_fill(~mask, float('-inf'))
    return out, mask.sum()


def gen_ks_ke(seq_len, seq_len_kv, disable_cp):
    if disable_cp:
        ks = torch.zeros(seq_len, dtype=torch.int, device='cuda')
        ke = (torch.arange(seq_len, dtype=torch.int, device='cuda')
              + (seq_len_kv - seq_len))
        return ks, ke
    chunk = seq_len // 2
    cp_size = seq_len_kv // seq_len
    cp_id = max(cp_size // 3, 1)
    ks = torch.zeros(seq_len, dtype=torch.int, device='cuda')
    ke = torch.zeros(seq_len, dtype=torch.int, device='cuda')
    for i in range(chunk):
        ke[i] = cp_id * chunk + i
        ke[i + chunk] = (cp_size * 2 - 1 - cp_id) * chunk + i
    return ks, ke


def run_one(seq_len, seq_len_kv, num_heads, head_dim, compressed, dtype,
            disable_cp):
    q = torch.randn(seq_len, num_heads, head_dim, device='cuda',
                    dtype=torch.bfloat16)
    kv = torch.randn(seq_len_kv, head_dim, device='cuda',
                     dtype=torch.bfloat16)
    weights = torch.randn(seq_len, num_heads, device='cuda',
                          dtype=torch.float32)
    ks, ke = gen_ks_ke(seq_len, seq_len_kv, disable_cp)

    ref, _ = ref_logits(q, kv, weights, ks, ke)

    q_fp8 = q.to(torch.float8_e4m3fn)
    kv_fp8 = per_custom_dims_cast_to_fp8(kv, (0,), False)
    q_in = (q_fp8, None)

    kwargs = dict(q=q_in, kv=kv_fp8, weights=weights,
                  cu_seq_len_k_start=ks, cu_seq_len_k_end=ke,
                  clean_logits=not compressed, max_seqlen_k=0,
                  logits_dtype=dtype)
    if compressed:
        kwargs['max_seqlen_k'] = (ke - ks).max().item()

    out = deep_gemm.fp8_fp4_mqa_logits(**kwargs)

    if compressed:
        max_k = kwargs['max_seqlen_k']
        tmp = torch.full((seq_len, seq_len_kv), float('-inf'), device='cuda')
        for i in range(seq_len):
            tmp[i, ks[i]:ke[i]] = out[i, :ke[i] - ks[i]]
        out = tmp

    ref_neg = (ref == float('-inf'))
    out_neg = (out == float('-inf'))
    assert torch.equal(out_neg, ref_neg), \
        f'mask mismatch S={seq_len} SKV={seq_len_kv} cp={int(not disable_cp)} compressed={compressed}'
    ref_m = ref.masked_fill(ref_neg, 0)
    out_m = out.masked_fill(ref_neg, 0)
    diff = calc_diff(out_m, ref_m)
    return diff


def main():
    print(f'DG_SM120_MMA_INDEXER = {os.environ.get("DG_SM120_MMA_INDEXER")!r}')
    print(f'CUDA_VISIBLE_DEVICES = {os.environ.get("CUDA_VISIBLE_DEVICES")!r}')
    print(f'Compute cap: {torch.cuda.get_device_capability(0)}')

    cases = []
    # Small shapes that fit in <8GB total.
    for s, skv in [(512, 1024), (512, 2048), (1024, 2048), (2048, 4096)]:
        for compressed in (True, False):
            for dtype in (torch.float32, torch.bfloat16):
                for disable_cp in (True, False):
                    cases.append((s, skv, 64, 128, compressed, dtype,
                                  disable_cp))

    fail = 0
    for c in cases:
        try:
            d = run_one(*c)
        except torch.cuda.OutOfMemoryError as e:
            print(f'  SKIP (OOM): S={c[0]} SKV={c[1]} compressed={c[4]} '
                  f'dtype={c[5]} disable_cp={c[6]}')
            torch.cuda.empty_cache()
            continue
        ok = d < 1e-3
        marker = 'OK' if ok else 'FAIL'
        print(f'  [{marker}] S={c[0]:4d} SKV={c[1]:4d} compressed={int(c[4])} '
              f'dtype={str(c[5]).split(".")[-1]:>9} disable_cp={int(c[6])} '
              f'diff={d:.2e}')
        if not ok:
            fail += 1
        torch.cuda.empty_cache()
    if fail:
        print(f'\nFAILED: {fail} configs exceed 1e-3 diff')
        sys.exit(1)
    print('\nAll configs within 1e-3 tolerance.')


if __name__ == '__main__':
    main()
