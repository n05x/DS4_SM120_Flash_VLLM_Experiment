#!/usr/bin/env python3
"""Compute a rough DeepSeek-V4 Flash SM120 decode latency budget.

This script turns the latest measured API throughput and MoE microbench numbers
into a per-token/per-layer budget.  It is intentionally simple: it does not try
to model overlap or CUDA graph details, but it makes clear which measured kernel
families are large enough to matter for the remaining gap to 100 tok/s.
"""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=int, default=43)
    parser.add_argument("--current-tok-s", type=float, default=91.45)
    parser.add_argument("--target-tok-s", type=float, default=100.0)
    parser.add_argument("--moe-fc1-us", type=float, default=96.541)
    parser.add_argument("--moe-fc2-us", type=float, default=59.260)
    parser.add_argument("--dense-fp8-us", type=float, default=43.0)
    parser.add_argument("--bhr-us", type=float, default=32.0)
    args = parser.parse_args()

    current_ms = 1000.0 / args.current_tok_s
    target_ms = 1000.0 / args.target_tok_s
    needed_ms = current_ms - target_ms
    per_layer_needed_us = needed_ms * 1000.0 / args.layers
    moe_us = args.moe_fc1_us + args.moe_fc2_us
    moe_ms_all_layers = moe_us * args.layers / 1000.0

    print(f"layers: {args.layers}")
    print(f"current_tok_s: {args.current_tok_s:.3f}")
    print(f"target_tok_s: {args.target_tok_s:.3f}")
    print(f"current_ms_per_token: {current_ms:.3f}")
    print(f"target_ms_per_token: {target_ms:.3f}")
    print(f"needed_savings_ms_per_token: {needed_ms:.3f}")
    print(f"needed_savings_us_per_layer: {per_layer_needed_us:.3f}")
    print(f"measured_moe_fc1_plus_fc2_us_per_layer: {moe_us:.3f}")
    print(f"measured_moe_fc1_plus_fc2_ms_all_layers: {moe_ms_all_layers:.3f}")
    print(
        "moe_10pct_savings_ms_all_layers: "
        f"{moe_ms_all_layers * 0.10:.3f}"
    )
    print(
        "moe_15pct_savings_ms_all_layers: "
        f"{moe_ms_all_layers * 0.15:.3f}"
    )
    print(
        "interpretation: helper tweaks worth only a few us/layer cannot close "
        "the gap; a 10-15% MoE-body or larger-boundary fusion win is large "
        "enough to matter for 100 tok/s, especially if it also reduces fixed "
        "per-layer overhead."
    )


if __name__ == "__main__":
    main()
