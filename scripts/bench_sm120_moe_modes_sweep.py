#!/usr/bin/env python3
"""Run a compact SM120 MoE body-mode sweep without loading vLLM.

The goal is to make larger-boundary MoE decisions reproducible.  It invokes
bench_sm120_moe_chain.py for a small set of routed-shape/mode combinations and
prints a table of the key FC1/FC2/chain timings.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


_METRIC_RE = re.compile(r"^(fc1_us|act_quant_us|fc2_us|reduce_us|chain_us|chain_reduce_us|graph_chain_us|graph_chain_reduce_us):\s+([0-9.]+)")


@dataclass(frozen=True)
class Shape:
    name: str
    m: int
    active_groups: int
    reduce_topk: int = 0


@dataclass(frozen=True)
class RunResult:
    shape: Shape
    mode: str
    raster: int | None
    metrics: dict[str, float]


def parse_shape(text: str) -> Shape:
    # name:m:active_groups[:reduce_topk]
    parts = text.split(":")
    if len(parts) not in (3, 4):
        raise argparse.ArgumentTypeError(
            "shape must be name:m:active_groups[:reduce_topk]"
        )
    name, m, active_groups, *rest = parts
    return Shape(name, int(m), int(active_groups), int(rest[0]) if rest else 0)


def run_one(
    bench_script: Path,
    shape: Shape,
    mode: str,
    raster: int | None,
    hidden: int,
    moe_hidden: int,
    warmup: int,
    iters: int,
    graph: bool,
    device: str,
) -> RunResult:
    env = os.environ.copy()
    if raster is not None:
        env["DG_SM120_MOE_RASTER_ORDER"] = str(raster)
    cmd = [
        sys.executable,
        str(bench_script),
        "--m",
        str(shape.m),
        "--hidden",
        str(hidden),
        "--moe-hidden",
        str(moe_hidden),
        "--active-groups",
        str(shape.active_groups),
        "--mode",
        mode,
        "--warmup",
        str(warmup),
        "--iters",
        str(iters),
        "--device",
        device,
    ]
    if shape.reduce_topk:
        cmd += ["--reduce-topk", str(shape.reduce_topk)]
    if graph:
        cmd.append("--graph")
    proc = subprocess.run(
        cmd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
    )
    metrics: dict[str, float] = {}
    for line in proc.stdout.splitlines():
        match = _METRIC_RE.match(line.strip())
        if match:
            metrics[match.group(1)] = float(match.group(2))
    if "chain_us" not in metrics:
        print(proc.stdout, file=sys.stderr)
        raise RuntimeError(f"missing chain_us for {shape.name}/{mode}/raster={raster}")
    return RunResult(shape, mode, raster, metrics)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shape",
        action="append",
        type=parse_shape,
        default=None,
        help="Shape as name:m:active_groups[:reduce_topk]. Repeatable.",
    )
    parser.add_argument(
        "--mode",
        action="append",
        default=None,
        help="Mode to test. Repeatable. Defaults to skip_fill,direct_groups.",
    )
    parser.add_argument(
        "--raster",
        action="append",
        type=int,
        default=None,
        help="Optional raster order(s) to test. Defaults to current env/no override.",
    )
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--moe-hidden", type=int, default=2048)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--graph", action="store_true")
    args = parser.parse_args()

    shapes = args.shape or [
        Shape("mtp-ish", 24, 12, 2),
        Shape("no-shrink-half", 162, 64, 0),
        Shape("no-shrink-full", 162, 128, 0),
    ]
    modes = args.mode or ["skip_fill", "direct_groups"]
    rasters: list[int | None] = args.raster if args.raster is not None else [None]

    bench_script = Path(__file__).with_name("bench_sm120_moe_chain.py")
    results: list[RunResult] = []
    for shape in shapes:
        for mode in modes:
            for raster in rasters:
                results.append(
                    run_one(
                        bench_script,
                        shape,
                        mode,
                        raster,
                        args.hidden,
                        args.moe_hidden,
                        args.warmup,
                        args.iters,
                        args.graph,
                        args.device,
                    )
                )

    headers = [
        "shape",
        "mode",
        "raster",
        "fc1_us",
        "act_us",
        "fc2_us",
        "reduce_us",
        "chain_us",
        "chain_reduce_us",
        "graph_chain_us",
        "graph_chain_reduce_us",
    ]
    print("\t".join(headers))
    for result in results:
        m = result.metrics
        row = [
            result.shape.name,
            result.mode,
            "env" if result.raster is None else str(result.raster),
            f"{m.get('fc1_us', float('nan')):.3f}",
            f"{m.get('act_quant_us', float('nan')):.3f}",
            f"{m.get('fc2_us', float('nan')):.3f}",
            "" if "reduce_us" not in m else f"{m['reduce_us']:.3f}",
            f"{m['chain_us']:.3f}",
            "" if "chain_reduce_us" not in m else f"{m['chain_reduce_us']:.3f}",
            "" if "graph_chain_us" not in m else f"{m['graph_chain_us']:.3f}",
            "" if "graph_chain_reduce_us" not in m else f"{m['graph_chain_reduce_us']:.3f}",
        ]
        print("\t".join(row))


if __name__ == "__main__":
    main()
