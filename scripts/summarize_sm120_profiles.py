#!/usr/bin/env python3
"""Summarize SM120 profiling artifacts.

Inputs supported:
  * DG_SM120_KERNEL_PROFILE TSV lines emitted by csrc/sm120_profile.hpp
  * PyTorch/vLLM profiler trace JSON files with Chrome trace events

The goal is to make the next kernel iteration evidence-driven without loading
the full vLLM model just to answer "what was hot in the last profile?".
"""

from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path
from statistics import mean, median


DG_TSV_RE = re.compile(
    r"^(?P<name>\S+)\tcalls=(?P<calls>\d+)\tavg_ms=(?P<avg>[0-9.]+)"
    r"\tmax_ms=(?P<max>[0-9.]+)\tlast_ms=(?P<last>[0-9.]+)"
    r"\tm=(?P<m>\d+)\tn=(?P<n>\d+)\tk=(?P<k>\d+)\tgroups=(?P<groups>\d+)"
)


def shorten_kernel_name(name: str, limit: int = 180) -> str:
    name = name.replace("void ", "").replace("class ", "").replace("struct ", "")
    name = re.sub(r"\s+", " ", name)
    return name if len(name) <= limit else name[: limit - 1] + "…"


def summarize_dg_tsv(path: Path, top: int) -> None:
    by_shape: dict[tuple[str, int, int, int, int], list[tuple[int, float, float, float]]] = (
        collections.defaultdict(list)
    )
    for line in path.read_text(errors="replace").splitlines():
        match = DG_TSV_RE.match(line.strip())
        if not match:
            continue
        key = (
            match.group("name"),
            int(match.group("m")),
            int(match.group("n")),
            int(match.group("k")),
            int(match.group("groups")),
        )
        by_shape[key].append(
            (
                int(match.group("calls")),
                float(match.group("avg")),
                float(match.group("max")),
                float(match.group("last")),
            )
        )

    print(f"\n== DG TSV: {path} ==")
    print(f"matched_records={sum(len(v) for v in by_shape.values())} shapes={len(by_shape)}")
    rows = []
    for key, samples in by_shape.items():
        last_ms = [sample[3] for sample in samples]
        final_calls, final_avg, final_max, final_last = samples[-1]
        rows.append(
            (
                mean(last_ms),
                key,
                len(samples),
                median(last_ms),
                min(last_ms),
                max(last_ms),
                final_calls,
                final_avg,
                final_max,
                final_last,
            )
        )

    for (
        last_mean,
        (name, m, n, k, groups),
        sample_count,
        last_median,
        last_min,
        last_max,
        final_calls,
        final_avg,
        final_max,
        final_last,
    ) in sorted(rows, reverse=True)[:top]:
        print(
            f"{last_mean:8.3f}ms last_mean "
            f"med={last_median:7.3f} min={last_min:7.3f} max={last_max:7.3f} "
            f"samples={sample_count:4d} calls={final_calls:7d} "
            f"final_avg={final_avg:7.3f} final_max={final_max:7.3f} "
            f"shape=m{m} n{n} k{k} g{groups} {name}"
        )


def summarize_trace_json(path: Path, top: int) -> None:
    data = json.load(path.open())
    agg: dict[str, list[float]] = collections.defaultdict(lambda: [0.0, 0.0, 0.0])
    for event in data.get("traceEvents", []):
        if event.get("cat") != "kernel" or event.get("ph") != "X":
            continue
        duration_us = float(event.get("dur", 0.0))
        row = agg[str(event.get("name", ""))]
        row[0] += duration_us
        row[1] += 1
        row[2] = max(row[2], duration_us)

    print(f"\n== Trace JSON kernels: {path} ==")
    print(f"kernels={len(agg)}")
    for name, (total_us, count, max_us) in sorted(
        agg.items(), key=lambda item: item[1][0], reverse=True
    )[:top]:
        print(
            f"{total_us / 1000:10.3f}ms total "
            f"count={int(count):7d} avg={total_us / count:8.3f}us "
            f"max={max_us:8.3f}us {shorten_kernel_name(name)}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("profiles", nargs="+", type=Path)
    parser.add_argument("--top", type=int, default=40)
    args = parser.parse_args()

    for path in args.profiles:
        if not path.exists():
            raise SystemExit(f"missing profile: {path}")
        if path.suffix == ".json":
            summarize_trace_json(path, args.top)
        else:
            summarize_dg_tsv(path, args.top)


if __name__ == "__main__":
    main()
