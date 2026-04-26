#!/usr/bin/env python3
"""Streaming DeepSeek V4 Flash benchmark for TTFT and steady decode rate.

The existing shell benchmark is intentionally non-streaming, so its tok/s folds
prefill/TTFT into generation. This script measures OpenAI-compatible streaming
chat completions and reports both end-to-end request throughput and warmed
steady-state decode throughput.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


@dataclass
class RequestResult:
    index: int
    ok: bool
    completion_tokens: int
    prompt_tokens: int | None
    token_events: int
    elapsed_s: float
    first_token_offset_s: float | None
    end_offset_s: float
    ttft_s: float | None
    steady_decode_s: float | None
    steady_decode_tok_s: float | None
    request_tok_s: float
    error: str | None = None


def env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return float("nan")
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    rank = (len(ordered) - 1) * pct / 100.0
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    frac = rank - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def tokenize_count(args: argparse.Namespace, prompt: str) -> int | None:
    url = args.base_url.rstrip("/") + "/tokenize"
    body = json.dumps({"model": args.model, "prompt": prompt}).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=args.timeout_s) as response:
            payload = json.load(response)
            count = payload.get("count")
            return int(count) if count is not None else None
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError, ValueError):
        return None


def build_token_target_prompt(args: argparse.Namespace, index: int) -> str:
    """Build a prompt close to an actual tokenizer count.

    The previous prompt generator used long hex-like words, which overshot the
    requested token count badly for DeepSeek's tokenizer.  Use vLLM's /tokenize
    endpoint and a binary search over common one-ish-token words so long-context
    TTFT tests request roughly the intended prompt size instead of accidentally
    exercising much larger contexts.
    """
    prefix = (
        f"nonce-{time.time_ns()}-{index}: "
        if args.unique_prompts
        else ""
    )
    intro = (
        prefix
        + "Summarize the following synthetic engineering notes in one paragraph.\n"
    )
    outro = "\nEnd of notes."
    target = max(args.prompt_token_target, 1)
    unit = " token"

    def candidate(units: int) -> str:
        return intro + (unit * units) + outro

    # Fast path if /tokenize is unavailable: common space-prefixed words are
    # close to one token each, unlike hex strings.
    base_count = tokenize_count(args, intro + outro)
    if base_count is None:
        return candidate(max(1, target - 16))

    lo, hi = 0, max(1, target - base_count + 16)
    while True:
        count = tokenize_count(args, candidate(hi))
        if count is None or count >= target or hi >= target * 4:
            break
        hi *= 2

    best = hi
    while lo <= hi:
        mid = (lo + hi) // 2
        count = tokenize_count(args, candidate(mid))
        if count is None:
            break
        if count < target:
            lo = mid + 1
        else:
            best = mid
            hi = mid - 1
    return candidate(best)


def build_payload(args: argparse.Namespace, index: int) -> dict[str, Any]:
    prompt = args.prompt
    if args.prompt_token_target > 0:
        prompt = build_token_target_prompt(args, index)
    if args.unique_prompts:
        # Change the prompt from byte zero enough to avoid prefix-cache artifacts
        # in TTFT/prefill measurements, while keeping it short for decode tests.
        if args.prompt_token_target <= 0:
            prompt = f"nonce-{time.time_ns()}-{index}: {prompt}"
    payload: dict[str, Any] = {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if args.min_tokens is not None:
        payload["min_tokens"] = args.min_tokens
    if args.ignore_eos:
        payload["ignore_eos"] = True
    return payload


def run_one(args: argparse.Namespace, index: int, barrier: threading.Barrier | None) -> RequestResult:
    if barrier is not None:
        barrier.wait()
    url = args.base_url.rstrip("/") + "/v1/chat/completions"
    body = json.dumps(build_payload(args, index)).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    start = time.perf_counter()
    token_times: list[float] = []
    completion_tokens: int | None = None
    prompt_tokens: int | None = None
    token_events = 0
    try:
        with urllib.request.urlopen(request, timeout=args.timeout_s) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                usage = chunk.get("usage")
                if usage:
                    if usage.get("completion_tokens") is not None:
                        completion_tokens = int(usage["completion_tokens"])
                    if usage.get("prompt_tokens") is not None:
                        prompt_tokens = int(usage["prompt_tokens"])
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                # vLLM streams one event per generated token for normal chat
                # completions. Count only content-bearing chunks; final usage and
                # role-only chunks are excluded.
                if delta.get("content"):
                    token_events += 1
                    token_times.append(time.perf_counter())
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        elapsed = time.perf_counter() - start
        return RequestResult(
            index=index,
            ok=False,
            completion_tokens=0,
            prompt_tokens=prompt_tokens,
            token_events=token_events,
            elapsed_s=elapsed,
            first_token_offset_s=None,
            end_offset_s=elapsed,
            ttft_s=None,
            steady_decode_s=None,
            steady_decode_tok_s=None,
            request_tok_s=0.0,
            error=str(exc),
        )

    end = time.perf_counter()
    elapsed = end - start
    if completion_tokens is None:
        # If the server does not emit streaming usage, ignore_eos+min=max means
        # max_tokens is the exact intended token count. Otherwise content events
        # are the best available estimate.
        completion_tokens = args.max_tokens if args.ignore_eos else token_events
    if token_events <= 0 and completion_tokens > 0:
        # Some servers can coalesce content chunks. Do not invent a TTFT, but keep
        # request tok/s useful.
        token_events = completion_tokens

    ttft_s = token_times[0] - start if token_times else None
    first_token_offset_s = ttft_s
    steady_decode_s: float | None = None
    steady_decode_tok_s: float | None = None
    if token_times:
        # Use response end rather than last content-chunk timestamp. Some OpenAI
        # streaming servers coalesce multiple generated tokens into one content
        # event, so content-event count/timing is not a reliable token clock.
        steady_decode_s = end - token_times[0]
        steady_tokens = max(completion_tokens - 1, 1)
        steady_decode_tok_s = steady_tokens / steady_decode_s if steady_decode_s > 0 else None
    request_tok_s = completion_tokens / elapsed if elapsed > 0 else 0.0
    return RequestResult(
        index=index,
        ok=True,
        completion_tokens=completion_tokens,
        prompt_tokens=prompt_tokens,
        token_events=token_events,
        elapsed_s=elapsed,
        first_token_offset_s=first_token_offset_s,
        end_offset_s=elapsed,
        ttft_s=ttft_s,
        steady_decode_s=steady_decode_s,
        steady_decode_tok_s=steady_decode_tok_s,
        request_tok_s=request_tok_s,
    )


def summarize(results: list[RequestResult], wall_s: float) -> int:
    failures = [r for r in results if not r.ok]
    successes = [r for r in results if r.ok]
    for r in sorted(results, key=lambda x: x.index):
        if r.ok:
            ttft = "nan" if r.ttft_s is None else f"{r.ttft_s:.4f}"
            steady = "nan" if r.steady_decode_tok_s is None else f"{r.steady_decode_tok_s:.2f}"
            prompt_tok = "nan" if r.prompt_tokens is None else str(r.prompt_tokens)
            print(
                f"request[{r.index}]: prompt_tokens={prompt_tok} tokens={r.completion_tokens} "
                f"events={r.token_events} elapsed_s={r.elapsed_s:.4f} "
                f"ttft_s={ttft} request_tok_s={r.request_tok_s:.2f} "
                f"steady_decode_tok_s={steady}"
            )
        else:
            print(f"request[{r.index}]: ERROR after {r.elapsed_s:.4f}s: {r.error}", file=sys.stderr)

    total_tokens = sum(r.completion_tokens for r in successes)
    prompt_token_values = [r.prompt_tokens for r in successes if r.prompt_tokens is not None]
    request_rates = [r.request_tok_s for r in successes]
    ttfts = [r.ttft_s for r in successes if r.ttft_s is not None]
    steady_rates = [r.steady_decode_tok_s for r in successes if r.steady_decode_tok_s is not None]
    steady_tokens = sum(max(r.completion_tokens - 1, 0) for r in successes if r.steady_decode_tok_s is not None)
    first_offsets = [r.first_token_offset_s for r in successes if r.first_token_offset_s is not None]
    if first_offsets and successes:
        steady_wall = max(r.end_offset_s for r in successes) - min(first_offsets)
    else:
        steady_wall = 0.0

    print("summary:")
    print(f"  requests: {len(results)}")
    print(f"  successes: {len(successes)}")
    print(f"  failures: {len(failures)}")
    print(f"  completion_tokens_total: {total_tokens}")
    if prompt_token_values:
        print(f"  prompt_tokens_total: {sum(prompt_token_values)}")
        print(f"  mean_prompt_tokens: {statistics.fmean(prompt_token_values):.1f}")
    print(f"  wall_seconds: {wall_s:.4f}")
    print(f"  aggregate_request_tok_s: {(total_tokens / wall_s) if wall_s > 0 else 0.0:.2f}")
    if request_rates:
        print(f"  mean_request_tok_s: {statistics.fmean(request_rates):.2f}")
        print(f"  min_request_tok_s: {min(request_rates):.2f}")
        print(f"  max_request_tok_s: {max(request_rates):.2f}")
    if ttfts:
        print(f"  mean_ttft_s: {statistics.fmean(ttfts):.4f}")
        print(f"  p50_ttft_s: {percentile(ttfts, 50):.4f}")
        print(f"  p90_ttft_s: {percentile(ttfts, 90):.4f}")
        print(f"  p99_ttft_s: {percentile(ttfts, 99):.4f}")
        if prompt_token_values:
            prefill_rates = [
                r.prompt_tokens / r.ttft_s
                for r in successes
                if r.prompt_tokens is not None and r.ttft_s is not None and r.ttft_s > 0
            ]
            if prefill_rates:
                print(f"  mean_prompt_tokens_per_ttft_s: {statistics.fmean(prefill_rates):.2f}")
    if steady_rates:
        print(f"  mean_steady_decode_tok_s: {statistics.fmean(steady_rates):.2f}")
        print(f"  min_steady_decode_tok_s: {min(steady_rates):.2f}")
        print(f"  max_steady_decode_tok_s: {max(steady_rates):.2f}")
        if steady_wall > 0:
            print(f"  aggregate_steady_decode_tok_s: {steady_tokens / steady_wall:.2f}")
    return 1 if failures else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=os.environ.get("BASE_URL", "http://127.0.0.1:8080"))
    parser.add_argument("--model", default=os.environ.get("MODEL", "deepseek-ai/DeepSeek-V4-Flash"))
    parser.add_argument("--prompt", default=os.environ.get("PROMPT", "Hello, how are you?"))
    parser.add_argument("--prompt-token-target", type=int, default=int(os.environ.get("PROMPT_TOKEN_TARGET", "0")), help="Generate an approximate unique prompt of this many word-like units; usage.prompt_tokens is reported as truth")
    parser.add_argument("--max-tokens", type=int, default=int(os.environ.get("MAX_TOKENS", "512")))
    parser.add_argument("--min-tokens", type=int, default=int(os.environ.get("MIN_TOKENS", os.environ.get("MAX_TOKENS", "512"))))
    parser.add_argument("--temperature", type=float, default=float(os.environ.get("TEMPERATURE", "0")))
    parser.add_argument("--concurrency", type=int, default=int(os.environ.get("CONCURRENCY", "1")))
    parser.add_argument("--timeout-s", type=float, default=float(os.environ.get("TIMEOUT_S", "900")))
    parser.add_argument("--ignore-eos", action=argparse.BooleanOptionalAction, default=env_bool("IGNORE_EOS", True))
    parser.add_argument("--unique-prompts", action=argparse.BooleanOptionalAction, default=env_bool("UNIQUE_PROMPTS", True))
    parser.add_argument("--start-barrier", action=argparse.BooleanOptionalAction, default=env_bool("START_BARRIER", True))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print(f"Streaming benchmark {args.model} at {args.base_url}")
    print(
        f"concurrency={args.concurrency} max_tokens={args.max_tokens} "
        f"min_tokens={args.min_tokens} ignore_eos={args.ignore_eos} unique_prompts={args.unique_prompts}"
    )
    barrier = threading.Barrier(args.concurrency) if args.start_barrier and args.concurrency > 1 else None
    start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [pool.submit(run_one, args, i, barrier) for i in range(args.concurrency)]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]
    wall_s = time.perf_counter() - start
    return summarize(results, wall_s)


if __name__ == "__main__":
    raise SystemExit(main())
