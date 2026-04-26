#!/usr/bin/env bash
set -euo pipefail

BASE_URL=${BASE_URL:-http://127.0.0.1:8080}
CONCURRENCY_LIST=${CONCURRENCY_LIST:-"1 2 4 8"}
MAX_TOKENS=${MAX_TOKENS:-512}
MIN_TOKENS=${MIN_TOKENS:-$MAX_TOKENS}
PROMPT=${PROMPT:-"Write a concise paragraph about GPU kernel optimization."}

for c in ${CONCURRENCY_LIST}; do
  echo "== streaming concurrency ${c} =="
  BASE_URL=${BASE_URL} \
  CONCURRENCY=${c} \
  MAX_TOKENS=${MAX_TOKENS} \
  MIN_TOKENS=${MIN_TOKENS} \
  PROMPT=${PROMPT} \
  python3 scripts/bench_deepseek_v4_flash_streaming.py
  echo
done
