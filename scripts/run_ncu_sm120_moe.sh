#!/usr/bin/env bash
# Targeted Nsight Compute runner for the SM120 FP8xFP4 MoE grouped GEMM.
set -euo pipefail

SERVICE=${SERVICE:-vllm}
OUT_DIR=${OUT_DIR:-profiles/ncu-sm120-moe}
M_LIST=${M_LIST:-"6 16 24"}
SHAPES=${SHAPES:-"fc1:4096:4096 fc2:4096:2048 up:7168:4096 down:4096:7168"}
MODE=${MODE:-row_grouped_skip_fill}
WARMUP=${WARMUP:-5}
ITERS=${ITERS:-1}
NUM_GROUPS=${NUM_GROUPS:-128}
KERNEL_NAME=${KERNEL_NAME:-regex:.*(device_kernel|sm120|cutlass|fp8|fp4).*}
SECTIONS=${SECTIONS:-"SpeedOfLight MemoryWorkloadAnalysis SourceCounters"}
NCU=${NCU:-ncu}
RUN_IN_DOCKER=${RUN_IN_DOCKER:-1}

mkdir -p "${OUT_DIR}"

run_cmd() {
  if [[ "${RUN_IN_DOCKER}" == "1" ]]; then
    docker compose exec -T "${SERVICE}" bash -lc "cd /workspace/DeepGEMM && $*"
  else
    bash -lc "$*"
  fi
}

if [[ "${RUN_IN_DOCKER}" == "1" ]]; then
  if ! docker compose exec -T "${SERVICE}" bash -lc "command -v ${NCU} >/dev/null"; then
    echo "ERROR: ${NCU} is not available inside service ${SERVICE}." >&2
    echo "Rebuild the image after Dockerfile.vllm-nightly-sm120 installs cuda-nsight-compute-13-0, or set RUN_IN_DOCKER=0 in an environment with DeepGEMM installed." >&2
    exit 127
  fi
elif ! command -v "${NCU}" >/dev/null; then
  echo "ERROR: ${NCU} is not available on PATH." >&2
  exit 127
fi

# Warm once outside NCU to pay import/JIT costs before capture.
first_shape=${SHAPES%% *}
IFS=: read -r _first_name first_n first_k <<<"${first_shape}"
first_m=${M_LIST%% *}
echo "== warmup m=${first_m} n=${first_n} k=${first_k} mode=${MODE} =="
run_cmd "python3 scripts/bench_sm120_moe_small_m.py --m ${first_m} --n ${first_n} --k ${first_k} --groups ${NUM_GROUPS} --modes ${MODE} --warmup ${WARMUP} --iters ${WARMUP}"

section_args=()
for section in ${SECTIONS}; do
  section_args+=("--section" "${section}")
done
section_text=""
for arg in "${section_args[@]}"; do
  section_text+=" $(printf '%q' "${arg}")"
done

for shape in ${SHAPES}; do
  IFS=: read -r name n k <<<"${shape}"
  for m in ${M_LIST}; do
    out="${OUT_DIR}/sm120-moe-${name}-m${m}-n${n}-k${k}-${MODE}"
    echo "== ncu ${name}: m=${m} n=${n} k=${k} mode=${MODE} -> ${out}.ncu-rep =="
    cmd="DG_SM120_KERNEL_PROFILE=0 DG_JIT_WITH_LINEINFO=1 ${NCU} --config-file off --force-overwrite --import-source yes --replay-mode kernel --launch-count 1 --kernel-name '${KERNEL_NAME}' ${section_text} --export '${out}' python3 scripts/bench_sm120_moe_small_m.py --m ${m} --n ${n} --k ${k} --groups ${NUM_GROUPS} --modes ${MODE} --warmup ${WARMUP} --iters ${ITERS}"
    run_cmd "${cmd}"
  done
done
