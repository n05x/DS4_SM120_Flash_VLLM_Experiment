#!/usr/bin/env bash
# vLLM + DeepSeek-V4-Flash on SM_120, running from the local venv (cu130 wheels
# on cu12.9 host). Mirrors the docker-compose entrypoint but skips Docker.
#
# Env you can override:
#   MODEL_DIR  path to local DSv4-Flash-FP8 checkpoint
#   PORT       host port (default 8080 to avoid sglang on 8000)
#   TP         tensor parallel size (default 4 for the four Pro 6000s)
#   CUDA_VISIBLE  GPU indices (default 0,1,2,3)
set -euo pipefail

REPO=/ai/vllm/DS4_SM120_Flash_VLLM_Experiment
VENV="$REPO/venv"
MODEL_DIR=${MODEL_DIR:-/nvme/models/safetensors/DeepSeek-V4-Flash}
PORT=${PORT:-8080}
TP=${TP:-4}
CUDA_VISIBLE=${CUDA_VISIBLE:-0,1,2,3}

source "$VENV/bin/activate"

# Re-apply runtime patches (idempotent — they short-circuit if already applied).
"$VENV/bin/python" "$REPO/docker/patch_vllm_deepseekv4_venv.py"

export CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_HOME=/usr/local/cuda-12.9
export TORCH_CUDA_ARCH_LIST=12.0
export CUTE_DSL_ARCH=sm_120a
export B12X_CUTE_COMPILE_CACHE_DIR=/tmp/b12x_cute_cache

# Core vllm + DeepGEMM toggles (mirrors docker-compose defaults).
export VLLM_USE_DEEP_GEMM=1
export VLLM_USE_DEEP_GEMM_E8M0=1
export VLLM_ENGINE_READY_TIMEOUT_S=3600
export VLLM_RPC_TIMEOUT=600000
export TILELANG_CLEANUP_TEMP_FILES=1
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1
export VLLM_ALLREDUCE_USE_SYMM_MEM=0

# DG_SM120_* knobs proven good in the AGENTS.md notes.
export DG_SM120_SKIP_PADDED_ZERO=1
export DG_SM120_FP4_COLS_PER_BLOCK=8
export DG_SM120_FULL_CONTEXT_BF16_BMM=1
export DG_SM120_INDEXED_BF16_BMM=1
export DG_SM120_WORKSPACE_FUSED_ATTENTION=1
export DG_SM120_WORKSPACE_SPLIT_ATTENTION=1
export DG_SM120_ENABLE_FP8_M1_KBLOCK=1
export DG_SM120_ENABLE_FP8_M1_WARPCOL_HEURISTIC=1
export DG_SM120_ENABLE_FP8_M1_FUSED=1
export DG_SM120_ENABLE_BHR_M1_MMA=1
export DG_SM120_MOE_SKIP_SFA_FILL=1
export DG_SM120_MOE_SKIP_SFA_FILL_MAX_M=16
export DG_SM120_MOE_ROW_GROUPED=1
export DG_SM120_MOE_ROW_GROUPED_MAX_M=16
export DG_SM120_MOE_ROW_GROUPED_SKIP_SFA_FILL=1
export DG_SM120_CACHE_FP8_SFB=1
export DG_SM120_CACHE_FP8_SFA_FILL=1
export DG_SM120_SPEC_ARGMAX_FASTPATH=1
export DG_SM120_BHR_D_PER_BLOCK=1
export DG_SM120_BHR_KBLOCK_SCALE=1
export DG_SM120_ENABLE_BHR_WARPCOL_HEURISTIC=1
export DG_SM120_BHR_WARPCOL_WARPS=4
export DG_SM120_BHR_WARPCOL_COLS=0
export DG_SM120_ENABLE_SMALL_M_MMA=0
export DG_SM120_MAIN_TOPK_CAP=128
export DG_SM120_ACTIVE_HEADS=32
export DG_SM120_SEQUENTIAL_COMPRESSOR=1
export DG_SM120_FLASHMLA_PREFILL_WORKSPACE_FACTOR=1
export DG_SM120_PREFILL_WORKSPACE_CHUNK=16
export DG_SM120_PREFILL_INDEXED_SPLIT=0
export DG_SM120_PREFILL_GATHER_WORKSPACE=0
export DG_SM120_ENABLE_B12X_MOE=0
export DG_SM120_BYPASS_TP_ALLREDUCE=0

# vLLM runtime config.
export VLLM_MAX_MODEL_LEN=131072
export VLLM_MAX_NUM_BATCHED_TOKENS=4096
export VLLM_MAX_NUM_SEQS=4
export VLLM_MAX_CUDAGRAPH_CAPTURE_SIZE=8
export VLLM_PERFORMANCE_MODE=balanced
export VLLM_OPTIMIZATION_LEVEL=2
export VLLM_GPU_MEMORY_UTILIZATION=0.92
export VLLM_KV_CACHE_MEMORY_BYTES=8589934592

SPEC_CFG="$REPO/docker/vllm_speculative_mtp1_local_argmax.json"

exec "$VENV/bin/vllm" serve "$MODEL_DIR" \
  --served-model-name deepseek-v4-flash \
  --trust-remote-code \
  --attention-backend FLASHMLA_SPARSE \
  --moe-backend deep_gemm \
  --kv-cache-dtype fp8_ds_mla \
  --block-size 256 \
  --enable-expert-parallel \
  -tp "$TP" \
  --tokenizer-mode deepseek_v4 \
  --tool-call-parser deepseek_v4 \
  --enable-auto-tool-choice \
  --reasoning-parser deepseek_v4 \
  --max-model-len "$VLLM_MAX_MODEL_LEN" \
  --max-num-batched-tokens "$VLLM_MAX_NUM_BATCHED_TOKENS" \
  --max-num-seqs "$VLLM_MAX_NUM_SEQS" \
  --max-num-partial-prefills 1 \
  --max-long-partial-prefills 1 \
  --enable-chunked-prefill \
  --optimization-level "$VLLM_OPTIMIZATION_LEVEL" \
  --performance-mode "$VLLM_PERFORMANCE_MODE" \
  --max-cudagraph-capture-size "$VLLM_MAX_CUDAGRAPH_CAPTURE_SIZE" \
  --gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION" \
  --kv-cache-memory-bytes "$VLLM_KV_CACHE_MEMORY_BYTES" \
  --speculative-config "$(cat "$SPEC_CFG")" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --disable-uvicorn-access-log
