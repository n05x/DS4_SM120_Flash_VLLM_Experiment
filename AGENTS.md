# DeepGEMM SM120 / DeepSeek V4 Flash Handoff

This repository is a research fork of DeepGEMM aimed at running
`deepseek-ai/DeepSeek-V4-Flash` under vLLM on 2x NVIDIA RTX PRO 6000
Blackwell workstation GPUs, which report compute capability SM120.

## Current Functional State

- Target container image: `vllm/vllm-openai:deepseekv4-cu130`.
- Service entry point: `docker-compose.yml`.
- API port: host `8080` mapped to container `8000`.
- Model: `deepseek-ai/DeepSeek-V4-Flash`.
- vLLM configuration:
  - `--attention-backend FLASHMLA_SPARSE`
  - `--moe-backend deep_gemm`
  - `--kv-cache-dtype fp8_ds_mla`
  - tensor parallel size `2`
  - expert parallel enabled
  - max model length `131072`
- The model loads, serves, and returns valid chat completions.
- The target of `100+ tok/s` per single request has not been reached.

## Runtime Defaults

`docker-compose.yml` currently defaults to:

- `VLLM_MAX_MODEL_LEN=131072`
- `VLLM_KV_CACHE_MEMORY_BYTES=8589934592`
- `VLLM_MAX_NUM_BATCHED_TOKENS=4096`
- `VLLM_MAX_NUM_SEQS=4`
- `VLLM_MAX_CUDAGRAPH_CAPTURE_SIZE=8`
- `VLLM_PERFORMANCE_MODE=balanced`
- `VLLM_OPTIMIZATION_LEVEL=2`
- `DG_SM120_ACTIVE_HEADS=32`
- `DG_SM120_SEQUENTIAL_COMPRESSOR=1`
- `DG_SM120_ENABLE_B12X_MOE=0`
- `DG_SM120_PREFILL_WORKSPACE_CHUNK=16`
- `DG_SM120_PREFILL_INDEXED_SPLIT=0`
- `DG_SM120_PREFILL_GATHER_WORKSPACE=0`
- `DG_SM120_MOE_SKIP_SFA_FILL=1`
- `DG_SM120_MOE_SKIP_SFA_FILL_MAX_M=16`
- `DG_SM120_MOE_ROW_GROUPED=1`
- `DG_SM120_MOE_ROW_GROUPED_MAX_M=16`
- `DG_SM120_MOE_ROW_GROUPED_SKIP_SFA_FILL=1`
- `DG_SM120_CACHE_FP8_SFB=1`
- default speculative config: `docker/vllm_speculative_mtp1_local_argmax.json` (`num_speculative_tokens=1`, local argmax reduction)
- `DG_SM120_BYPASS_TP_ALLREDUCE=0`
- `DG_SM120_MHC_REUSE_BUFFERS=0`

Keep `DG_SM120_BYPASS_TP_ALLREDUCE=0` for valid output. Setting it to `1` is a
diagnostic-only invalid-output path used to estimate tensor-parallel
communication overhead.

## What Has Been Implemented

### vLLM Container and Patcher

- `Dockerfile.vllm-nightly-sm120` builds from the DeepSeek V4 CUDA 13 vLLM
  image and installs the development dependencies needed for local extension
  builds.
- `docker-compose.yml` installs this repo editable, patches installed vLLM at
  startup, and serves DeepSeek V4 Flash.
- `docker/patch_vllm_deepseekv4.py` patches the installed vLLM package in-place
  instead of requiring a full vLLM rebuild.
- SM120 is treated as DeepGEMM-capable in vLLM platform checks.
- DeepGEMM MXFP4/FP8 paths are enabled on SM120.
- MXFP4 scales are prepacked for the SM120 DeepGEMM path.
- DeepSeek V4 FP8 einsum dispatches to SM120 DeepGEMM kernels where available.
- Sparse MLA decode paths are patched to use SM120 extension kernels.
- Sparse-indexer behavior is patched for SM120 and small/full-context shapes.
- DeepSeek V4 compressor scheduling is patched to avoid first-request OOM.
- Optional layer and kernel profiling hooks exist but are off by default.
- SM120 attention now avoids padding Q from 32 local heads to the native
  FlashMLA 64-head requirement. The patched layer accepts the existing padded
  output buffer while passing the unpadded Q tensor to SM120 fallback kernels.
- Optional mHC buffer reuse exists behind `DG_SM120_MHC_REUSE_BUFFERS=1`, but
  remains off by default because it helped synthetic `mhc_pre` microbenchmarks
  and did not improve end-to-end token generation.
- Optional tensor-parallel allreduce bypass exists behind
  `DG_SM120_BYPASS_TP_ALLREDUCE=1`. It is invalid for real serving and should
  only be used for profiling communication overhead.

### DeepGEMM Extension Work

SM120-specific CUDA/C++ sources are present:

- `csrc/sm120_fp8_fp8_cutlass.cu`
- `csrc/sm120_fp8_fp4_cutlass.cu`
- `csrc/sm120_fp8_gemm_fallback.cu`
- `csrc/sm120_mqa_logits_fallback.cu`
- `csrc/sm120_hc_prenorm_fallback.cu`
- `csrc/sm120_sparse_mla_decode.cu`
- `csrc/sm120_profile.hpp`

Important implemented paths:

- SM120 FP8/FP4 MoE support sufficient for model load.
- SM120 C128 FP8 decode/projection support sufficient for functional serving.
- SM120 FP8 `bhr,hdr->bhd` einsum path for the DeepSeek V4 output projection.
- SM120 sparse MLA decode kernels, including BF16 workspace variants.
- SM120 FP8xFP8 dense CUTLASS path now caches static SFB/weight scale layouts per weight tensor and shape behind `DG_SM120_CACHE_FP8_SFB=1`, avoiding repeated SFB fill/convert launches for reused C128/decode weights.
- SM120 BHR `bhr,hdr->bhd` dispatch now uses the warp-column path for
  decode-sized shapes (`R <= 512`, `D <= 2048`, `B*H <= 64`) as well as longer
  contexts.
- SM120 CUTLASS MoE scale-fill skipping is now size-gated:
  `DG_SM120_MOE_SKIP_SFA_FILL=1` skips compact SFA fill only for
  `m <= DG_SM120_MOE_SKIP_SFA_FILL_MAX_M` by default. This keeps the small-M
  decode microbench win without hurting larger batched shapes.
- SM120 CUTLASS MoE row-grouped compact setup is enabled for low-M decode:
  `DG_SM120_MOE_ROW_GROUPED=1` uses one grouped-CUTLASS problem per routed row
  for `m <= DG_SM120_MOE_ROW_GROUPED_MAX_M`.
  `DG_SM120_MOE_ROW_GROUPED_SKIP_SFA_FILL=1` skips the row-group scale fill for
  packed UE8M0 activation scales. This is bit-exact in the tested packed-scale
  path and removes two setup launches from the tiny-M decode path.
- SM120 sparse MLA prefill from BF16 workspace:
  - exported as `deep_gemm._C.sm120_sparse_mla_prefill_from_bf16_workspace`
  - validated synthetically for BF16 tolerance
  - not enough by itself to fix long-prompt TTFT
- SM120 subchunked sparse prefill path in the vLLM patcher:
  - builds a bounded BF16 workspace per chunk
  - calls `sm120_sparse_mla_decode_from_bf16_workspace_split`
  - avoids allocating one huge gathered workspace for long prompts
- Experimental SM120 direct-indexed sparse prefill exists behind
  `DG_SM120_PREFILL_INDEXED_SPLIT=1`. It is correct in synthetic tests but
  slower than the current chunked workspace path, so keep it disabled by
  default.
- Experimental SM120 C++ BF16/FP16 workspace gather exists behind
  `DG_SM120_PREFILL_GATHER_WORKSPACE=1`. It is correct but was slightly slower
  than PyTorch advanced-index gather in synthetic tests, so keep it disabled by
  default unless testing allocator/memory behavior.

### Bench and Utility Scripts

Useful scripts include:

- `scripts/test_deepseek_v4_flash_api.sh`
- `scripts/bench_deepseek_v4_flash_api.sh`
- `scripts/bench_deepseek_v4_flash_parallel_sweep.sh`
- `scripts/estimate_deepseek_v4_flash_memory.py`
- `scripts/bench_sm120_fp8_bhr_hdr_bhd.py`
- `scripts/bench_sm120_fp8_m1.py`
- `scripts/bench_sm120_fp8_projection_shapes.py`
- `scripts/bench_sm120_moe_small_m.py`
- `scripts/bench_sm120_mhc.py`
- `scripts/bench_sm120_workspace_attention.py`

## What Worked

- DeepSeek V4 Flash can load and serve on 2x RTX PRO 6000 with this fork.
- Explicit KV reservation fixed the first-request OOM:
  - `--kv-cache-memory-bytes 8589934592`
- CUDA graph capture works with the current non-eager configuration.
- Parallel serving works better than single-request serving:
  - warmed MTP/local-argmax c4 256-token decode reached roughly `176 tok/s` aggregate steady decode
  - warmed single-request MTP/local-argmax 512-token decode is roughly `85-86 tok/s` steady decode

- MTP speculative decode with local argmax reduction is now the default serving path:
  - config file: `docker/vllm_speculative_mtp1_local_argmax.json`
  - `num_speculative_tokens=1` with local argmax keeps c1 around `85-86 tok/s` while improving warmed c4 aggregate decode to about `176 tok/s`.
  - `num_speculative_tokens=2` reached about `92 tok/s` c1 after warmup but regressed c4 aggregate to about `115 tok/s`, so it is not the balanced default.
- FP8xFP8 SFB scale caching is correct and improves targeted dense FP8 microbenchmarks:
  - `m=4,n=1536,k=4096`: `36.926 us` -> `32.808 us`
  - `m=4,n=4096,k=4096`: `38.939 us` -> `32.823 us`
  - `m=8,n=4096,k=4096`: `38.938 us` -> `32.810 us`
  - `m=8,n=16384,k=1024`: `30.740 us` -> `22.580 us`
  - cache on/off was bit-exact for the validated UE8M0 scale case.
  - end-to-end c1 only moved slightly (`~86 tok/s`), so this removes wasted repeated work but is not the missing architectural 100+ tok/s lever.
- C128 FP8 M=1 projection microbenchmarks improved substantially after the
  contiguous UE8M0 k-block specialization:
  - example projection shapes improved from roughly `18-25 us` to `14-18 us`
  - this did not translate into a large end-to-end speedup
- The SM120 no-Q-padding patch is valid and keeps the service working, but only
  moved end-to-end single-request throughput by a very small amount.
- The size-gated MoE SFA-fill heuristic is correct in microbenchmarks:
  - `m=1`: `skip_fill` about `24.7 us` vs default about `26.8 us`
  - `m=64`: unconditional `skip_fill` was worse, so the default gate is `m<=16`
  - end-to-end API throughput barely moved, so this is not the missing
    `100+ tok/s` lever
- TP allreduce is not the main bottleneck:
  - diagnostic invalid-output bypass only improved warmed single-request
    throughput by roughly `4%`
- Native SM120 sparse MLA prefill op compiled and passed synthetic checks:
  - BF16 output max error around `0.0039-0.0078`
  - LSE max error around `4.8e-7`
- Synthetic sparse prefill showed the existing workspace split attention kernel
  is much faster than the naive native fused prefill kernel:
  - `S=64,H=64,K=2048`: native about `11.3 ms`, workspace split plus gather
    about `2.85 ms`
  - `S=128,H=64,K=2048`: native about `17.4 ms`, workspace split plus gather
    about `6.3 ms`
- The direct-indexed split sparse prefill path is correct but not faster:
  - `S=16,H=32,K=2048`: native about `2.90 ms`, direct indexed split about
    `704 us`, chunked workspace about `431 us`
  - `S=128,H=32,K=2048`: native about `14.9 ms`, direct indexed split about
    `7.46 ms`, chunked workspace about `4.85 ms`
- Long decode length alone is not the main problem:
  - small-prompt completions from `128` to `1024` output tokens stayed around
    `58-65 tok/s`

## What Failed or Underperformed

- Direct FlashMLA sparse prefill cannot be used as-is on SM120:
  - calling the bundled binary directly fails with
    `Sparse Attention Forward Kernel is only supported on SM90a and SM100f architectures.`
  - do not just bypass Python-side capability checks; the failure is enforced
    by the compiled FlashMLA extension itself
  - this is why `flash_mla_sparse_fwd` had to be patched for SM120 instead of
    simply allowing SM120 through `is_flashmla_sparse_supported`
- Long uncached prompts are still too slow because TTFT/prefill dominates.
  - decode-only tests with short prompts can look acceptable and are misleading
    for long-context use
  - repeated prompts can also be misleading because prefix caching hides prefill
    cost; use unique random text from token 0 when measuring TTFT
  - non-streaming completion tok/s is a bad metric for long prompts because it
    divides completion tokens by prefill + decode wall time
- The first Python/PyTorch sparse prefill correctness fallback was functional
  but extremely slow.
  - it did per-token Python looping, `index_select`, FP32 conversion, matmul,
    softmax, and output copy
  - it kept the model usable but caused severe TTFT collapse on substantial
    prompts
  - do not revive this path except as a correctness reference
- Native `sm120_sparse_mla_prefill_from_bf16_workspace` improved the fallback
  but was still too slow end-to-end.
  - it passed synthetic correctness, but it is still scalar CUDA over BF16
    workspace values rather than a production tensor-core sparse attention
  - synthetic result example: `S=128,H=64,K=2048` took about `17.4 ms`
  - it should be treated as a fallback/reference kernel, not the final prefill
    solution
- The existing workspace split attention kernel was faster synthetically, but
  using it from vLLM is still not a complete fix.
  - it requires materializing gathered BF16 workspace chunks from sparse
    indices
  - the gather/materialization overhead and repeated per-layer calls still
    dominate enough that live TTFT only improved modestly
  - a full-size gathered workspace for long prompts would be too large, so the
    path must stay subchunked unless memory use is redesigned
- The C++ reusable workspace gather did not beat PyTorch gather in synthetic
  prefill tests:
  - `S=16,H=32,K=2048`: chunked workspace about `431 us`, C++ gather plus
    chunked workspace about `443 us`
  - Keep `DG_SM120_PREFILL_GATHER_WORKSPACE=0` unless profiling allocator churn
    or memory-pressure behavior specifically.
- The subchunked workspace path improved TTFT only modestly in the live server:
  - around `4.1k` prompt tokens: `13.2s` TTFT -> `11.3s`
  - around `8.2k` prompt tokens: `28.7s` TTFT -> `24.3s`
  - around `16.4k` prompt tokens: `62-64s` TTFT -> `55.8s`
- Synthetic chunk sweep suggested chunk `16` may be faster than chunk `64`, and
  compose now defaults to chunk `16`.
  - chunk `16` was best in one synthetic sweep for `S=512,H=64,K=2048`
  - larger chunks reduce Python/operator loop count but increase gathered
    workspace pressure and were not consistently faster
  - do not assume bigger chunk is better
- C128 FP8 M=1 projection microbench improvements did not materially change API
  throughput.
  - the contiguous UE8M0 k-block path is real and faster on projection-shaped
    microbenches
  - end-to-end speed barely moved, so C128 projection is not the only bottleneck
  - future work should not spend another cycle only tuning this path without a
    profile proving it is hot
- The first SM120 strategy of "just use SM100" is invalid.
  - DeepGEMM's SM100 path relies on TMEM assumptions that do not map directly
    to SM120
  - C4 can fall back elsewhere, but C128 routes into DeepGEMM and needed SM120
    handling
- vLLM rebuilds are expensive and were mostly avoided.
  - patching the installed vLLM package at container startup was enough for the
    experiments here
  - rebuilding vLLM is unlikely by itself to fix performance unless it includes
    new compiled FlashMLA/CUDA kernels
- vLLM profiler restarts are risky and should be done only when the API can be
  down for a while.
  - one attempt failed because `VLLM_EXTRA_ARGS` expanded JSON without quoting,
    producing invalid `{profiler:torch,...}` input to vLLM
  - `docker-compose.yml` now has `VLLM_PROFILER_CONFIG_JSON` to pass
    `--profiler-config` as one quoted argument, but this path still needs a
    clean validation run
  - dotted kebab-case CLI keys like
    `--profiler-config.torch-profiler-dir` were rejected
  - avoid enabling `DG_SM120_KERNEL_PROFILE` while CUDA graph capture is active
    unless you intend to benchmark with synchronization overhead
- Optional b12x MoE integration exists behind `DG_SM120_ENABLE_B12X_MOE=1` but
  remains disabled because it has not been proven stable end-to-end.
  - it may still be useful as a reference for SM120 FP8 activation x packed FP4
    expert-weight flow and scale swizzles
  - do not enable it by default without correctness and model-load validation
- Single-request `100+ tok/s` has not been achieved.
  - aggregate throughput can exceed `100 tok/s` with concurrent requests, but
    per-request decode and long-prompt TTFT remain below the goal
  - do not report aggregate concurrency numbers as satisfying the single-request
    target

## Current Bottleneck Understanding

There are two separate regimes:

- Decode-only throughput is decent but still below target for one request.
- Long prompt throughput collapses because uncached prefill/TTFT is slow.

The most important remaining bottleneck is DeepSeek V4 sparse MLA prefill on
SM120. The bundled FlashMLA sparse prefill kernel rejects SM120, and the current
replacement paths still involve BF16 workspace materialization and many
per-layer operations.

The second likely bottleneck is sparse MLA decode and MoE decode efficiency.
The current implementation still contains fallback/workspace-heavy paths that
are functional but not production-grade.

For the measured single-request decode path, the dominant remaining targets
are the tiny-M SM120 FP8xFP4 routed expert GEMM body, the dense FP8xFP8
C128/decode GEMM body, and per-layer fixed overhead. The setup overhead has
been reduced and bounded:

- `row_grouped_skip_fill` removes the compact init/fill setup launches for
  `m <= 16` and is bit-exact against the default CUTLASS path in synthetic
  tests.
- CUDA profiling on `m=6,n=4096,k=2048` shows the row-grouped path spends about
  `344.981 us` of `356.500 us` total CUDA time over 10 calls in the CUTLASS
  device kernel itself. The row setup kernel is about `11.519 us` over 10 calls.
- This means another helper/setup kernel tweak cannot plausibly produce the
  missing 2x. Meaningful decode improvement needs a better tiny-M expert GEMM
  work decomposition or a more persistent/fused MoE layer kernel.

## Recent Delta Since Last Commit

- Added the SM120 no-Q-padding vLLM patch. This removes one per-layer padding
  launch/copy for Q on SM120 while preserving padded output-buffer compatibility.
- Added default-off `DG_SM120_MHC_REUSE_BUFFERS` support in the vLLM patcher.
  Keep it off unless re-testing mHC allocation reuse.
- Added default-off `DG_SM120_BYPASS_TP_ALLREDUCE` support for diagnostics.
  Never use it for valid output.
- Extended the BHR warp-column heuristic to short decode shapes.
- Changed MoE compact SFA-fill skipping to be gated by
  `DG_SM120_MOE_SKIP_SFA_FILL_MAX_M`.
- Added SM120 MoE row-grouped compact setup for low-M decode, gated by
  `DG_SM120_MOE_ROW_GROUPED` and `DG_SM120_MOE_ROW_GROUPED_MAX_M`.
- Added `DG_SM120_MOE_ROW_GROUPED_SKIP_SFA_FILL` for packed UE8M0 activation
  scales. In synthetic tests it is bit-exact and improves low-M setup overhead.
- Added `row_grouped` and `row_grouped_skip_fill` modes to
  `scripts/bench_sm120_moe_small_m.py`.
- Tested but rejected several MoE ideas:
  - CUTLASS `N=64` tile shape failed to compile because the SM120 blockscaled
    scale-factor TMA layout requires 128-wide compatible tiles.
  - CUTLASS `N=256` tile shape failed because auto pipeline stage count dropped
    below the required minimum.
  - CUTLASS cooperative grouped schedule compiled but regressed key shapes
    versus ping-pong.
  - the existing hand-written `DG_SM120_ENABLE_SMALL_M_MMA` path remains slower
    than CUTLASS for relevant shapes.
  - a CUDA-core skinny dot-product prototype was correct but slower for
    realistic `m=6` shapes and was removed.
- Added `VLLM_PROFILER_CONFIG_JSON` compose plumbing for quoted profiler config.
  This was added after an unquoted profiler restart broke startup.
- Added `DG_SM120_PREFILL_INDEXED_SPLIT` and
  `DG_SM120_PREFILL_GATHER_WORKSPACE` as default-off experimental prefill
  switches.
- Added a direct-indexed sparse prefill split API and a reusable BF16/FP16
  workspace gather API. Both are correct synthetically, but neither is a
  default speed win over the current chunked workspace path.
- Added `cuda-nsight-compute-13-0` to the development container and a targeted
  `scripts/run_ncu_sm120_moe.sh` runner. Nsight Compute currently launches but
  host GPU performance counters are blocked by `ERR_NVGPUCTRPERM`; fix host
  counter permissions before relying on NCU metrics.


- Added default MTP speculative serving config (`docker/vllm_speculative_mtp1_local_argmax.json`) and changed compose's default CUDA graph capture size to `8`. This is the best balanced config found so far: c1 improves from the non-MTP `~69-70 tok/s` range to `~85-86 tok/s`, while warmed c4 aggregate steady decode is about `176 tok/s`.
- Added DeepSeek V4 MTP local-argmax patching in `docker/patch_vllm_deepseekv4.py` so draft token selection avoids full-vocab all-gather. Patching only generic `deepseek_mtp.py` failed; the working path patches `deepseek_v4_mtp.py` as well.
- Added SM120 FP8xFP8 static SFB scale-layout caching in `csrc/sm120_fp8_fp8_cutlass.cu`. Microbenchmarks improved selected dense FP8 small-M shapes by roughly `1.13x-1.36x`; end-to-end c1 remained below target.
- Tested and rejected MTP `num_speculative_tokens=2`/capture-size `16` as a default: warm c1 reached about `92 tok/s`, but c4 aggregate steady decode regressed to about `115 tok/s`.
- Tested and rejected MTP `num_speculative_tokens=3`: c1 steady decode dropped to about `77-81 tok/s`.
- Re-tested the restored default service after the latest rebuild:
  - correctness request returned `OK.`
  - c1 512-token decode: about `85.5 request tok/s`, `87.8 steady tok/s`
  - c4 256-token decode remains valid but varies by run; recent aggregate
    steady measurements were roughly `140-146 tok/s`
- Tested and rejected a dense FP8xFP8 per-row small-M MMA path for broad
  decode-sized `m=2/4/8` shapes. It compiled and was correct, but regressed
  badly versus the current CUTLASS path; examples:
  - `m=2,n=1536,k=4096`: default `30.868 us`, small-M MMA `81.940 us`
  - `m=2,n=16384,k=1024`: default `20.592 us`, small-M MMA `50.057 us`
  The prototype was removed; do not re-add a one-row-per-MMA dense C128 path
  without a new design that fixes the occupancy/register-use problem.
- Tested and rejected additional runtime/config knobs as defaults:
  - `DG_SM120_PREFILL_WORKSPACE_CHUNK=64` worsened a 516-token prompt TTFT to
    about `4.99s` and hurt c1 decode in that run; keep chunk `16`.
  - `VLLM_MAX_NUM_BATCHED_TOKENS=6144` with MTP2 failed startup with
    `CUDA_ERROR_ILLEGAL_ADDRESS`; keep `4096`.
  - MTP2 plus `DG_SM120_MAIN_TOPK_CAP=64` did not improve c1 and regressed c4;
    keep MTP1 plus top-k cap `128`.
  - `DG_SM120_SEQUENTIAL_COMPRESSOR=0` was correct but did not improve TTFT or
    decode; keep it enabled for stability.

Latest MoE microbench results from one-off CUDA 13 container rebuilds:

- `m=6,n=4096,k=2048`: default `41.130 us`,
  `row_grouped_skip_fill` `37.175 us`, bit-exact, `1.106x`.
- `m=6,n=4096,k=4096`: default about `71.778 us`,
  `row_grouped_skip_fill` about `64.679 us`, bit-exact, `1.110x`.
- `m=6,n=7168,k=2048`: default `61.379-61.536 us`,
  `row_grouped_skip_fill` `57.002-57.100 us`, bit-exact, about `1.08x`.
- `m=6,n=7168,k=4096`: default `96.052 us`,
  `row_grouped_skip_fill` `84.007 us`, bit-exact, `1.143x`.
- `m=6,n=4096,k=7168`: default `117.970 us`,
  `row_grouped_skip_fill` `116.479 us`, bit-exact, only `1.013x`.
- `m=24,n=4096,k=2048`: default `99.072 us`,
  `row_grouped_skip_fill` `100.077 us`, bit-exact, slight regression. Keep the
  max-M guard enabled.
- Rechecked after the Ralph build restore:
  - `m=6,n=4096,k=4096`: `row_grouped_skip_fill` about `63.6 us`
  - `m=6,n=4096,k=2048`: `row_grouped_skip_fill` about `37.6-38.2 us`
  - Limiting CUTLASS `hw_info.sm_count` to b12x-like active-cluster values
    (`48,64,84,107,127,148`) did not improve these shapes and often regressed,
    so no SM-count override is kept.

## Next Work

1. Build a production SM120 sparse MLA prefill kernel.
   Avoid Python loops, avoid full BF16 workspace materialization, and compute
   directly from FP8 MLA cache plus sparse indices where possible.

2. Validate `DG_SM120_PREFILL_WORKSPACE_CHUNK=16` end-to-end.
   Compose now defaults to `16`; measure token-counted TTFT for about `4k`,
   `8k`, and `16k` uncached prompt tokens.

3. Replace workspace-heavy sparse MLA decode with direct SM120 tensor-core
   computation.
   Current workspace split kernels are useful stopgaps but should not be the
   final production path.

4. Profile the live prefill path with valid vLLM profiler config.
   Use unique prompts from token 0 so prefix caching does not hide TTFT.
   Prefer `VLLM_PROFILER_CONFIG_JSON='{"profiler":"torch",...}'` over
   `VLLM_EXTRA_ARGS` so JSON is passed as one argument.

5. Revisit the C128 decode/MoE path after prefill is fixed.
   The C128 projection microbench improved, but end-to-end did not. For MoE
   decode, setup overhead is now small enough that the next useful work is a
   production tiny-M FP8xFP4 expert GEMM or persistent/fused MoE layer, not more
   helper-kernel cleanup.

6. Audit all remaining fallback paths.
   Files with `fallback` in the name are functional scaffolding, not proof that
   performance is production-ready.

## Useful Commands

Start or restart:

```bash
docker compose up -d --force-recreate vllm
```

Health check:

```bash
curl -sS http://127.0.0.1:8080/health
```

Tiny correctness request:

```bash
curl -sS http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-ai/DeepSeek-V4-Flash","messages":[{"role":"user","content":"Reply exactly OK."}],"max_tokens":8,"temperature":0}'
```

512-token API benchmark:

```bash
BASE_URL=http://127.0.0.1:8080 \
MAX_TOKENS=512 \
MIN_TOKENS=512 \
CONCURRENCY=4 \
scripts/bench_deepseek_v4_flash_api.sh
```

DeepGEMM extension rebuild inside the container:

```bash
docker compose exec -T vllm bash -lc \
  'cd /workspace/DeepGEMM && MAX_JOBS=8 DG_FORCE_BUILD=1 python3 setup.py build_ext --inplace'
```

## Caveats

- This is a working research prototype, not an upstream-quality patch.
- vLLM is patched in-place at container startup rather than rebuilt from source.
- `vllm_src/` is only reference source; the running container patches the
  installed package in the image.
- `VLLM_*` local config variables produce "unknown vLLM environment variable"
  warnings. They are consumed by the compose startup shell before `vllm serve`,
  so the warnings are noisy but expected.
- Keep `DG_SM120_KERNEL_PROFILE=0` for real benchmarks. The kernel profiler uses
  CUDA event synchronization and will distort throughput when enabled.
- Keep `DG_SM120_BYPASS_TP_ALLREDUCE=0` for real benchmarks. The bypass is
  deliberately invalid and only measures how much TP allreduce costs.
