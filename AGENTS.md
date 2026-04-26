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
- `DG_SM120_PREFILL_WORKSPACE_CHUNK=256`
- `DG_SM120_PREFILL_TORCH_BMM=1`
- `DG_SM120_PREFILL_TORCH_COMPILE=0`
- `DG_SM120_PREFILL_TORCH_COMPILE_MIN_ROWS=64` (opt-in guard only)
- `DG_SM120_PREFILL_CUDNN=0`
- `DG_SM120_PREFILL_CUDNN_UNMASKED=0`
- `DG_SM120_PREFILL_TRUST_INDICES=1`
- `DG_SM120_PREFILL_TRIM_TOPK=1`
- `DG_SM120_PREFILL_TRIM_TOPK_MIN_WIDTH=2048`
- `DG_SM120_PREFILL_DYNAMIC_COMPRESSED_N=1`
- `DG_SM120_PREFILL_EMPTY_COMBINED_INDICES=1`
- `DG_SM120_PREFILL_INDEXED_SPLIT=0`
- `DG_SM120_PREFILL_GATHER_WORKSPACE=0`
- `DG_SM120_PREFILL_INDEX_SELECT=0`
- `DG_SM120_PREFILL_DIRECT_FP8_MAP=0`
- `DG_SM120_PREFILL_DIRECT_FP8_MAP_STRICT=0`
- `DG_SM120_MOE_SKIP_SFA_FILL=1`
- `DG_SM120_MOE_SKIP_SFA_FILL_MAX_M=16`
- `DG_SM120_MOE_ROW_GROUPED=1`
- `DG_SM120_MOE_ROW_GROUPED_MAX_M=6`
- `DG_SM120_MOE_ROW_GROUPED_SKIP_SFA_FILL=1`
- `DG_SM120_MOE_DIRECT_GROUPS_WHEN_NO_SHRINK=1`
- `DG_SM120_CACHE_FP8_SFB=1`
- default speculative config: `docker/vllm_speculative_mtp1_local_argmax.json` (`num_speculative_tokens=1`, local argmax reduction)
- `DG_SM120_BYPASS_TP_ALLREDUCE=0`
- `DG_SM120_MHC_REUSE_BUFFERS=1`

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
- mHC pre/post temporary buffer reuse is enabled by default with
  `DG_SM120_MHC_REUSE_BUFFERS=1`. This follows the SGLang-style lesson that
  repeated small framework allocations/metadata work can cap decode. The post
  buffer cache uses a two-buffer non-aliasing guard so the next layer does not
  accidentally read and write the same residual allocation.
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
- SM120 CUTLASS MoE direct expert-group setup is enabled for no-shrink shapes:
  `DG_SM120_MOE_DIRECT_GROUPS_WHEN_NO_SHRINK=1` avoids the compact active-count
  setup path when routed rows already exceed the full expert count. This is a
  body/scheduling configuration for larger/concurrent routed shapes, not a
  single-token helper tweak.
- SM120 MoE unpermute/reduce has a CUDA helper for the BF16 final expert-output
  reduction:
  - exported as `deep_gemm._C.sm120_moe_unpermute_reduce_bf16`
  - mapped variant supports vLLM expert-parallel `expert_map`
  - patched into `deepgemm_unpermute_and_reduce` for SM120 BF16 hidden
    sizes at least `512`, otherwise vLLM falls back to the stock Triton
    `ep_gather`
- SM120 sparse MLA prefill from BF16 workspace:
  - exported as `deep_gemm._C.sm120_sparse_mla_prefill_from_bf16_workspace`
  - validated synthetically for BF16 tolerance
  - not enough by itself to fix long-prompt TTFT
- SM120 subchunked sparse prefill path in the vLLM patcher:
  - builds a bounded BF16 workspace per chunk
  - calls `sm120_sparse_mla_decode_from_bf16_workspace_split`
  - avoids allocating one huge gathered workspace for long prompts

- SM120 graph-native compressor metadata build:
  - exported as `deep_gemm._C.sm120_build_compressor_metadata` plus the lower-level `sm120_fill_token_to_req_indices`
  - fills DeepSeek V4 compressor `token_to_req_indices` from device `query_start_loc` and clamps compressor block-table entries nonnegative in one device launch
  - replaces CPU `repeat_interleave(...).pin_memory()` plus H2D copy and a separate Torch `block_table.clamp_(min=0)` launch in the vLLM patcher when SM120 support is present
- SM120 graph-native sparse-SWA metadata build:
  - exported as `deep_gemm._C.sm120_build_sparse_swa_metadata`
  - fuses sparse-SWA `token_to_req_indices` fill, `slot_mapping >= 0`
    validity generation, and `decode_swa_lens` tail clearing into one device
    launch
  - replaces another decode-time CPU `repeat_interleave(...).pin_memory()`
    metadata path in `vllm/v1/attention/backends/mla/sparse_swa.py`
  - exported `deep_gemm._C.sm120_build_sparse_swa_prefill_metadata` for
    sparse-SWA prefill gather lengths, replacing vLLM's Triton
    `_compute_prefill_metadata_kernel` with one graph-native SM120 CUDA launch
    when available
- SM120 graph-native FlashMLA sparse request-id metadata:
  - reuses `deep_gemm._C.sm120_fill_token_to_req_indices` in
    `FlashMLASparseMetadataBuilder.build`
  - replaces host NumPy `np.repeat(...)` plus `torch.from_numpy(...).copy_` for
    `req_id_per_token` with a stable device-buffer fill
  - this path has been live-loaded by a later service restart and passed tiny
    API correctness
- Experimental SM120 direct-indexed sparse prefill exists behind
  `DG_SM120_PREFILL_INDEXED_SPLIT=1`. It is correct in synthetic tests but
  slower than the current chunked workspace path, so keep it disabled by
  default.
- Experimental SM120 C++ BF16/FP16 workspace gather exists behind
  `DG_SM120_PREFILL_GATHER_WORKSPACE=1`. It is correct but was slightly slower
  than PyTorch advanced-index gather in synthetic tests, so keep it disabled by
  default unless testing allocator/memory behavior.
- Experimental SM120 direct FP8 sparse-prefill workspace-map exists behind
  `DG_SM120_PREFILL_DIRECT_FP8_MAP=1`.
  - exported as
    `deep_gemm._C.sm120_sparse_mla_prefill_from_two_fp8_workspace_map`
  - supports the DeepSeek V4 compressed cache plus SWA cache by encoding
    primary cache rows as nonnegative workspace-map entries and SWA rows as
    negative entries
  - patched into the vLLM sparse-prefill loop as an opt-in fallback branch, with
    `DG_SM120_PREFILL_DIRECT_FP8_MAP_STRICT=1` available to re-raise direct-path
    errors during development
  - correct in synthetic tests, but slower than the current BF16 workspace/BMM
    bridge even after adding a grouped single-kernel branch for `topk <= 768`,
    because it still performs scalar FP8 cache reads rather than a
    tensor-core/FlashMLA-style direct sparse attention computation

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
- `scripts/bench_sm120_moe_modes_sweep.py`
- `scripts/bench_sm120_moe_unpermute_reduce.py`
- `scripts/bench_sm120_mhc.py`
- `scripts/bench_sm120_compressor_metadata.py`
- `scripts/bench_sm120_direct_prefill_map.py`
- `scripts/bench_sm120_workspace_attention.py`
- `scripts/summarize_sm120_profiles.py`
- `SGLANG_DEEP_SEEK_V4_Implentation.md`

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
  - `docker-compose.yml` must read `VLLM_SPECULATIVE_CONFIG_FILE` and
    `VLLM_SPECULATIVE_CONFIG_JSON` with `printenv` inside the container command.
    Using `${...}` directly in the compose command body lets Docker Compose
    interpolate the host environment to an empty string, launching vLLM without
    `--speculative-config` even though the container env has the default file.
    That silently drops c1 decode back to the non-MTP `~68-69 tok/s` path.
  - `num_speculative_tokens=1` with local argmax keeps warmed c1 typically around
    `85-90 tok/s` steady decode, with a best observed c1 run near `94 tok/s`,
    while improving warmed c4 aggregate decode to roughly `185-190 tok/s` in
    the best recent runs.
  - `num_speculative_tokens=2` was re-tested on 2026-04-26: one c1 run showed `96.88 tok/s` steady after a long first-token delay, but repeat c1 was `89.90 tok/s` steady and c4 regressed to `148.62 tok/s` aggregate steady. Keep MTP1 as the balanced default.
- SM120 MoE BF16 unpermute/reduce helper is correct in synthetic tests and
  trims a repeated per-layer Triton gather:
  - `tokens=162, hidden=7168, topk=8, expert_map=False`: Triton
    `11.302 us`, SM120 CUDA `9.927 us`, bit-exact
  - `tokens=162, hidden=7168, topk=8, expert_map=True`: recent final-build
    run measured Triton `11.403 us`, SM120 CUDA `8.217 us`, bit-exact
  - `tokens=32, hidden=7168, topk=8, expert_map=True`: Triton `11.260 us`,
    SM120 CUDA `5.184 us`, bit-exact
  - live c1 1024-token decode after patch is not consistently the previous
    best-run `93.16 request tok/s`, `93.96 steady tok/s`; rechecks on the same
    running service were `~85-90 tok/s` steady decode.
- SGLang-inspired mHC allocation reuse is valid and now default-on:
  - synthetic `scripts/bench_sm120_mhc.py` with `tokens=1-2` improved from
    roughly `60-61 us` total two-block mHC overhead to `41-48 us` depending on
    hidden size
  - two-layer alias-hazard correctness check was bit-exact with reuse enabled
  - warm production serving after restart remained valid (`Reply exactly OK.`)
    with c1 1024-token steady decode in the `88.75-90.42 tok/s` repeat range and c4 512-token
    aggregate steady decode around `188.75 tok/s`
- SM120 sparse MLA prefill now has a default-on tensor-core PyTorch BMM path:
  - `DG_SM120_PREFILL_TORCH_BMM=1` gathers each prefill chunk into a bounded
    BF16 workspace, then computes `Q @ K^T`, softmax, and `P @ V` with batched
    matmuls instead of the scalar SM120 workspace attention kernels.
  - `DG_SM120_PREFILL_TRUST_INDICES=1` skips per-chunk clamp/cast work for
    vLLM-generated sparse indices and reuses a cached device position vector
    for top-k masking.
  - This is still a workspace-based bridge rather than the final direct
    production FlashMLA-equivalent kernel, but it removes a large scalar-kernel
    bottleneck for long prompts.
  - Synthetic `S=64,H=32,K=2048` sparse prefill: native scalar about
    `9.33 ms`, direct indexed split about `3.49-3.51 ms`, chunked workspace
    gather plus split about `2.83-2.84 ms`, gather plus tensor-core BMM about
    `0.50-0.51 ms`, trusted-index tensor-core BMM about `0.36 ms`, with BF16
    output bit-exact to the split reference and LSE max diff around `9.5e-7`.
    A fresh run while the service was live measured trusted-index BMM
    `365.078 us`, `index_select` BMM `360.719 us`, and cached-index-select
    BMM `359.869 us`, so `index_select(out=...)`/workspace caching is not a
    meaningful standalone lever.
  - Fresh 2026-04-26 static-shape checks with the current capped sparse width
    (`topk=128`) showed `torch.compile(..., mode="reduce-overhead")` can improve
    the isolated cached index-select+BMM prefill bridge: at `S=64,H=32,K=128`,
    eager was `183.628 us` and compiled was `121.863 us`; at
    `S=256,H=32,K=128`, eager was `185.987 us` and compiled was `130.095 us`,
    all bit-exact within BF16/LSE tolerance. Live vLLM serving rejected this as
    a default: enabling the compiled/index-select bridge caused an illegal CUDA
    memory access on the second ~4k-prompt request after a restart. Keep
    `DG_SM120_PREFILL_TORCH_COMPILE=0` and `DG_SM120_PREFILL_INDEX_SELECT=0` by
    default; the guarded compiled path remains opt-in only for focused debugging.
  - Patched helper microbench at `S=64,H=32,K=2048,chunk=64`: old extension
    path about `1.79 ms`, tensor-core BMM path about `0.46 ms`.
  - Live post-restart long-prompt checks showed warmed TTFT around `0.39-0.40s`
    for `~1k` prompt tokens, `~2.0s` for `~4k`, `~5.4s` for `~8k`, and
    `~16.7-17.4s` for `~16k`; fresh checks measured `~4k`/64-token TTFT
    `1.985s` with `85.90 tok/s` steady decode, `~8k` TTFT `5.452s` with
    `82.04 tok/s`, and `~16k` TTFT `16.736s` with `77.08 tok/s`.
    Short-prompt c1 decode remains around `89-91 tok/s` steady after warmup.
  - After the graph-native metadata patch and restored safe prefill defaults,
    fresh 2026-04-26 live checks measured:
    - correctness: tiny chat returned `OK.`
    - c1 256-token short-prompt streaming: `89.33 request tok/s`,
      `95.20 steady tok/s`, `0.187s` TTFT
    - c4 256-token short-prompt streaming: `137.44 aggregate request tok/s`,
      `140.05 aggregate steady tok/s`; per-request steady decode remained
      `35.46-47.54 tok/s`
    - unique ~4.1k-prompt / 32-token completion: `2.095s` TTFT
      (`1957 prompt tok/s`) and `81.96 steady tok/s`
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
- The workstation GPUs are currently power-capped at `300 W` even though
  `nvidia-smi -q -d POWER` reports `600 W` default/max. During c1 decode both
  GPUs sit at `~300 W` and `99-100%` utilization; GPU2 was observed around
  `2.42-2.48 GHz` SM clocks under the cap. Do not change the power limit: the
  user intentionally capped the cards to control heat. Treat the cap as a fixed
  environmental constraint and optimize kernels/configuration within it.
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

- The SM120 graph-native compressor metadata build is correct and removes repeated CPU/framework metadata work in isolation:
  - `lens=[1,1,1,1]`: fused device fill+clamp about `2.435 us` vs CPU repeat/pin/copy plus Torch clamp about `25.662 us` (`10.5x`)
  - `lens=[64,128,256,512]`: fused device fill+clamp about `2.422 us` vs CPU repeat/pin/copy plus Torch clamp about `35.580 us` (`14.7x`)
  - `lens=[2048,2048,4096,8192]`: fused device fill+clamp about `2.458 us` vs CPU repeat/pin/copy plus Torch clamp about `43.799 us` (`17.8x`)
  - The sibling sparse-SWA metadata builder is also correct in isolation:
    `2.57-2.58 us` vs CPU repeat/pin/copy plus validity/tail-clear
    `32.7-49.3 us` (`12.7x-19.1x`) for the same three token-count cases.
  - Fresh FlashMLA sparse `req_id_per_token` metadata microbench after patching another host-side path:
    - `lens=[1,1,1,1]`: SM120 device fill `2.494 us` vs NumPy repeat/from_numpy/copy `6.098 us` (`2.45x`)
    - `lens=[64,128,256,512]`: SM120 device fill `2.496 us` vs NumPy repeat/from_numpy/copy `6.522 us` (`2.61x`)
  - This is a graph/metadata hygiene win, not by itself enough to close the 100+ tok/s gap.

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
- The direct FP8 workspace-map sparse-prefill prototype proves that removing
  BF16 workspace materialization is not enough if the replacement is scalar.
  Fresh `scripts/bench_sm120_direct_prefill_map.py` results after the Ralph
  rebuild:
  - `tokens=16,heads=32,topk=512`: direct path bit-exact, BF16 workspace split
    `106.847 us`, direct FP8 map `481.008 us`
  - `tokens=16,heads=32,topk=2048`: direct path bit-exact, BF16 workspace split
    `400.935 us`, direct FP8 map `1966.068 us`
  - keep `DG_SM120_PREFILL_DIRECT_FP8_MAP=0`; the useful next direct-prefill
    work is a tensor-core/FlashMLA-style kernel that computes from FP8 cache and
    sparse indices without scalar per-candidate cache scans.
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
- Early scalar-kernel chunk sweeps suggested chunk `16` could beat chunk `64`,
  but that result no longer applies to the default tensor-core BMM bridge.
  Early patched-helper sweeps without adaptive top-k trimming made chunk `64`
  look best. After adaptive top-k trimming and dynamic C128 workspace sizing,
  chunk `256` became the better default for prefill TTFT because the effective
  sparse width is much smaller and fewer per-chunk framework launches dominate.
  - do not generalize old scalar-kernel or untrimmed-workspace chunk sweeps; the
    optimal chunk depends on the current trimmed tensor-core BMM bridge.
- C128 FP8 M=1 projection microbench improvements did not materially change API
  throughput.
  - the contiguous UE8M0 k-block path is real and faster on projection-shaped
    microbenches
  - end-to-end speed barely moved, so C128 projection is not the only bottleneck
  - future work should not spend another cycle only tuning this path without a
    profile proving it is hot
- A custom CUDA compact MoE permute/scatter prototype was correct but rejected.
  - Synthetic isolated `ep_scatter` improved only about `1.1x` (`~20.3 us` to
    `~18.4 us`) and the first live decode test regressed, likely because the
    CUDA block-per-token copy shape interacted worse with the full MoE/CUDA
    graph than Triton's existing vectorized scatter.
  - The prototype was removed rather than kept as another disabled helper.
  - If revisiting MoE routing overhead, design around the whole permute +
    grouped-GEMM + unpermute pipeline rather than only replacing `ep_scatter`.
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

The latest sparse-prefill bridge breakdown has shifted with the current
`DG_SM120_MAIN_TOPK_CAP=128` and `DG_SM120_PREFILL_WORKSPACE_CHUNK=256` defaults.
At `S=256,H=32,K=128`, cached eager index-select+BMM measured `185.987 us`, while
the compiled cached variant measured `130.095 us`; the older scalar/direct
extension paths remain much slower (`native 1265.577 us`, split `700.647 us`,
chunked workspace `423.947 us`). This makes default-on compiled BMM worthwhile
for substantial prompts, but it is still a bridge: useful remaining sparse-MLA
work is a native fused/direct path that removes framework launch/allocation
boundaries and BF16 workspace traffic entirely.


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
- Fresh MoE chain microbenchmarks confirm the remaining cost is the GEMM body,
  not activation or unpermute helper overhead: `m=12,hidden=4096,moe_hidden=2048`
  measured FC1 `96.224 us`, activation/quant `3.402 us`, FC2 `53.431 us`, reduce
  `6.204 us`, and graph chain+reduce `181.249 us`; `m=6` measured FC1
  `63.610 us`, activation/quant `3.413 us`, FC2 `39.025 us`, reduce `6.203 us`,
  and graph chain+reduce `106.773 us`. A larger fusion must replace or
  substantially restructure the FP8xFP4 grouped GEMMs themselves; fusing only
  activation/reduce cannot close the gap to 100+ tok/s alone.

## Recent Delta Since Last Commit

- Added SM120 graph-native compressor metadata build (`sm120_build_compressor_metadata`) and patched vLLM `CompressorMetadataBuilder.build` to use it when available. This fuses token-to-request fill and block-table nonnegative clamp into one device launch, avoiding per-build CPU `repeat_interleave(...).pin_memory()` plus H2D copy and a separate Torch clamp launch.
- Added SM120 graph-native sparse-SWA metadata build (`sm120_build_sparse_swa_metadata`) and patched vLLM `DeepseekSparseSWAMetadataBuilder.build` to use it when available. This fuses token-to-request fill, slot-validity generation, and decode-lens tail clearing into one device launch.
- Added SM120 graph-native FlashMLA sparse `req_id_per_token` metadata patch. `FlashMLASparseMetadataBuilder.build` now uses `sm120_fill_token_to_req_indices` on SM120 instead of host `np.repeat(...)` plus `torch.from_numpy(...).copy_`; this is installed for the next vLLM restart but was not live-loaded into the already-running service.
- Added `scripts/bench_sm120_compressor_metadata.py` to verify correctness and benchmark the metadata path. Fresh isolated results show roughly `2.42-2.46 us` for fused compressor device fill+clamp versus `25.7-43.8 us` for CPU repeat/pin/copy plus Torch clamp, and roughly `2.57-2.58 us` for sparse-SWA metadata versus `32.7-49.3 us` for the corresponding CPU/framework metadata sequence.
- Extended `scripts/bench_sm120_sparse_prefill.py` with a compiled cached-index-select BMM variant so prefill chunk sizes can be tested without vLLM restarts. Older uncapped/topk-2048 experiments were shape-sensitive and often regressed, but fresh capped-topk (`K=128`) measurements now favor compile for substantial chunks.
- Added sentinel/invalid-index clamping around the opt-in `DG_SM120_PREFILL_INDEX_SELECT` path, but kept compose default-off after live validation showed the compiled/index-select prefill bridge can crash vLLM on repeated long-prompt requests.
- Added an experimental/default-off `DG_SM120_PREFILL_CUDNN` sparse-prefill bridge. It uses CUDA 13/PyTorch cuDNN SDPA with `compute_log_sumexp=True` over the gathered per-token sparse workspace, preserving DeepSeek-V4 attention-sink gating while avoiding explicit `scores`/`softmax`/`probs` tensors in Python. The cuDNN op ignored additive bias for masked sparse entries in a sentinel-index test, so the patcher requires `DG_SM120_PREFILL_CUDNN_UNMASKED=1` before it will actually use this path; with only `DG_SM120_PREFILL_CUDNN=1` it falls back to the safe BMM bridge. Synthetic unmasked helper tests show it is correct within BF16 tolerance and faster for very small chunks, but not a default win for the current capped sparse-prefill path; keep cuDNN off unless a workload-specific benchmark proves otherwise.
- Tested but rejected `DG_SM120_PREFILL_TORCH_COMPILE=1` as a serving default.
  The guarded compiled BMM bridge was faster synthetically for capped-topk
  (`K=128`) chunks, but a live restart crashed with an illegal CUDA memory
  access on the second ~4k-prompt request. Compose defaults are restored to
  `DG_SM120_PREFILL_TORCH_COMPILE=0` and `DG_SM120_PREFILL_INDEX_SELECT=0`.
- Added an experimental/default-off `DG_SM120_PREFILL_DIRECT_FP8_MAP` sparse-prefill branch. It builds a device workspace map from vLLM block tables and can read directly from the compressed FP8 cache plus SWA cache without materializing BF16 workspace rows. Synthetic tests are bit-exact, but the scalar direct cache-read kernel is much slower than the current BF16 workspace/BMM bridge (`topk=2048`: `1966 us` vs `401 us` on the synthetic map bench). A grouped single-kernel branch for `topk <= 768` only moved `topk=512` from about `506 us` to `481 us`, so this remains a scaffold for the eventual tensor-core direct sparse-attention kernel rather than a default serving path.
- Added `scripts/bench_sm120_direct_prefill_map.py` to keep that direct-FP8-map experiment reproducible without loading vLLM or DeepSeek V4.
- Removed a redundant `topk_indices_buffer[:num_padded_tokens] = -1` fill from the SM120 full-context sparse-indexer path; `sm120_fill_decode_all_indices` already writes every returned top-k column including invalid `-1` entries.

- Added the SM120 no-Q-padding vLLM patch. This removes one per-layer padding
  launch/copy for Q on SM120 while preserving padded output-buffer compatibility.
- Promoted `DG_SM120_MHC_REUSE_BUFFERS=1` to the compose default and extended
  the vLLM patcher from pre-buffer reuse to pre/post mHC temporary reuse. The
  post cache avoids input/output aliasing with a two-buffer ring.
- Replaced neutral UE8M0 scale-fill CUDA launches in the SM120 CUTLASS FP8 and
  FP8xFP4 wrappers with optional `cudaMemsetAsync(0x7f)` initialization via
  `DG_SM120_USE_MEMSET_SCALE_FILL` (default-on). Microbenchmarks were neutral,
  but the path removes tiny fill-kernel launches where scale padding still
  needs initialization.
- Added default-off `DG_SM120_BYPASS_TP_ALLREDUCE` support for diagnostics.
  Never use it for valid output.
- Extended the BHR warp-column heuristic to short decode shapes.
- Changed MoE compact SFA-fill skipping to be gated by
  `DG_SM120_MOE_SKIP_SFA_FILL_MAX_M`.
- Added SM120 MoE row-grouped compact setup for low-M decode, gated by
  `DG_SM120_MOE_ROW_GROUPED` and `DG_SM120_MOE_ROW_GROUPED_MAX_M`.
- Added `DG_SM120_MOE_ROW_GROUPED_SKIP_SFA_FILL` for packed UE8M0 activation
  scales. In synthetic tests it is bit-exact and improves low-M setup overhead.
- Promoted `DG_SM120_MOE_DIRECT_GROUPS_WHEN_NO_SHRINK=1` to the compose
  default for larger routed shapes where compacting active groups cannot reduce
  the number of CUTLASS grouped problems.
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

- Added `scripts/bench_sm120_prefill_helper_chunks.py` to benchmark the actual
  patched vLLM SM120 sparse-prefill helper across chunk sizes. Fresh isolated
  helper sweep with `H=32,K=2048,topk=2048` confirmed chunk `64` is still the
  best default among practical sizes:
  - `tokens=64`: chunk `16` `779.142 us`, chunk `32` `394.021 us`, chunk `64`
    `318.614 us`; all bit-exact vs the first run.
  - `tokens=128`: chunk `16` `1492.258 us`, chunk `32` `779.662 us`, chunk
    `64` `667.119 us`, chunk `128` `723.778 us`; all bit-exact.
  - `tokens=256`: chunk `16` `2934.376 us`, chunk `32` `1558.884 us`, chunk
    `64` `1283.972 us`, chunk `128` `1454.805 us`, chunk `256` `1492.165 us`;
    all bit-exact.
  This older sweep was superseded by the post-trim/post-dynamic-workspace sweep
  below; it should not be used to justify reverting the current `chunk=256`
  default.

- Added adaptive sparse-prefill top-k trimming in the SM120 FlashMLA sparse
  prefill bridge (`DG_SM120_PREFILL_TRIM_TOPK=1`). The motivation is to avoid
  doing gather/BMM/softmax work over padded C128/C4 sparse widths when
  `topk_length` says only a much smaller prefix is valid. It is intentionally
  gated by `DG_SM120_PREFILL_TRIM_TOPK_MIN_WIDTH=2048`, because synthetic
  evidence showed trimming 1024-wide rows down to 32 valid entries was slower
  on the tensor-core BMM bridge, while trimming 2048-wide rows was a win:
  - `topk=1024,lens=32,tokens=128,chunk=64`: adaptive no-trim `425.779 us`;
    forced trim `477.078 us` (worse).
  - `topk=1024,lens=32,tokens=256,chunk=64`: adaptive no-trim `776.536 us`.
  - `topk=2048,lens=32,tokens=128,chunk=64`: adaptive trim `476.058 us`;
    no-trim `642.695 us` (`1.35x` faster).
  - `topk=2048,lens=32,tokens=256,chunk=64`: adaptive trim `860.676 us`;
    no-trim previously measured about `1292.653 us` (`1.50x` faster).
  This is a real wasted-work reduction for padded wide sparse prefill, not a
  single-token decode helper tweak. It will only affect live prompts whose
  sparse prefill rows are padded to at least 2048 entries.

- Added dynamic C128 compressed-region sizing in the patched DeepSeek V4
  prefill path (`DG_SM120_PREFILL_DYNAMIC_COMPRESSED_N=1`). vLLM's stock C128
  prefill workspace sizes the compressed region from `max_model_len/128`, so a
  modest prompt under a 128k context still carries a 1024-entry compressed
  offset through workspace allocation and sparse-index combination. The SM120
  patch now sizes each prefill chunk from the chunk's actual sequence length,
  keeps SWA offset tight, and slices C128 top-k columns to the needed aligned
  width. This is a prefill-boundary reduction, not a token-decode helper tweak.
  - First implementation bug: SWA-only layers have `top_k=0`; slicing their
    `topk_indices` to zero columns made Triton reject `tl.arange(0, 0)`. The
    patch now only slices C128 top-k columns on the non-SWA dynamic path.
  - Fresh correctness after the fix: tiny chat returned `OK.` with HTTP 200.
  - With chunk `64`, dynamic sizing was neutral/slightly worse at ~4k
    (`2.094s` TTFT vs `2.089s`) but helped ~8k (`5.078s` vs `5.367s`) and
    ~16k (`16.005s` vs `16.279s`).
  - Follow-up patch stores `prefill_seq_lens_cpu` in
    `DeepseekSparseSWAMetadata` so dynamic workspace sizing can avoid a
    per-layer GPU `seq_lens.max().item()` synchronization while still using the
    exact context+query sequence lengths already available to the metadata
    builder.
  - A second follow-up caches the computed C128 compressed chunk sizes on the
    shared `swa_metadata` object (`_sm120_dynamic_c128_chunk_ns`) so only the
    first C128 prefill layer computes them. Later same-step C128 layers reuse
    the tuple instead of repeating Python/CPU/GPU max work.

- Retuned `DG_SM120_PREFILL_WORKSPACE_CHUNK` from `64` to `256` after the
  adaptive top-k/dynamic-workspace patches changed the bridge cost balance.
  The updated helper sweep at `topk=2048,lens=32,H=32` showed fewer, larger
  chunks are now faster because the effective sparse width is small:
  - `tokens=128`: chunk `64` `436.947 us`, chunk `128` `224.418 us`
  - `tokens=256`: chunk `64` `849.956 us`, chunk `128` `430.700 us`, chunk
    `256` `223.571 us`
  - `tokens=512`: chunk `64` `1669.588 us`, chunk `128` `840.309 us`, chunk
    `256` `429.741 us`
  Live serving with chunk `256` stayed correct and improved long-prompt TTFT:
  - c1 512-token short prompt: `88.83 request tok/s`, `91.37 steady tok/s`
  - c4 256-token short prompt warm repeat: `188.12 aggregate request tok/s`,
    `193.11 aggregate steady tok/s`
  - unique ~4.1k prompt / 16 output: TTFT `1.897s`, prompt-through-TTFT
    `2161 tok/s`, steady decode `86.71 tok/s`
  - unique ~8.2k prompt / 16 output: TTFT `4.762s`, prompt-through-TTFT
    `1721 tok/s`, steady decode `74.67 tok/s`
  - unique ~16.4k prompt / 16 output: TTFT `15.368s`, prompt-through-TTFT
    `1066 tok/s`, steady decode `82.23 tok/s`
  After the CPU-seq-lens metadata restart, correctness still returned `OK.`;
  an 8k prompt measured TTFT `4.942s` and `84.62 tok/s` steady decode, c1
  512-token short prompt measured `84.77 request tok/s` / `86.24 steady tok/s`,
  and c4 256-token warm repeats recovered to `183.37 aggregate request tok/s` /
  `187.87 aggregate steady tok/s`. The first c4 after long-prefill traffic is
  often slower, so use warmed repeats when judging concurrency.
  After the cached-chunk-size patch and restart, correctness again returned
  `OK.`; c1 512-token short prompt measured `87.40 request tok/s` /
  `90.03 steady tok/s`, 8k prompt measured TTFT `4.960s` and
  `84.84 steady tok/s`, and c4 256-token warmed to
  `190.08 aggregate request tok/s` / `198.56 aggregate steady tok/s` with no
  recent error/OOM grep hits. This keeps concurrency healthy while shaving
  repeated C128 prefill metadata work.

- Rechecked the alternate workspace-materialization paths under the current
  trimmed helper. The C++ reusable workspace gather is still much slower
  (`tokens=128/256,topk=2048,lens=32,chunk=64`: about `19.4/38.6 ms`), and
  `torch.index_select(out=...)` remained slightly slower than PyTorch advanced
  indexing (`516.9/920.5 us` vs `488.6/864.3 us`) as a standalone path. Keep
  `DG_SM120_PREFILL_GATHER_WORKSPACE=0`; after live crash testing, also keep
  `DG_SM120_PREFILL_INDEX_SELECT=0` unless specifically debugging the compiled
  BMM bridge.

- Avoided the `torch.full(..., -1)` clear in
  `combine_topk_swa_indices` by allocating `combined_indices` with
  `torch.empty` under `DG_SM120_PREFILL_EMPTY_COMBINED_INDICES`. Synthetic
  isolated combine benchmark measured `31.737 us` -> `18.238 us` (`1.74x`)
  with valid prefixes/lens equal. The first naive live attempt crashed because
  padded/tail entries are gathered before `combined_lens` masks them. The
  production path now forces safe clamped gather indices when empty-combined
  mode is enabled, while still using `combined_lens` for attention validity.
  It is now default-on:
  - naive unsafe attempt: CUDA device-side assert on first request; rejected.
  - safe-tail attempt: correctness returned `OK.`, 8k prompt measured TTFT
    `4.962s` / `89.27 steady tok/s`, c1 512-token short prompt measured
    `88.92 request tok/s` / `91.14 steady tok/s`, and c4 256-token warm repeat
    measured `189.70 aggregate request tok/s` / `196.89 aggregate steady tok/s`.

- Reverted the attempted guarded default-on `DG_SM120_PREFILL_TORCH_COMPILE`
  setting after live validation. Synthetic `K=128` evidence showed about a
  `1.4x` steady-state helper win for `S=64-256`, but the live vLLM service
  returned one degraded ~4k-prompt request and then failed with an illegal CUDA
  memory access on the next long-prompt request. The compiled bridge remains in
  the patcher behind `DG_SM120_PREFILL_TORCH_COMPILE=1` and
  `DG_SM120_PREFILL_TORCH_COMPILE_MIN_ROWS=64`, but production compose defaults
  it off.
  - Fresh post-revert restart evidence: container env shows
    `DG_SM120_PREFILL_TORCH_COMPILE=0` and `DG_SM120_PREFILL_INDEX_SELECT=0`;
    tiny correctness returned `OK.`; c1 short-prompt 256-token streaming measured
    `88.84 request tok/s` / `94.40 steady tok/s`; c1 1024-token streaming repeats
    measured `95.60 request tok/s` / `96.62 steady tok/s` and then
    `94.67 request tok/s` / `95.59 steady tok/s`; two consecutive unique ~4.1k
    prompt / 64-token runs succeeded with TTFT `1.883s` then `1.680s` and steady
    decode `84.63` then `85.95 tok/s`; an ~8.2k prompt / 64-token run measured
    TTFT `4.817s`, prompt-through-TTFT `1703 tok/s`, and steady decode
    `93.62 tok/s`; an ~16.4k prompt / 32-token run measured TTFT `15.352s`,
    prompt-through-TTFT `1069 tok/s`, and steady decode `87.53 tok/s`; warmed
    c4 256-token streaming measured
    `194.10 aggregate request tok/s` / `199.19 aggregate steady tok/s`; recent
    logs had no illegal-memory/OOM/traceback/runtime-error hits.

- Restored production serving after the eager/profile run and verified the service
  is back on CUDA graphs/profiler-off settings (`DG_SM120_KERNEL_PROFILE=0`,
  `VLLM_ENFORCE_EAGER=0`, MTP1 active). Warm post-restore benchmark evidence:
  - c1 1024-token streaming repeat: `91.25 request tok/s`, `92.01 steady tok/s`
  - c4 512-token streaming repeat: `181.22 aggregate request tok/s`,
    `183.00 aggregate steady tok/s`
- Added `scripts/summarize_sm120_profiles.py` to summarize both built-in
  `DG_SM120_KERNEL_PROFILE` TSV output and vLLM/PyTorch Chrome trace JSONs.
  Latest summaries were saved to `profiles/profile_summary_latest.txt`.
- Eager built-in profile of a c1 128-token request showed current decode shapes
  are mostly MTP-shaped dense FP8 `m=2` and MoE routed `m=12` calls. The hottest
  DG-profiled shapes by last-sample latency were dense FP8 `m=2,n=2048,k=4096`
  / `m=2,n=4096,k=4096` (~43-44 us), MoE FP8xFP4 `m=12,n=4096,k=2048`
  (~42 us), BHR `m=2,n=1024,k=4096` (~32 us), and sparse MLA workspace split
  (~19-20 us).
- Added a targeted `scripts/run_ncu_sm120_moe.sh` runner for SM120 FP8xFP4 MoE
  profiling. The current running service image does not have `ncu` on PATH even
  though `Dockerfile.vllm-nightly-sm120` installs `cuda-nsight-compute-13-0`, so
  a targeted NCU attempt exited before profiling. Rebuild the development image
  or use a host/container with `ncu` installed; prior attempts with NCU available
  hit `ERR_NVGPUCTRPERM`, so host GPU performance-counter permissions may still
  need to be fixed before relying on NCU metrics.


- Added default MTP speculative serving config (`docker/vllm_speculative_mtp1_local_argmax.json`) and changed compose's default CUDA graph capture size to `8`. This is the best balanced config found so far: c1 improves from the non-MTP `~69-70 tok/s` range to `~85-86 tok/s`, while warmed c4 aggregate steady decode is about `176 tok/s`.
- Added DeepSeek V4 MTP local-argmax patching in `docker/patch_vllm_deepseekv4.py` so draft token selection avoids full-vocab all-gather. Patching only generic `deepseek_mtp.py` failed; the working path patches `deepseek_v4_mtp.py` as well.
- Added SM120 FP8xFP8 static SFB scale-layout caching in `csrc/sm120_fp8_fp8_cutlass.cu`. Microbenchmarks improved selected dense FP8 small-M shapes by roughly `1.13x-1.36x`; end-to-end c1 remained below target.
- Tested and rejected MTP `num_speculative_tokens=2`/capture-size `16` as a default: warm c1 reached about `92 tok/s`, but c4 aggregate steady decode regressed to about `115 tok/s`.
- Tested and rejected MTP `num_speculative_tokens=3`: c1 steady decode dropped to about `77-81 tok/s`.
- Fixed compose startup plumbing for `VLLM_SPECULATIVE_CONFIG_*` and
  `VLLM_PROFILER_CONFIG_*` by reading runtime env with `printenv`.
  - Regression found after a restart: container env still showed
    `VLLM_SPECULATIVE_CONFIG_FILE=/workspace/DeepGEMM/docker/vllm_speculative_mtp1_local_argmax.json`,
    but the actual `vllm serve` argv lacked `--speculative-config` because
    Docker Compose had interpolated `${VLLM_SPECULATIVE_CONFIG_FILE:-}` to
    empty at compose-render time.
  - Symptom: c1 1024-token streaming dropped to `~68.9 tok/s` and no
    `SpecDecoding metrics` appeared in logs.
  - After the fix, argv includes `--speculative-config
    {"method":"mtp","num_speculative_tokens":1,"use_local_argmax_reduction":true}`
    and logs again show MTP/local-argmax setup plus `SpecDecoding metrics`.
- Gated the SM120 BHR M1-MMA path to `R <= 1024` in
  `csrc/sm120_fp8_gemm_fallback.cu`.
  - Microbench evidence before gating: `B=1,H=32,D=1024,R=4096` with forced
    M1-MMA was about `129.095 us`, while the normal/warp path was about
    `111.9-112.3 us`.
  - Post-gate evidence: forcing the env on `R=4096` routes to the faster path
    and measures about `111.649 us`, while `R=512` still uses M1-MMA at about
    `16.486 us` and `R=1024` remains about `30.819 us`.
- Re-tested the restored default service after the latest rebuild:
  - correctness request returned `OK.`
  - c1 1024-token decode after the unpermute/reduce helper: best observed run
    was about `93.16 request tok/s`, `93.96 steady tok/s`, but repeat checks on
    the same running service are typically `~85-90 tok/s` steady decode, so do
    not treat `93+ tok/s` as a stable baseline.
  - after restoring the compose speculative-config path and applying the BHR
    gate, c1 1024-token streaming measured `86.09/87.30`, `87.31/87.89`, and
    `83.44/84.09` request/steady tok/s in three direct runs; variation tracks
    speculative acceptance and the 300 W power cap.
  - after widening the BHR warp-column heuristic to `R > 1024` and lowering
    row-grouped MoE to `m<=6`, c1 1024-token streaming measured about
    `90.12/91.46`, `90.77/91.42`, and `89.95/90.59` request/steady tok/s
    immediately after warmup; after a later restart the same config warmed at
    about `88-90 tok/s` steady; final row6 warm repeat measured `89.03/89.64` request/steady. Treat `~90 tok/s`, not `100+`, as the current repeatable c1 decode region under the fixed 300 W power cap.
  - c4 512-token streaming warm repeat after the final row-grouped `m<=6` clamp measured `183.38 request tok/s` aggregate and `189.37 steady tok/s` aggregate. Previous warm c4 was about `181.32/183.36`, so c4 remains in the same or slightly better aggregate range while avoiding bad m=8/m=12 row-grouped shapes.
  - 4.1k prompt / 1-token completion TTFT measured `3.049s`
    (`1345 prompt tok/s`), 8.2k measured `7.488s` (`1095 prompt tok/s`), and
    16.4k measured `21.304s` (`769 prompt tok/s`).
- Added SM120 BF16 MoE unpermute/reduce CUDA helper plus
  `scripts/bench_sm120_moe_unpermute_reduce.py`. The helper is patched into
  vLLM for SM120 BF16 hidden sizes at least `6144` and supports expert-map
  routing. It is bit-exact against vLLM's Triton `ep_gather` in synthetic
  tests and gives a modest live c1 decode improvement.
- Tested and rejected a compact MoE permute/scatter CUDA replacement. It was
  reconstruct-correct and slightly faster in isolated `ep_scatter`
  microbenchmarks, but regressed live c1 decode, so the prototype was removed.
- Tested and rejected a dense FP8xFP8 per-row small-M MMA path for broad
  decode-sized `m=2/4/8` shapes. It compiled and was correct, but regressed
  badly versus the current CUTLASS path; examples:
  - `m=2,n=1536,k=4096`: default `30.868 us`, small-M MMA `81.940 us`
  - `m=2,n=16384,k=1024`: default `20.592 us`, small-M MMA `50.057 us`
  The prototype was removed; do not re-add a one-row-per-MMA dense C128 path
  without a new design that fixes the occupancy/register-use problem.

- Tested CUTLASS MoE `TileShape = Shape<_64, _128, _128>` as the likely
  small-M architectural knob suggested by the documented SM120 tile set. It did
  not compile with the current MXFP4/UE8M0 TMA scale layouts: CUTLASS failed
  `TMA requires CTA_Tile and SLayout top-level size equivalence`. Reverted to
  the working `128x128x128` tile. Do not retry this exact one-line tile swap; a
  smaller-M MoE kernel needs a matching scale layout / specialized policy, not
  just a CTA shape change.

- Added `scripts/bench_sm120_moe_chain.py` to measure the larger MoE boundary
  `FC1 -> SiLU/FP8 quant -> FC2`, optionally followed by SM120 weighted
  unpermute/reduce via `--reduce-topk`, with the actual SM120 grouped FP8xFP4
  kernels and fused activation-quant helper. Fresh synthetic evidence:
  - MTP-shaped `m=12,hidden=4096,moe_hidden=2048,active_groups=12,skip_fill`:
    FC1 `96.498-96.541 us`, activation/quant `3.474-3.770 us`, FC2
    `53.417-53.432 us`, separate sum about `153.4-153.7 us`, sequential chain
    about `169.6-170.1 us`. CUDA graph replay of the full chain measured
    `161.924 us` (`0.952x` eager chain).
  - tiny decode-shaped `m=6,active_groups=6,row_grouped_skip_fill`: FC1
    `63.719-64.677 us`, activation/quant `3.451-3.491 us`, FC2
    `39.117-39.392 us`, separate sum about `106.3-107.6 us`, chain about
    `106.8 us`. CUDA graph replay measured `102.612 us` (`0.961x` eager chain).
  - Activation/quant is already small (~3.5 us), and graph replay removes only
    about `4-8 us` from this synthetic chain. The larger fusion opportunity is
    eliminating intermediate global-memory traffic and per-GEMM body/fixed
    overhead across FC1/FC2 rather than only rewriting the SiLU helper or relying
    on graph launch reduction.

- Nsight Compute is not currently available in the running vLLM container
  (`command -v ncu` returned missing), so the latest MoE-chain profiling used
  the built-in `DG_SM120_KERNEL_PROFILE` CUDA-event hooks instead. Fresh
  profiled `m=12` chain evidence with profiling enabled:
  - full chain benchmark: FC1 `117.655 us`, activation/quant `4.807 us`, FC2
    `70.626 us`, separate sum `193.089 us`, eager chain `203.923 us`, graph
    chain `165.613 us`.
  - profile stream shows warm `sm120_moe_fp8_fp4_cutlass` calls around
    `~111-118 us` for `m=12,n=4096,k=4096` and `~60-68 us` for
    `m=12,n=4096,k=2048`; first call was a cold outlier at `1.936 ms`.
  - Profiling overhead inflates absolute times, but the shape ranking matches
    unprofiled evidence: FC1 dominates, FC2 is second, activation/quant is tiny.

- Added `scripts/report_sm120_decode_budget.py` to convert live decode and MoE
  microbench data into a per-token target budget. With fresh c1 steady decode
  `91.45 tok/s`, target `100 tok/s`, and 43 layers, the remaining gap is about
  `0.935 ms/token` or `21.743 us/layer`. The latest measured MTP-shaped MoE
  FC1+FC2 body is about `155.801 us/layer` (`96.541 + 59.260 us`), or
  `6.699 ms` across all layers. A 10-15% MoE-body / larger-boundary fusion win
  is therefore large enough to close most or all of the single-request gap,
  whereas another 1-3 us helper tweak is not.

- Fresh MTP-shaped MoE `m=12` microbenchmarks reinforce that larger-boundary MoE
  fusion is the useful direction, not row-group helper tuning:
  - `m=12,n=4096,k=2048`: default `61.187 us`, skip-fill `59.260 us`,
    row_grouped_skip_fill `60.770 us`, small_m `89.475 us`; all bit-exact.
  - `m=12,n=4096,k=4096`: default `98.647 us`, skip-fill `96.541 us`,
    row_grouped_skip_fill `98.563 us`, small_m `174.857 us`; all bit-exact.
  - row-grouped is useful only for very tiny routed shapes (`m<=6` default); for
    MTP-shaped `m=12`, the CUTLASS GEMM body dominates and helper variants are
    neutral or slower.

- Tested and rejected additional runtime/config knobs as defaults:
  - Under the older scalar workspace sparse-prefill kernels,
    `DG_SM120_PREFILL_WORKSPACE_CHUNK=64` worsened a 516-token prompt TTFT to
    about `4.99s`; after the tensor-core BMM prefill path, chunk `64` is the
    default because it reduces Python/chunk overhead without using a large
    all-prompt workspace.
  - `VLLM_MAX_NUM_BATCHED_TOKENS=6144` with MTP2 failed startup with
    `CUDA_ERROR_ILLEGAL_ADDRESS`; keep `4096`.
  - MTP2 plus `DG_SM120_MAIN_TOPK_CAP=64` did not improve c1 and regressed c4;
    keep MTP1 plus top-k cap `128`.
  - `DG_SM120_SEQUENTIAL_COMPRESSOR=0` was correct but did not improve TTFT or
    decode; keep it enabled for stability.
  - PyTorch `scaled_dot_product_attention` is not a good replacement for the
    per-token sparse prefill workspace shape: a small `S=16` experiment was
    about `9 ms` versus BMM around `0.18 ms`, and `S=64` attempted an
    impractical multi-GB attention allocation.
  - `index_select(out=...)` and cached-workspace variants for the BMM bridge
    were correct but essentially neutral versus trusted advanced indexing in
    synthetic tests (`~360 us` vs `~365 us` at `S=64,H=32,K=2048`). Do not
    spend another cycle on prefill gather mechanics alone; the breakdown points
    at the full fused/direct boundary.
  - Actual patched-vLLM-helper compile test, isolated from the running service:
    noncompiled first `166.6 ms`, warmed `323.444 us`; compiled first `1.722 s`,
    warmed `403.651 us`; output max diff `3.05e-5`, LSE max diff `9.54e-7`.
    Correct but slower in the real helper boundary, so keep compile default-off.
  - Host `nsys --cuda-trace-scope=system-wide` did not capture CUDA kernels from
    the containerized vLLM workers in one attempt (`nsys stats` reported no CUDA
    kernel/API trace data). Use an in-container Nsight install, vLLM profiler,
    or the built-in `DG_SM120_KERNEL_PROFILE` hooks for future kernel-level
    traces.


Latest BHR intermediate-R microbench after widening the warp-column heuristic to `R > 1024` (`B=1,H=32,D=1024`, bit-exact):

- `R=512`: M1-MMA remains fastest, about `16.507 us`.
- `R=1024`: M1-MMA / warp-column both around `30.8 us`.
- `R=1152`: warp-column path about `33.25 us` versus prior generic fallback around `75.9 us`.
- `R=1536`: warp-column path about `43.1 us` versus prior generic fallback around `93.6 us`.
- `R=1920`: warp-column path about `51.8 us` versus prior generic fallback around `116.8 us`.
- `R=4096`: long warp-column about `108.2 us`; keep M1-MMA gated off above `R=1024`.

Additional MoE row-grouped sweep on 2026-04-26 showed the old `m<=16` row-grouped default could hurt wider/concurrent decode shapes:

- `m=6,n=7168,k=4096`: row-grouped skip-fill remains best (`~83 us` vs default `~96 us`).
- `m=8,n=7168,k=4096`: row-grouped skip-fill regressed (`~116.8 us` vs default `~113.4 us`), despite improving narrower shapes; this is why the default is `m<=6` rather than `m<=8`.
- `m=12,n=7168,k=4096`: old row-grouped skip-fill could regress badly (`~235 us` in one sweep); with the `m<=6` clamp it falls back near default (`~156 us`).
- `m=24+`: row-grouped/skip-fill generally regresses and must stay disabled by the max-M guard.

Latest live verification after the default-off cuDNN prefill experiment, without restarting vLLM or changing power limits:

- Health endpoint: HTTP 200.
- Short prompt, `MAX_TOKENS=128`, `CONCURRENCY=1`: request `84.60 tok/s`, steady decode `91.47 tok/s`, TTFT `0.123s`.
- Short prompt, `MAX_TOKENS=128`, `CONCURRENCY=4`: aggregate request `164.65 tok/s`, aggregate steady decode `172.48 tok/s`, per-request steady decode `44.66-54.10 tok/s`.
- Unique ~4.1k-token prompt, `MAX_TOKENS=16`, `CONCURRENCY=1`: TTFT `2.174s`, prompt-through-TTFT `1886 tok/s`, steady decode `86.24 tok/s`.

Latest sparse-prefill helper experiment (`scripts/bench_sm120_prefill_helper_chunks.py`) comparing the current BMM bridge with the experimental cuDNN SDPA bridge while the model remained loaded:

- `tokens=64, topk=2048, chunk=64`: BMM `316.4 us` in one short run / `464.1 us` with explicit gather; cuDNN `400.2 us` / `539.2 us` with explicit gather.
- `tokens=128, topk=2048, chunk=64`: BMM `675.2 us`; cuDNN `821.8 us`.
- `tokens=128, topk=4096, chunk=128`: BMM `1487.0 us`; cuDNN `1428.4 us`, but BMM `chunk=32` was still better at `1349.6 us`.
- Masked/sentinel safety check: current cuDNN SDPA appears to ignore additive bias for masked sparse entries at `head_dim=512`, causing large differences if forced on masked sparse rows. The installed patch therefore requires explicit `DG_SM120_PREFILL_CUDNN_UNMASKED=1`; with `DG_SM120_PREFILL_CUDNN=1` but unmasked disabled, a sentinel test matched the safe BMM bridge exactly (`fallback_diff=0`, all-invalid LSE `-inf`).
- Conclusion: cuDNN SDPA is a useful fused unmasked/reference path and lowers explicit score/prob tensor pressure, but it is not a default throughput win or a masked production path for the current live configuration. Real prefill improvement still needs a direct SM120 sparse attention/compressor kernel that avoids BF16 workspace materialization entirely.

Latest Ralph direct-prefill-map implementation check without restarting vLLM:

- DeepGEMM extension rebuild inside the running container completed after adding
  the required CUDA guard include; `docker/patch_vllm_deepseekv4.py` patched the
  installed vLLM files and `python3 -m py_compile` passed in-container.
- `docker compose config --quiet` passed after adding the default-off direct
  FP8 map environment variables to compose.
- Direct FP8 map correctness/perf (`scripts/bench_sm120_direct_prefill_map.py`):
  - `tokens=16,heads=32,topk=512`: max output diff `0`, max LSE diff `0`;
    current BF16 workspace split `106.847 us`, direct FP8 map `481.008 us`.
  - `tokens=16,heads=32,topk=2048`: max output diff `0`, max LSE diff `0`;
    current BF16 workspace split `400.935 us`, direct FP8 map `1966.068 us`.
- Live service remained healthy on the default path:
  - health endpoint HTTP 200
  - tiny correctness request returned `OK.` with HTTP 200
  - short prompt, `MAX_TOKENS=512`, `CONCURRENCY=1`: request
    `86.46 tok/s`, steady decode `87.98 tok/s`, TTFT `0.114s`
- Conclusion: the opt-in direct FP8 map branch is useful as a correctness and
  integration scaffold, but not a throughput path. The next production sparse
  MLA/compressor step must be a real tensor-core direct sparse attention kernel,
  not scalar direct cache reads.

Latest post-metadata-patch live check without restarting vLLM:

- Installed-file patch/compile in the container passed for `flashmla_sparse.py`.
- Health endpoint remained HTTP 200.
- Short prompt, `MAX_TOKENS=64`, `CONCURRENCY=1`: request `81.15 tok/s`, steady decode `94.88 tok/s`, TTFT `0.124s`.

Latest live check after applying adaptive sparse-prefill top-k trimming and
restarting vLLM on 2026-04-26 (power limits unchanged):

- Restart-to-health completed successfully; health endpoint returned HTTP 200
  after about `37` five-second polling intervals.
- Cold first short-prompt request after restart was slow because graph/model
  warmup dominated: `MAX_TOKENS=128`, `CONCURRENCY=1`, request `34.06 tok/s`,
  steady decode `59.10 tok/s`, TTFT `1.610s`.
- Warm short prompt, `MAX_TOKENS=128`, `CONCURRENCY=1`: request `86.21 tok/s`,
  steady decode `92.40 tok/s`, TTFT `0.110s`.
- Warm short prompt, `MAX_TOKENS=256`, `CONCURRENCY=1`: request `87.10 tok/s`,
  steady decode `90.19 tok/s`, TTFT `0.112s`.
- Warm short prompt, `MAX_TOKENS=512`, `CONCURRENCY=1`: request `88.79 tok/s`,
  steady decode `90.46 tok/s`, TTFT `0.117s`.
- Warm short prompt, `MAX_TOKENS=256`, `CONCURRENCY=4`: aggregate request
  `188.43 tok/s`, aggregate steady decode `192.72 tok/s`, per-request steady
  decode about `49 tok/s`.
- Unique ~4.1k-token prompt, `MAX_TOKENS=16`, `CONCURRENCY=1`: TTFT `2.089s`,
  prompt-through-TTFT `1963 tok/s`, steady decode `85.73 tok/s`.
- Unique ~8.2k-token prompt, `MAX_TOKENS=16`, `CONCURRENCY=1`: TTFT `5.367s`,
  prompt-through-TTFT `1527 tok/s`, steady decode `82.82 tok/s`.
- Unique ~16.4k-token prompt, `MAX_TOKENS=16`, `CONCURRENCY=1`: TTFT
  `16.279s`, prompt-through-TTFT `1007 tok/s`, steady decode `92.47 tok/s`.
- The single-request 100 tok/s target is still not met; this pass improved a
  padded-wide prefill micro-hotspot but did not change the core decode bottleneck
  (MoE FP8xFP4 FC1/FC2 body plus sparse MLA/compressor boundaries).

Latest graph-native sparse-SWA decode metadata pass on 2026-04-26:

- Added `deep_gemm._C.sm120_build_sparse_swa_decode_metadata`, a single SM120
  CUDA launch that builds token-to-request ids, slot-valid flags, decode SWA
  lens tail clears, and decode SWA indices. This replaces the previous SM120
  metadata launch plus vLLM Triton `_compute_swa_indices_and_lens_kernel` path
  when the new extension API is available.
- Correctness checks in `scripts/bench_sm120_compressor_metadata.py` cover both
  2-D and vLLM-style `[max_tokens, 1, window]` decode-index buffers. The first
  restart attempt found that vLLM allocates `decode_swa_indices` as 3-D; the
  extension now accepts both 2-D synthetic and 3-D live layouts.
- Synthetic old-vs-new launch comparison using the installed vLLM Triton kernel:
  for decode counts `1,2,4,8,16,32`, old SM120 metadata + Triton SWA index
  compute was about `9.8-10.0 us`; the fused SM120 metadata+indices launch was
  about `4.10 us`.
- Standalone metadata benchmark after rebuild:
  - `lens=[1,1,1,1]`: fused sparse-SWA decode metadata `4.101 us`; basic
    sparse-SWA metadata-only builder `2.571 us`; CPU sparse-SWA metadata
    fallback `33.184 us`.
  - `lens=[1,1,1,1,1,1,1,1]`: fused sparse-SWA decode metadata `4.098 us`.
- Live vLLM was restarted after the fix and reached healthy state. Tiny API
  correctness returned `OK.`.
- Fresh live throughput after restart and warmup, with power limits unchanged:
  - short prompt, `MAX_TOKENS=512`, `CONCURRENCY=1`: `93.56 tok/s` API
    throughput in a warmed run.
  - short prompt, `MAX_TOKENS=256`, `CONCURRENCY=4`: `179.18 tok/s` aggregate
    API throughput, per-request about `44.8-48.4 tok/s`.
  - unique ~4.1k-token prompt, `MAX_TOKENS=64`, `CONCURRENCY=1`: TTFT
    `1.906s`, prompt-through-TTFT `2151 tok/s`, steady decode `84.93 tok/s`.
- Decode budget after the `93.56 tok/s` warmed API result: current token time
  is about `10.688 ms`, so the remaining `100 tok/s` target gap is about
  `0.688 ms/token` or `16.0 us/layer` over 43 layers. A 10% saving on the
  measured MoE FC1+FC2 body (`~156.99 us/layer`) is about `0.675 ms/token`, so
  a real MoE-body/larger-boundary fusion win is now large enough to plausibly
  close the remaining short-decode gap.
- This is a real graph-native metadata win and moves warmed single-request API
  throughput into the low-90 tok/s range, but it still does not meet the
  `100+ tok/s` single-request target. Remaining high-value work is still the
  larger MoE body/fusion and direct tensor-core sparse MLA/compressor path.

Latest MoE chain budget (`scripts/bench_sm120_moe_chain.py --graph`) while the model remained loaded:

- `m=6, hidden=4096, moe_hidden=2048`: FC1 `64.0 us`, activation quant `3.5 us`, FC2 `39.3 us`, eager chain `107.7 us`, CUDA-graph replay `103.1 us`.
- Fresh MTP-shaped `m=12, hidden=4096, moe_hidden=2048`: FC1 `96.617 us`,
  activation quant `3.527 us`, FC2 `53.322 us`, separate sum `153.467 us`,
  eager chain `170.707 us`, CUDA-graph replay `162.193 us`.
- `m=24, hidden=4096, moe_hidden=2048`: FC1 `181.0 us`, activation quant `3.5 us`, FC2 `95.6 us`, eager chain `293.2 us`, CUDA-graph replay `280.6 us`.
- `scripts/bench_sm120_moe_chain.py` now supports `--reduce-topk` to include
  the SM120 weighted unpermute/reduce boundary after FC2. Fresh
  `m=24,reduce_topk=2`: FC1 `101.085 us`, activation `3.934 us`, FC2
  `55.905 us`, reduce `4.347 us`, eager chain+reduce `181.456 us`, graph
  chain+reduce `169.016 us`. This shows FC2 output reduction is already small
  after the SM120 CUDA helper; fusing FC2 epilogue directly into reduce may save
  a few microseconds, but cannot close the main 100 tok/s gap by itself.
- Conclusion: graph replay only removes a few percent. The larger MoE fusion target remains the grouped FP8xFP4 GEMM bodies and the global-memory boundary between FC1 activation/quant and FC2, not Python launch overhead or another tiny setup kernel.
- Fresh unpermute/reduce check remains correct and useful but is not the main
  remaining decode bottleneck: `tokens=64,hidden=4096,topk=8`, BF16 weights,
  int32 ids, expert map enabled: Triton `ep_gather` `11.460 us` vs SM120 CUDA
  `4.167 us` (`2.75x`), bit-exact.
- Fresh non-helper CUTLASS scheduling/config evidence:
  - Added `scripts/bench_sm120_moe_modes_sweep.py` to make these body-mode
    comparisons reproducible without loading vLLM.
  - `DG_SM120_MOE_RASTER_ORDER=0/1/2` was neutral for
    `m=24,active_groups=12` (`chain_us` roughly `174.9-176.2 us`), so the
    default raster order is unchanged.
  - For no-shrink larger routed shapes, direct expert-group setup is better:
    `m=162,active_groups=64`: skip-fill compact path FC1 `412.775 us`, FC2
    `217.107 us`, chain `639.878 us`; direct expert groups FC1 `406.515 us`,
    FC2 `209.805 us`, chain `619.223 us`.
  - `m=162,active_groups=128` also favored direct groups on chain time
    (`1257.820 us` vs `1283.739 us`) despite a small FC2-only tradeoff.
  - Fresh automated sweep (`scripts/bench_sm120_moe_modes_sweep.py --warmup 5
    --iters 20`) reproduced the same ranking:
    - `m=24,active_groups=12,reduce_topk=2`: skip-fill chain+reduce
      `182.064 us`, direct-groups chain+reduce `181.032 us` (neutral)
    - `m=162,active_groups=64`: skip-fill chain `638.587 us`, direct-groups
      chain `618.308 us`
    - `m=162,active_groups=128`: skip-fill chain `1277.292 us`, direct-groups
      chain `1266.567 us`
  - Therefore compose now defaults
    `DG_SM120_MOE_DIRECT_GROUPS_WHEN_NO_SHRINK=1`; it should not affect
    `m < num_groups` tiny/MTP decode shapes and should help larger concurrent
    routed shapes.

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
- Latest high-value-path checks after the graph-native sparse-SWA metadata
  pass:
  - Fused sparse-SWA decode metadata remains correct and fast:
    `scripts/bench_sm120_compressor_metadata.py --lens 1,1,1,1 --iters 100`
    measured `sm120_sparse_swa_decode_metadata_us=4.087` versus
    `cpu_sparse_swa_metadata_us=35.097` (`13.43x`). Compressor metadata
    fused fill+clamp measured `2.422 us` versus CPU repeat/copy+clamp
    `26.660 us` (`11.0x`).
  - The model stayed up while extension-only experiments ran; fresh API checks
    against the already-loaded service were `256` decode tokens at
    concurrency `1`: `90.68 tok/s` aggregate / `90.84 tok/s` request, and
    concurrency `4`: `185.51 tok/s` aggregate / `46.43-48.28 tok/s`
    per request.
  - Rejected a broader CUTLASS K-tile change from `128` to `256`: it compiled
    only with explicit `StageCount<2>`, but the resulting kernel failed the
    packed-scale FP4 CUTLASS path at runtime. The change was reverted before
    continuing.
  - Rechecked the larger MoE-chain boundary after the revert:
    `m=6,active_groups=6,reduce_topk=2` measured FC1 `70.249 us`,
    activation `3.979 us`, FC2 `51.648 us`, reduce `4.295 us`,
    CUDA-graph chain+reduce `121.583 us`. This reinforces that a useful MoE
    speedup needs a persistent/fused routed expert body, not another launch
    tweak.
  - Rejected enabling the direct FP8 sparse-prefill workspace-map prototype:
    synthetic correctness was exact for the test case, but it was much slower
    than the BF16 workspace split path (`tokens=16,topk=512`: `481.557 us`
    direct map vs `106.693 us` workspace; `tokens=64,topk=512`: `1851.865 us`
    vs `363.810 us`). A production direct path must be tensor-core/FlashMLA
    style rather than scalar cache reads.
- Latest graph-native sparse-SWA prefill metadata pass:
  - Added `deep_gemm._C.sm120_build_sparse_swa_prefill_metadata` and patched
    vLLM `DeepseekSparseSWAMetadataBuilder._build_deepseek_v4_metadata` to use
    it on SM120 before falling back to the stock Triton
    `_compute_prefill_metadata_kernel`.
  - Rebuilt the extension in the CUDA 13 vLLM container; only
    `sm120_metadata.cu` and `python_api.cpp` rebuilt for this pass.
  - Correctness and microbench:
    `scripts/bench_sm120_compressor_metadata.py --lens 1,1,1,1 --iters 100
    --warmup 20 --prefill-num-decodes 1` measured sparse-SWA prefill metadata
    `2.535 us` versus the Torch reference `27.862 us`; fused sparse-SWA decode
    metadata stayed `4.107 us`.
  - vLLM was restarted once after the safe metadata patch; `/health` reached
    HTTP 200 and a tiny chat correctness request returned `OK.`.
  - Fresh warmed live API evidence after that restart:
    - shell benchmark, short prompt, `MAX_TOKENS=512`, `CONCURRENCY=1`:
      `91.30 tok/s`.
    - shell benchmark, short prompt, `MAX_TOKENS=512`, `CONCURRENCY=4`:
      `181.57 tok/s` aggregate.
    - streaming short prompt, `MAX_TOKENS=256`, `CONCURRENCY=1`: TTFT
      `0.143s`, steady decode `85.17 tok/s`.
    - streaming unique `4100`-token prompt, `MAX_TOKENS=64`,
      `CONCURRENCY=1`: TTFT `1.871s`, prompt-through-TTFT `2191 tok/s`,
      steady decode `80.12 tok/s`.
  - This removes another graph metadata launch from the prefill path, but it
    still does not close the single-request `100+ tok/s` decode gap. The
    remaining high-value work is still a larger MoE body/fusion win or direct
    tensor-core sparse MLA/compressor implementation.
- Fresh MoE row-grouped range check after the metadata pass:
  - Synthetic chain checks suggested extending
    `DG_SM120_MOE_ROW_GROUPED_MAX_M` from `6` to `16` could help
    MTP-shaped `m=12` and `m=16` routed-M shapes:
    - `m=12,active_groups=12`: graph chain+reduce `167.978 us` at max-M `6`
      vs `164.832 us` at max-M `12`/`16`.
    - `m=16,active_groups=16`: graph chain+reduce `184.489 us` at max-M `6`
      vs `181.239 us` at max-M `16`.
  - Live vLLM disproved this as a useful default: after a rebuild/restart with
    max-M `16`, short-prompt `MAX_TOKENS=512,CONCURRENCY=1` fell to about
    `87.9-88.4 tok/s` and `CONCURRENCY=4` fell to about `162.5 tok/s`
    aggregate. The change was reverted; keep the default max-M at `6` unless a
    future profile proves the live routed shape distribution changed.

## Next Work

1. Build a production SM120 sparse MLA prefill kernel.
   The tensor-core PyTorch BMM bridge is much faster than the scalar kernels,
   and the direct FP8 workspace-map prototype showed scalar direct cache reads
   are even slower. The final target should avoid Python loops, avoid full BF16
   workspace materialization, and compute directly from the cache/sparse indices
   with a tensor-core/FlashMLA-equivalent SM120 implementation.

2. Continue validating the BMM prefill bridge under live traffic.
   Compose now defaults to `DG_SM120_PREFILL_WORKSPACE_CHUNK=256`,
   `DG_SM120_PREFILL_TORCH_BMM=1`, and `DG_SM120_PREFILL_TRUST_INDICES=1`;
   recent token-counted checks cover about `4k`, `8k`, and `16k` uncached
   prompt tokens. Keep using unique prompts from token 0 so prefix caching does
   not hide prefill cost.

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
- Never change the RTX PRO 6000 power limits. They are intentionally capped at
  `300 W` by the user to limit heat; full `600 W` operation overheats this
  system. Treat the cap as a fixed deployment constraint, not a tuning knob.
- `VLLM_*` local config variables produce "unknown vLLM environment variable"
  warnings. They are consumed by the compose startup shell before `vllm serve`,
  so the warnings are noisy but expected.
- Keep `DG_SM120_KERNEL_PROFILE=0` for real benchmarks. The kernel profiler uses
  CUDA event synchronization and will distort throughput when enabled.
- Keep `DG_SM120_BYPASS_TP_ALLREDUCE=0` for real benchmarks. The bypass is
  deliberately invalid and only measures how much TP allreduce costs.
