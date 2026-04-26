# SGLang DeepSeek-V4 Implementation Notes for SM120/vLLM Agents

Date: 2026-04-26

This report summarizes interesting ideas from the LMSYS/SGLang DeepSeek-V4 day-0 implementation and ranks what is most likely reusable for this DeepGEMM/vLLM SM120 fork targeting `deepseek-ai/DeepSeek-V4-Flash` on 2x RTX PRO 6000 Blackwell workstation GPUs.

Primary sources:

- LMSYS blog: <https://www.lmsys.org/blog/2026-04-25-deepseek-v4/>
- SGLang DeepSeek-V4 cookbook: <https://docs.sglang.io/cookbook/autoregressive/DeepSeek/DeepSeek-V4>
- SGLang roadmap issue: <https://github.com/sgl-project/sglang/issues/23602>
- Temporary inspected SGLang source clone: `/tmp/sglang-dsv4-inspect`, commit `7d495644313407bd2daa935db000149dd6c421ba`

## Executive Summary

SGLang's DeepSeek-V4 implementation is interesting because it attacks the same two bottleneck regimes this fork sees:

1. **Short/medium decode throughput**: lots of under-occupied small kernels, hybrid-attention metadata overhead, sparse attention/indexing, and MoE fixed overhead.
2. **Long-prompt TTFT / prompt processing**: expensive compressed attention prefill, compressor/indexer work, and sparse KV/cache management.

The most useful ideas are architectural rather than drop-in code. SGLang's public implementation and launch recipes target B200/GB200/GB300/H200-class deployments, and the SGLang roadmap still lists SM120 support as a potential item. Therefore, do not assume SGLang is a ready-made RTX PRO 6000 solution. Treat it as a reference for fusion boundaries, scheduling, and metadata handling.

## Ranked Opportunities

| Rank | Opportunity | Expected Value for This Fork | Porting Difficulty | Notes |
|---:|---|---|---|---|
| 1 | Device-side / CUDA-graph-native metadata prep for hybrid attention and MTP | High for decode and speculative decode | Medium-high | SGLang identifies metadata prep as a launch bottleneck and moves it into graph/device kernels. |
| 2 | Flash Compressor-style fused compression kernel | Very high for TTFT/prompt processing | High | Directly addresses current workspace/materialization-heavy prefill path. |
| 3 | Fused hybrid attention: SWA + C4/C128 in one FlashMLA-style call | High | Very high | Conceptually right, but current upstream FlashMLA support targets SM90/SM100, not SM120. |
| 4 | Full MoE layer fusion / Mega MoE boundary | High for decode | Very high | Fuse dispatch, FC1, activation, FC2, and combine instead of only optimizing helper kernels. |
| 5 | Lightning TopK-style long-context radix select | Medium-high for long context | Medium-high | Useful if top-k/indexer shows up as a long-context bottleneck. |
| 6 | HiSparse / ShadowRadix cache architecture | High for capacity and prefix reuse, lower for short decode | Very high | Valuable but invasive; likely not a short path to 100+ tok/s. |
| 7 | SGLang SM120/NVFP4 CUTLASS JIT kernels as implementation reference | Medium | Medium | Useful patterns for SM120 CUTLASS setup, but DeepSeek-V4 Flash MoE uses MXFP4/UE8M0-style expert weights, not necessarily NVFP4. |

## 1. In-Graph / Device-Side Metadata Preparation

### What SGLang Does

The LMSYS blog says DeepSeek-V4 hybrid attention metadata is heavy: SWA page indices, shadow-mapped pool slots, compressor/indexer plans, and per-pool write locations. SGLang avoids preparing this eagerly on the scheduler/Python path by rebuilding metadata inside captured CUDA graph/device kernels.

Relevant inspected SGLang files:

- `/tmp/sglang-dsv4-inspect/python/sglang/jit_kernel/fused_metadata_copy.py`
- `/tmp/sglang-dsv4-inspect/python/sglang/jit_kernel/csrc/elementwise/fused_metadata_copy.cuh`
- `/tmp/sglang-dsv4-inspect/python/sglang/srt/layers/attention/nsa/nsa_backend_mtp_precompute.py`
- `/tmp/sglang-dsv4-inspect/python/sglang/srt/layers/attention/nsa_backend.py`

The JIT fused metadata kernel combines several copy/update operations into one kernel launch and includes a multi-backend variant for speculative decoding. In `nsa_backend.py`, the precomputed metadata is copied into CUDA graph metadata through this fused path, with fallback to individual tensor copies if the JIT kernel is unavailable.

### Why It Matters Here

This fork already uses MTP1/local argmax as a default and sees decode improvement, but MTP2/MTP3 have not produced stable end-to-end wins. SGLang's implementation suggests the missing piece may be not only draft-token count, but the **per-step hybrid-attention metadata path** under speculative decode.

If vLLM is still doing repeated CPU/Python or multiple small tensor copy/update operations for DeepSeek-V4 attention metadata at replay time, that overhead can cap the benefit from speculative decode.

### Candidate Work for This Fork

- Profile vLLM per-step metadata preparation under CUDA graphs and MTP.
- Identify repeated tensor copies/cumsums/page-table transforms/index transforms in the decode path.
- Build one or more small CUDA kernels to generate/copy DeepSeek-V4 attention metadata directly from fixed graph inputs.
- Prefer one general graph-safe kernel over many narrow special-case helpers.

### Risk

This is framework-integration-heavy and may touch vLLM internals. It is not a simple DeepGEMM-only kernel swap.

## 2. Flash Compressor / Fused Compression

### What SGLang Does

The LMSYS blog describes a “Flash Compressor” for compressed attention. It fuses the compression chain into an on-chip pass, reducing HBM round trips and avoiding the naive multi-stage pipeline. The blog claims large improvement over a naive PyTorch path on H200.

### Why It Matters Here

Current long-prompt TTFT remains a major weakness in this fork. The existing SM120 prefill path still relies on BF16 workspace materialization/chunking and repeated per-layer work. That is exactly the class of path SGLang is trying to avoid with a fused compressor.

### Candidate Work for This Fork

Build a production SM120 compressor kernel for the DeepSeek-V4 compressed attention path:

- Read source hidden/KV inputs once.
- Compute compression scores and softmax/weighted reduction on-chip where possible.
- Write compressed KV directly into the target cache layout.
- Avoid full intermediate BF16 workspace materialization.
- Support both C4 and C128 compressor behavior if their formulas/layouts differ.

### Priority

Very high for prompt processing. This is probably more important for 4k/8k/16k+ prompts than additional tiny decode helper optimizations.

## 3. Fused Hybrid Attention: SWA + C4/C128 Extra Attention

### What SGLang Does

SGLang integrates a refreshed FlashMLA interface where DeepSeek-V4's sliding-window attention and extra compressed attention run in a single fused kernel call. The kernel accepts both normal KV and extra compressed KV/cache indices, sharing metadata construction in the forward pass.

The blog explicitly says the target GPUs are Hopper/SM90 and Blackwell/SM100 where the corresponding kernels are supported.

### Why It Matters Here

This fork has SM120 sparse MLA decode/prefill support, but it still contains fallback/workspace-heavy paths. Running SWA and extra attention through separate or partially-emulated paths leaves overhead in:

- extra launches,
- extra metadata preparation,
- workspace gather/materialization,
- repeated softmax/output staging.

### Candidate Work for This Fork

Use SGLang's fusion boundary as the target design:

- One SM120 kernel or tightly coordinated kernel sequence for SWA + C4/C128 extra attention.
- Direct reads from FP8/MLA cache plus sparse/dense compressed indices.
- No full BF16 gathered workspace for normal operation.
- Support decode first, then prefill.

### Risk

High. This may require a real SM120 FlashMLA-equivalent implementation rather than adapting the current fallback kernels.

## 4. Mega MoE / Full MoE Layer Fusion

### What SGLang Does

The blog describes DeepGEMM Mega MoE integration on Blackwell. The fused boundary is:

```text
EP dispatch -> FP8xFP4 FC1 -> SwiGLU -> FP8xFP4 FC2 -> EP combine
```

It also mentions avoiding two resident copies of expert tensors by adapting the transformed FP4/UE8M0 scale layout directly.

### Why It Matters Here

This fork's decode profile has repeatedly pointed at tiny-M FP8xFP4 routed expert GEMMs and per-layer fixed overhead. Microbenchmarks show helper/setup overhead has been reduced enough that the remaining cost is mostly inside grouped GEMM bodies and repeated layer orchestration.

Therefore, SGLang's Mega MoE direction supports the conclusion that the next meaningful MoE improvement is a **larger fusion boundary**, not more small helper tweaks.

### Candidate Work for This Fork

A production-worthy SM120 MoE path should aim to fuse at least:

1. routing/permute or compact route map consumption,
2. FC1 grouped expert GEMM,
3. SwiGLU/clamp/activation quantization,
4. FC2 grouped expert GEMM,
5. weighted unpermute/reduce.

If full fusion is too large initially, prototype a persistent routed expert kernel that keeps the expert tile scheduling and intermediate activation resident long enough to avoid round-tripping through global memory between FC1 activation and FC2.

### Notes

- Do not focus only on single-token or tiny-batch special cases. SGLang's value is in the architectural fusion boundary.
- The existing per-GEMM row-grouped/skip-fill work is useful but not sufficient.

## 5. Lightning TopK / Radix Select for Sparse Indexer

### What SGLang Does

The blog describes a Lightning TopK kernel for sparse attention index selection. For long contexts, a naive global sort can take over 100 us even at batch size 1. SGLang uses a radix-select design with CTA-local histograms and cluster reduction to reduce small-batch top-k latency.

Relevant inspected files in SGLang include generic fast top-k wrappers:

- `/tmp/sglang-dsv4-inspect/sgl-kernel/python/sgl_kernel/top_k.py`
- `/tmp/sglang-dsv4-inspect/sgl-kernel/CMakeLists.txt` includes `csrc/elementwise/topk.cu`

### Why It Matters Here

This fork currently has sparse top-k fallback/patching for SM120 and small/full-context behavior. If long-context decode or prefill profiles show top-k/indexer latency growing with sequence length, a radix-select kernel is likely a better direction than PyTorch `topk` or generic persistent-topk fallbacks.

### Candidate Work for This Fork

- Profile indexer top-k separately at 4k, 32k, 128k, and beyond.
- If top-k is hot, build a DeepSeek-V4-specific top-k=512 or top-k=2048 selector tuned for SM120.
- Include index transformation/page-table write in the same kernel if possible.

## 6. HiSparse and ShadowRadix

### What SGLang Does

SGLang introduces ShadowRadix for hybrid attention prefix caching. It maps virtual full-token positions into separate physical pools for SWA, C4, C128, and compression-state rings. This allows SWA slots to be released as the window moves while C4/C128 compressed shadows remain reusable.

SGLang also uses HiSparse to offload inactive C4 KV to CPU memory. The blog says C4 benefits because each step touches only a small fraction of compressed positions; C128 is dense and SWA is already small, so neither benefits the same way.

Relevant inspected SGLang files:

- `/tmp/sglang-dsv4-inspect/python/sglang/srt/managers/hisparse_coordinator.py`
- `/tmp/sglang-dsv4-inspect/python/sglang/srt/mem_cache/hisparse_memory_pool.py`
- `/tmp/sglang-dsv4-inspect/python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py`

### Why It Matters Here

This is most relevant to memory capacity and long-context serving. It does not directly solve short single-request decode throughput, but it could help with:

- fitting useful context windows,
- maintaining throughput at longer contexts,
- prefix-cache reuse,
- reducing GPU KV pressure for C4 layers.

### Porting Assessment

Very invasive. This likely requires vLLM cache-manager changes, not only DeepGEMM kernels.

## 7. SGLang SM120/NVFP4 CUTLASS Code as Reference

The inspected SGLang tree contains JIT NVFP4 kernels with explicit SM120 support:

- `/tmp/sglang-dsv4-inspect/python/sglang/jit_kernel/nvfp4.py`
- `/tmp/sglang-dsv4-inspect/python/sglang/jit_kernel/csrc/gemm/nvfp4/nvfp4_scaled_mm_sm120.cuh`
- `/tmp/sglang-dsv4-inspect/python/sglang/jit_kernel/csrc/moe/nvfp4_blockwise_moe.cuh`

Interesting details:

- The JIT code compiles architecture-family-specific targets with an `a` suffix through `override_jit_cuda_arch(..., suffix="a")`.
- SM120 scaled FP4 GEMM defines small-M and larger-M configs, including `Shape<_128, _128, _256>` for small M in one dense GEMM path.
- The blockwise MoE path has an SM120 grouped GEMM using CUTLASS `OpClassBlockScaledTensorOp`, pointer arrays, per-expert scale layouts, and cached async workspace.

Caveat: this is NVFP4-oriented. DeepSeek-V4 Flash's expert weights are MXFP4/UE8M0-style in the current vLLM/DeepGEMM path. Use these files as CUTLASS/SM120 integration examples, not as a direct replacement.

## SGLang Launch Recipe Signals

SGLang's public cookbook gives these high-level tuning signals:

- DeepSeek-V4 Flash is listed as a 284B model intended for B200/GB200/GB300/H200-class serving, generally with TP=4 in the public recipes.
- Low-latency recipes use EAGLE speculative decode with `num-steps=3`, `topk=1`, and `num-draft-tokens=4`.
- Balanced recipes use lighter speculation: `num-steps=1`, `num-draft-tokens=2`.
- Max-throughput recipes disable MTP because verify overhead can outweigh draft benefits at saturation.
- Context-parallel recipes are specifically for long-context prefill.

For this 2x SM120 workstation target, do not copy these values blindly. But the recipe structure is useful:

- Latency-sensitive decode and throughput-saturated serving likely want different speculative settings.
- Long-context prefill likely needs a separate path from decode tuning.
- Higher draft-token counts only pay off if the metadata and verification path are cheap enough.

## What Not To Over-Interpret

- SGLang's blog claims strong B200/H200 numbers, but those are not SM120 RTX PRO 6000 numbers.
- Their FlashMLA path explicitly targets SM90/SM100 support. SM120 still needs custom kernels.
- Their Mega MoE path depends on deployment assumptions such as DeepEP/symmetric memory that may not map cleanly to a 2-GPU workstation.
- Their roadmap still lists SM120 support as potential future work, so upstream SGLang is not a complete SM120 answer today.

## Recommended Next Agent Tasks

### Task A: Metadata Path Audit

Goal: find repeated vLLM DeepSeek-V4 metadata operations during decode/MTP that can be fused or moved into CUDA graph replay.

Look for:

- page-table copies/transforms,
- sequence length cumsums,
- sparse index metadata setup,
- FlashMLA/MLA scheduler metadata construction,
- operations repeated separately for draft and verify passes.

Deliverable: a ranked list of metadata ops by frequency and likely latency, with proposed fusion boundaries.

### Task B: Flash Compressor Design Sketch

Goal: design an SM120 fused compressor replacing workspace-heavy prefill pieces.

Deliverable should specify:

- exact inputs/outputs and cache layout,
- C4 vs C128 differences,
- softmax/reduction strategy,
- per-token/per-group parallelization,
- memory traffic estimate vs current workspace path,
- synthetic correctness benchmark plan.

### Task C: MoE Fusion Feasibility Study

Goal: decide whether to build a partial or full fused MoE layer kernel.

Deliverable should compare:

- current two grouped-GEMM plus helper-kernel path,
- persistent grouped-GEMM with activation in between,
- full dispatch→FC1→activation→FC2→combine mega-kernel,
- expected memory traffic and launch-count savings.

### Task D: Long-Context TopK Profiling

Goal: determine whether LightningTopK-style work is worth doing now.

Benchmark indexer/top-k latency at representative context lengths:

- 4k,
- 16k,
- 32k,
- 128k,
- as high as practical.

If top-k exceeds sparse attention/compressor costs, build a radix-select prototype.

## Bottom Line

The most actionable insight from SGLang is that DeepSeek-V4 performance is not won by isolated GEMM tuning alone. Their implementation attacks whole-path overhead: graph-native metadata, fused compression, fused hybrid attention, fused MoE, and long-context-aware cache/index management.

For this SM120 vLLM fork, the highest-value work is likely:

1. graph/device-native hybrid-attention metadata,
2. production fused compressor for prefill,
3. direct SM120 fused sparse MLA attention without BF16 workspace materialization,
4. larger-boundary MoE fusion.

These should take priority over more narrowly gated single-shape optimizations.
