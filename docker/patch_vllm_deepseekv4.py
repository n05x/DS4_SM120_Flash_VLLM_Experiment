import re
from pathlib import Path


def patch_cuda_platform_deep_gemm_sm120() -> None:
    path = Path("/usr/local/lib/python3.12/dist-packages/vllm/platforms/cuda.py")
    source = path.read_text()
    old = '        return cls.is_device_capability(90) or cls.is_device_capability_family(100)\n'
    new = (
        '        return (cls.is_device_capability(90) or '
        'cls.is_device_capability_family(100) or cls.is_device_capability(120))\n'
    )
    if old not in source:
        if "cls.is_device_capability(120)" in source:
            return
        raise RuntimeError(f"Could not patch DeepGEMM SM120 support in {path}")
    path.write_text(source.replace(old, new, 1))


def patch_block_scaled_mm() -> None:
    path = Path(
        "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/"
        "kernels/linear/scaled_mm/BlockScaledMMLinearKernel.py"
    )
    source = path.read_text()
    old = """        new_weight, new_weight_scale = process_fp8_weight_block_strategy(
            params.weight,
            weight_scale,
        )

        replace_parameter(layer, params.WEIGHT, new_weight.data)
"""
    new = """        new_weight, new_weight_scale = process_fp8_weight_block_strategy(
            params.weight,
            weight_scale,
        )
        if new_weight_scale.dtype == torch.float8_e8m0fnu:
            # torch stable custom-op conversion in this image does not accept
            # UE8M0 scale tensors, while the SM120 Cutlass path accepts fp32.
            new_weight_scale = new_weight_scale.to(torch.float32)

        replace_parameter(layer, params.WEIGHT, new_weight.data)
"""
    if old not in source:
        if "new_weight_scale.dtype == torch.float8_e8m0fnu" in source:
            return
        raise RuntimeError(f"Could not patch {path}")
    path.write_text(source.replace(old, new))


def patch_flashinfer_mxfp4_sm120() -> None:
    path = Path(
        "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/"
        "layers/fused_moe/experts/trtllm_mxfp4_moe.py"
    )
    source = path.read_text()
    old = (
        "        return p.is_cuda() and p.is_device_capability_family(100) "
        "and has_flashinfer()\n"
    )
    new = (
        "        return (p.is_cuda() and has_flashinfer() and "
        "(p.is_device_capability_family(100) or p.is_device_capability(120)))\n"
    )
    if old not in source:
        if "p.is_device_capability(120)" in source:
            return
        raise RuntimeError(f"Could not patch FlashInfer MXFP4 SM120 support in {path}")
    path.write_text(source.replace(old, new, 1))


def patch_triton_mxfp4_sm120() -> None:
    candidates = (
        Path(
            "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/"
            "layers/fused_moe/experts/gpt_oss_triton_kernels_moe.py"
        ),
        Path(
            "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/"
            "layers/fused_moe/gpt_oss_triton_kernels_moe.py"
        ),
    )
    path = next((candidate for candidate in candidates if candidate.exists()), None)
    if path is None:
        raise RuntimeError("Could not find Triton MXFP4 experts module")
    source = path.read_text()
    old = "        return (9, 0) <= (cap.major, cap.minor) < (11, 0)\n"
    new = "        return (9, 0) <= (cap.major, cap.minor) < (13, 0)\n"
    if old not in source:
        if "< (13, 0)" in source:
            return
        raise RuntimeError(f"Could not patch Triton MXFP4 SM120 support in {path}")
    path.write_text(source.replace(old, new))


def patch_deep_gemm_fp4_sm120() -> None:
    path = Path(
        "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/"
        "layers/fused_moe/experts/deep_gemm_moe.py"
    )
    source = path.read_text()
    old = (
        "        return (\n"
        "            is_deep_gemm_supported()\n"
        "            and current_platform.is_device_capability_family(100)\n"
        "        )\n"
    )
    new = (
        "        return (\n"
        "            is_deep_gemm_supported()\n"
        "            and (\n"
        "                current_platform.is_device_capability_family(100)\n"
        "                or current_platform.is_device_capability(120)\n"
        "            )\n"
        "        )\n"
    )
    if old not in source:
        if "current_platform.is_device_capability(120)" in source:
            return
        raise RuntimeError(f"Could not patch DeepGEMM FP4 SM120 support in {path}")
    path.write_text(source.replace(old, new, 1))


def patch_deep_gemm_e8m0_oracle_sm120() -> None:
    path = Path("/usr/local/lib/python3.12/dist-packages/vllm/utils/deep_gemm.py")
    source = path.read_text()
    old = (
        "            if current_platform.is_device_capability_family(100)\n"
        "            else cls.FLOAT32_CEIL_UE8M0\n"
    )
    new = (
        "            if (\n"
        "                current_platform.is_device_capability_family(100)\n"
        "                or current_platform.is_device_capability(120)\n"
        "            )\n"
        "            else cls.FLOAT32_CEIL_UE8M0\n"
    )
    if old not in source:
        if "current_platform.is_device_capability(120)" in source:
            return
        raise RuntimeError(f"Could not patch DeepGEMM E8M0 oracle in {path}")
    path.write_text(source.replace(old, new, 1))


def patch_deep_gemm_mxfp4_scale_prepack_sm120() -> None:
    path = Path(
        "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/"
        "layers/fused_moe/oracle/mxfp4.py"
    )
    source = path.read_text()
    if "sm120_prepack_fp8_fp4_sfb" in source:
        return

    old_gpt_oss = '''    if mxfp4_backend == Mxfp4MoeBackend.DEEPGEMM_MXFP4:
        from vllm.model_executor.layers.quantization.utils.fp8_utils import (
            _upcast_e8m0_to_fp32,
        )

        return (
            w13_weight.data,
            w2_weight.data,
            _upcast_e8m0_to_fp32(w13_weight_scale.data),
            _upcast_e8m0_to_fp32(w2_weight_scale.data),
            w13_bias,
            w2_bias,
        )
'''
    new_gpt_oss = '''    if mxfp4_backend == Mxfp4MoeBackend.DEEPGEMM_MXFP4:
        from vllm.model_executor.layers.quantization.utils.fp8_utils import (
            _upcast_e8m0_to_fp32,
        )

        if torch.cuda.get_device_capability(w13_weight_scale.device)[0] >= 12:
            import deep_gemm

            def _sm120_prepack(scale: torch.Tensor, n: int, k: int) -> torch.Tensor:
                raw = scale if scale.dtype == torch.float32 else scale.view(torch.uint8)
                return deep_gemm._C.sm120_prepack_fp8_fp4_sfb(raw, 128, n, k)

            return (
                w13_weight.data,
                w2_weight.data,
                _sm120_prepack(
                    w13_weight_scale.data,
                    w13_weight.shape[1],
                    w13_weight.shape[2] * 2,
                ),
                _sm120_prepack(
                    w2_weight_scale.data,
                    w2_weight.shape[1],
                    w2_weight.shape[2] * 2,
                ),
                w13_bias,
                w2_bias,
            )

        return (
            w13_weight.data,
            w2_weight.data,
            _upcast_e8m0_to_fp32(w13_weight_scale.data),
            _upcast_e8m0_to_fp32(w2_weight_scale.data),
            w13_bias,
            w2_bias,
        )
'''
    old_generic = '''    if mxfp4_backend == Mxfp4MoeBackend.DEEPGEMM_MXFP4:
        from vllm.model_executor.layers.quantization.utils.fp8_utils import (
            _upcast_e8m0_to_fp32,
        )

        # Weights stay as uint8 packed FP4 — no layout change needed.
        # Convert E8M0 uint8 scales to float32.
        return (
            w13_weight.data,
            w2_weight.data,
            _upcast_e8m0_to_fp32(w13_weight_scale.data),
            _upcast_e8m0_to_fp32(w2_weight_scale.data),
            w13_bias,
            w2_bias,
        )
'''
    new_generic = '''    if mxfp4_backend == Mxfp4MoeBackend.DEEPGEMM_MXFP4:
        from vllm.model_executor.layers.quantization.utils.fp8_utils import (
            _upcast_e8m0_to_fp32,
        )

        if torch.cuda.get_device_capability(w13_weight_scale.device)[0] >= 12:
            import deep_gemm

            def _sm120_prepack(scale: torch.Tensor, n: int, k: int) -> torch.Tensor:
                raw = scale if scale.dtype == torch.float32 else scale.view(torch.uint8)
                return deep_gemm._C.sm120_prepack_fp8_fp4_sfb(raw, 128, n, k)

            return (
                w13_weight.data,
                w2_weight.data,
                _sm120_prepack(
                    w13_weight_scale.data,
                    w13_weight.shape[1],
                    w13_weight.shape[2] * 2,
                ),
                _sm120_prepack(
                    w2_weight_scale.data,
                    w2_weight.shape[1],
                    w2_weight.shape[2] * 2,
                ),
                w13_bias,
                w2_bias,
            )

        # Weights stay as uint8 packed FP4; non-SM120 DeepGEMM still expects fp32 scales.
        return (
            w13_weight.data,
            w2_weight.data,
            _upcast_e8m0_to_fp32(w13_weight_scale.data),
            _upcast_e8m0_to_fp32(w2_weight_scale.data),
            w13_bias,
            w2_bias,
        )
'''
    replaced = 0
    if old_gpt_oss in source:
        source = source.replace(old_gpt_oss, new_gpt_oss, 1)
        replaced += 1
    if old_generic in source:
        source = source.replace(old_generic, new_generic, 1)
        replaced += 1
    if replaced == 0:
        raise RuntimeError(f"Could not patch DeepGEMM MXFP4 prepack in {path}")
    path.write_text(source)


def patch_sm120_b12x_mxfp4_load_transform() -> None:
    """Optionally load DeepGEMM MXFP4 weights in b12x's fused-MoE layout."""
    if __import__("os").environ.get("DG_SM120_ENABLE_B12X_MOE", "0").lower() not in (
        "1",
        "true",
        "yes",
        "on",
    ):
        return
    path = Path(
        "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/"
        "layers/fused_moe/oracle/mxfp4.py"
    )
    source = path.read_text()

    if "import os\n" not in source:
        if "from enum import Enum\n" in source:
            source = source.replace(
                "from enum import Enum\n", "from enum import Enum\nimport os\n", 1
            )
        else:
            source = "import os\n" + source

    if "_sm120_b12x_scale" in source:
        path.write_text(source)
        return

    needle = """        if torch.cuda.get_device_capability(w13_weight_scale.device)[0] >= 12:
            import deep_gemm

"""
    insert = """        if (
            torch.cuda.get_device_capability(w13_weight_scale.device)[0] >= 12
            and os.environ.get("DG_SM120_ENABLE_B12X_MOE", "0").lower()
            in ("1", "true", "yes", "on")
        ):
            from b12x.cute.fp4 import swizzle_block_scale

            def _sm120_b12x_scale(
                scale: torch.Tensor, rows: int, cols: int
            ) -> torch.Tensor:
                if scale.dim() != 3:
                    raise RuntimeError(
                        "DG_SM120_ENABLE_B12X_MOE requires raw 3D MXFP4 scales; "
                        f"got shape {tuple(scale.shape)}"
                    )
                if scale.shape[1] != rows:
                    raise RuntimeError(
                        "Unexpected MXFP4 scale row count for b12x transform: "
                        f"scale={tuple(scale.shape)}, rows={rows}, cols={cols}"
                    )
                cols16 = (cols + 15) // 16
                cols32 = (cols + 31) // 32
                if scale.dtype == torch.float32:
                    sf = scale
                elif scale.dtype == torch.float8_e8m0fnu:
                    sf = _upcast_e8m0_to_fp32(scale.view(torch.uint8))
                elif scale.dtype == torch.uint8:
                    sf = _upcast_e8m0_to_fp32(scale)
                else:
                    sf = scale.to(torch.float32)
                if sf.shape[-1] == cols32:
                    sf = sf.repeat_interleave(2, dim=-1)[..., :cols16]
                elif sf.shape[-1] != cols16:
                    raise RuntimeError(
                        "Unexpected MXFP4 scale column count for b12x transform: "
                        f"scale={tuple(scale.shape)}, expected {cols32} or {cols16}"
                    )
                return swizzle_block_scale(
                    sf[:, :rows, :cols16].contiguous().to(torch.float8_e4m3fn)
                ).contiguous()

            def _sm120_b12x_w13(weight: torch.Tensor) -> torch.Tensor:
                raw = weight.data.view(torch.uint8)
                half = raw.shape[1] // 2
                return torch.cat((raw[:, half:, :], raw[:, :half, :]), dim=1).contiguous()

            def _sm120_b12x_w13_scale(scale: torch.Tensor) -> torch.Tensor:
                half = scale.shape[1] // 2
                return torch.cat((scale[:, half:, :], scale[:, :half, :]), dim=1).contiguous()

            w13_weight_b12x = _sm120_b12x_w13(w13_weight)
            w2_weight_raw = w2_weight.data.view(torch.uint8)
            w2_weight_b12x = (
                w2_weight_raw if w2_weight_raw.is_contiguous()
                else w2_weight_raw.contiguous()
            )
            w13_scale_b12x = _sm120_b12x_w13_scale(w13_weight_scale.data)
            del w13_weight, w2_weight, w13_weight_scale
            torch.cuda.empty_cache()
            w13_scale_out = _sm120_b12x_scale(
                w13_scale_b12x,
                w13_weight_b12x.shape[1],
                w13_weight_b12x.shape[2] * 2,
            )
            del w13_scale_b12x
            w2_scale_out = _sm120_b12x_scale(
                w2_weight_scale.data,
                w2_weight_b12x.shape[1],
                w2_weight_b12x.shape[2] * 2,
            )
            del w2_weight_scale
            torch.cuda.empty_cache()
            return (
                w13_weight_b12x,
                w2_weight_b12x,
                w13_scale_out,
                w2_scale_out,
                w13_bias,
                w2_bias,
            )

"""
    replaced = source.count(needle)
    if replaced == 0:
        raise RuntimeError(f"Could not patch b12x MXFP4 load transform in {path}")
    source = source.replace(needle, insert + needle)
    path.write_text(source)


def patch_sm120_b12x_deep_gemm_moe() -> None:
    """Route the SM120 C128 DeepGEMM MXFP4 expert path through b12x on opt-in."""
    if __import__("os").environ.get("DG_SM120_ENABLE_B12X_MOE", "0").lower() not in (
        "1",
        "true",
        "yes",
        "on",
    ):
        return
    path = Path(
        "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/"
        "layers/fused_moe/experts/deep_gemm_moe.py"
    )
    source = path.read_text()

    if "import os\n" not in source:
        source = source.replace("import torch\n", "import os\n\nimport torch\n", 1)

    if "_SM120_B12X_WORKSPACES" not in source:
        helper = '''_SM120_B12X_WORKSPACES = {}


def _sm120_env_flag(name: str) -> bool:
    return os.environ.get(name, "0").lower() in ("1", "true", "yes", "on")


def _sm120_b12x_enabled() -> bool:
    return _sm120_env_flag("DG_SM120_ENABLE_B12X_MOE")


def _sm120_b12x_workspace(device: torch.device):
    from b12x.integration.tp_moe import allocate_tp_moe_workspace_pool

    index = device.index
    if index is None:
        index = torch.cuda.current_device()
    key = (device.type, index)
    workspace = _SM120_B12X_WORKSPACES.get(key)
    if workspace is None:
        workspace = allocate_tp_moe_workspace_pool()
        _SM120_B12X_WORKSPACES[key] = workspace
    return workspace


def _sm120_b12x_expert_scales(num_experts: int, device: torch.device):
    cache = getattr(_sm120_b12x_expert_scales, "_cache", {})
    key = (
        device.type,
        device.index if device.index is not None else torch.cuda.current_device(),
        num_experts,
        float(os.environ.get("DG_SM120_B12X_A1_GS", "1.0")),
        float(os.environ.get("DG_SM120_B12X_A2_GS", "1.0")),
    )
    cached = cache.get(key)
    if cached is None:
        a1 = torch.full((num_experts,), key[3], device=device, dtype=torch.float32)
        a2 = torch.full((num_experts,), key[4], device=device, dtype=torch.float32)
        alpha1 = torch.ones((num_experts,), device=device, dtype=torch.float32)
        alpha2 = torch.ones((num_experts,), device=device, dtype=torch.float32)
        cached = (a1, a2, alpha1, alpha2)
        cache[key] = cached
        setattr(_sm120_b12x_expert_scales, "_cache", cache)
    return cached


def _sm120_b12x_route(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    expert_map: torch.Tensor | None,
    apply_router_weight_on_input: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    weights = topk_weights.to(torch.float32)
    if apply_router_weight_on_input:
        weights = torch.ones_like(weights)
    if expert_map is None:
        return topk_ids.to(torch.int32).contiguous(), weights.contiguous()

    local_ids = expert_map.to(device=topk_ids.device)[topk_ids.to(torch.long)]
    valid = local_ids >= 0
    routed_ids = torch.where(valid, local_ids, torch.zeros_like(local_ids))
    routed_weights = torch.where(valid, weights, torch.zeros_like(weights))
    return routed_ids.to(torch.int32).contiguous(), routed_weights.contiguous()


def _sm120_b12x_can_apply(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scale: torch.Tensor | None,
    w2_scale: torch.Tensor | None,
    activation: MoEActivation,
) -> bool:
    return (
        _sm120_b12x_enabled()
        and hidden_states.is_cuda
        and torch.cuda.get_device_capability(hidden_states.device)[0] >= 12
        and activation == MoEActivation.SILU
        and hidden_states.dtype == torch.bfloat16
        and w1.dim() == 3
        and w2.dim() == 3
        and w1.shape[2] * 2 == hidden_states.shape[1]
        and w2.shape[1] == hidden_states.shape[1]
        and w1_scale is not None
        and w2_scale is not None
    )


def _sm120_b12x_apply(
    output: torch.Tensor,
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    expert_map: torch.Tensor | None,
    apply_router_weight_on_input: bool,
) -> None:
    from b12x.integration.tp_moe import b12x_moe_fp4

    if w1_scale.dtype != torch.float8_e4m3fn or w2_scale.dtype != torch.float8_e4m3fn:
        raise RuntimeError(
            "DG_SM120_ENABLE_B12X_MOE requires b12x-transformed E4M3 scale "
            "tensors. Restart vLLM with DG_SM120_ENABLE_B12X_MOE=1 before model load."
        )
    if w1_scale.dim() != 3 or w2_scale.dim() != 3:
        raise RuntimeError(
            "DG_SM120_ENABLE_B12X_MOE requires 3D b12x scale tensors; "
            f"got {tuple(w1_scale.shape)} and {tuple(w2_scale.shape)}"
        )

    num_experts = w1.shape[0]
    a1_gscale, a2_gscale, w1_alphas, w2_alphas = _sm120_b12x_expert_scales(
        num_experts, hidden_states.device
    )
    routed_ids, routed_weights = _sm120_b12x_route(
        topk_ids, topk_weights, expert_map, apply_router_weight_on_input
    )
    b12x_moe_fp4(
        hidden_states.contiguous(),
        a1_gscale,
        w1.view(torch.uint8),
        w1_scale,
        w1_alphas,
        a2_gscale,
        w2.view(torch.uint8),
        w2_scale,
        w2_alphas,
        routed_weights,
        routed_ids,
        workspace=_sm120_b12x_workspace(hidden_states.device),
        output=output,
        input_scales_static=True,
    )


'''
        marker = "logger = init_logger(__name__)\n\n\n"
        if marker not in source:
            raise RuntimeError(f"Could not find logger marker for b12x patch in {path}")
        source = source.replace(marker, marker + helper, 1)

    class_start = source.find("class DeepGemmFP4Experts")
    if class_start < 0:
        class_start = 0

    if (
        "def expects_unquantized_inputs(self) -> bool:\n"
        "        return _sm120_b12x_enabled()" not in source[class_start:]
    ):
        anchor = """    def supports_expert_map(self) -> bool:
        return True

"""
        insert = """    @property
    def expects_unquantized_inputs(self) -> bool:
        return _sm120_b12x_enabled()

"""
        anchor_pos = source.find(anchor, class_start)
        if anchor_pos < 0:
            raise RuntimeError(f"Could not add b12x expects_unquantized_inputs in {path}")
        anchor_end = anchor_pos + len(anchor)
        source = source[:anchor_end] + insert + source[anchor_end:]

    apply_region = source[class_start:]
    if "_sm120_b12x_can_apply(" not in apply_region.split("def apply(", 1)[-1]:
        old = """        assert a1q_scale is not None
        assert a2_scale is None
        assert self.block_shape is not None
        assert self.w1_scale is not None
        assert self.w2_scale is not None
"""
        new = """        assert a2_scale is None
        assert self.block_shape is not None
        assert self.w1_scale is not None
        assert self.w2_scale is not None

        if _sm120_b12x_can_apply(
            hidden_states, w1, w2, self.w1_scale, self.w2_scale, activation
        ):
            _sm120_b12x_apply(
                output,
                hidden_states,
                w1,
                w2,
                self.w1_scale,
                self.w2_scale,
                topk_weights,
                topk_ids,
                expert_map,
                apply_router_weight_on_input,
            )
            return

        assert a1q_scale is not None
"""
        old_pos = source.find(old, class_start)
        if old_pos < 0:
            raise RuntimeError(f"Could not patch b12x DeepGEMM apply in {path}")
        source = source[:old_pos] + new + source[old_pos + len(old):]

    path.write_text(source)


def patch_deepseek_v4_attention() -> None:
    path = Path(
        "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/"
        "layers/deepseek_v4_attention.py"
    )
    source = path.read_text()
    marker = "def deepseek_v4_fp8_einsum(\n"
    helper = '''def _deepseek_v4_expand_fp8_scale(scale: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    num_blocks = (target.shape[-1] + 127) // 128
    if scale.dtype == torch.int32 and scale.shape[-1] == (num_blocks + 3) // 4:
        shifts = torch.arange(4, device=scale.device, dtype=torch.int64) * 8
        exponents = ((scale.to(torch.int64).unsqueeze(-1) >> shifts) & 0xFF).to(torch.float32)
        scale = torch.exp2(exponents - 127.0)
        scale = torch.where(exponents == 0, torch.zeros_like(scale), scale)
        scale = scale.flatten(-2)[..., :num_blocks]
    else:
        scale = scale.to(torch.float32)

    if scale.shape[-1] != num_blocks:
        raise RuntimeError(f"Unexpected FP8 scale shape {tuple(scale.shape)} for target {tuple(target.shape)}")
    if scale.shape[:-1] != target.shape[:-1]:
        if (
            scale.dim() == target.dim()
            and target.dim() == 3
            and scale.shape[0] == target.shape[0]
            and scale.shape[-1] == num_blocks
            and target.shape[-2] % scale.shape[-2] == 0
        ):
            scale = scale.repeat_interleave(target.shape[-2] // scale.shape[-2], dim=-2)
        elif (
            scale.dim() == target.dim()
            and target.dim() == 2
            and len(scale.shape) == 2
            and target.shape[0] % scale.shape[0] == 0
        ):
            scale = scale.repeat_interleave(target.shape[0] // scale.shape[0], dim=0)
        else:
            scale = torch.broadcast_to(scale, (*target.shape[:-1], num_blocks))
    return scale.repeat_interleave(128, dim=-1)[..., : target.shape[-1]]


'''
    if marker not in source:
        raise RuntimeError(f"Could not find fp8_einsum marker in {path}")
    if "_deepseek_v4_expand_fp8_scale" not in source:
        source = source.replace(marker, helper + marker, 1)

    fast_path = '''    if (
        equation == "bhr,hdr->bhd"
        and a.is_cuda
        and torch.cuda.get_device_capability(a.device)[0] >= 12
    ):
        import deep_gemm
        deep_gemm._C.sm120_fp8_bhr_hdr_bhd(a, a_scale, b, b_scale, out)
        return
'''
    new = fast_path + '''    fp8_einsum(equation, (a, a_scale), (b, b_scale), out, recipe=tuple(recipe))
'''
    old_torch_fallback = '''    if (
        equation == "bhr,hdr->bhd"
        and a.is_cuda
        and torch.cuda.get_device_capability(a.device)[0] >= 12
    ):
        a_deq = a.to(torch.float32) * _deepseek_v4_expand_fp8_scale(a_scale, a)
        b_deq = b.to(torch.float32) * _deepseek_v4_expand_fp8_scale(b_scale, b)
        if b_deq.dim() == 2 and out.dim() == 3 and b_deq.shape[0] == out.shape[1] * out.shape[2]:
            b_deq = b_deq.view(out.shape[1], out.shape[2], b_deq.shape[-1])
        out.copy_(torch.einsum(equation, a_deq, b_deq).to(out.dtype))
        return
    fp8_einsum(equation, (a, a_scale), (b, b_scale), out, recipe=tuple(recipe))
'''
    old_torch_block_only = '''    if (
        equation == "bhr,hdr->bhd"
        and a.is_cuda
        and torch.cuda.get_device_capability(a.device)[0] >= 12
    ):
        a_deq = a.to(torch.float32) * _deepseek_v4_expand_fp8_scale(a_scale, a)
        b_deq = b.to(torch.float32) * _deepseek_v4_expand_fp8_scale(b_scale, b)
        if b_deq.dim() == 2 and out.dim() == 3 and b_deq.shape[0] == out.shape[1] * out.shape[2]:
            b_deq = b_deq.view(out.shape[1], out.shape[2], b_deq.shape[-1])
        out.copy_(torch.einsum(equation, a_deq, b_deq).to(out.dtype))
        return
'''
    old_direct = "    fp8_einsum(equation, (a, a_scale), (b, b_scale), out, recipe=tuple(recipe))\n"
    if "deep_gemm._C.sm120_fp8_bhr_hdr_bhd" in source:
        while fast_path + fast_path in source:
            source = source.replace(fast_path + fast_path, fast_path)
    elif old_torch_fallback in source:
        source = source.replace(old_torch_fallback, new, 1)
    elif old_torch_block_only in source and "sm120_fp8_bhr_hdr_bhd" in source:
        source = source.replace(old_torch_block_only, "", 1)
    elif old_direct in source:
        source = source.replace(old_direct, new, 1)
    elif "sm120_fp8_bhr_hdr_bhd" not in source:
        raise RuntimeError(f"Could not patch fp8_einsum body in {path}")
    path.write_text(source)


def patch_deepseek_v4_prefill_dynamic_compressed_workspace() -> None:
    """Shrink DeepSeek V4 sparse-prefill C128 workspace/index width on SM120.

    vLLM's C128 prefill path sizes the gathered workspace with
    N=max_model_len/compress_ratio and passes the full padded C128 top-k width
    into combine_topk_swa_indices. For a 128k model length this means each
    modest 4k prompt still carries a 1024-wide compressed region even though the
    prompt only has about 32 compressed C128 entries. The downstream SM120
    prefill bridge can trim padded work, but by then vLLM has already allocated
    and combined the oversized sparse-index rows.

    For prefill only, compute a per-chunk compressed capacity from the chunk's
    actual sequence lengths, round the C128 top-k width to FlashMLA's 128-entry
    alignment, and allocate the gathered KV workspace with the smaller C128
    offset. This is a broader sparse-prefill boundary reduction rather than a
    single-token decode helper tweak.
    """
    path = Path(
        "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/"
        "layers/deepseek_v4_attention.py"
    )
    source = path.read_text()
    if "DG_SM120_PREFILL_DYNAMIC_COMPRESSED_N" in source:
        old = """        kv = None
        if not sm120_dynamic_compressed_n:
            kv = workspace_manager.get_simultaneous(
                ((PREFILL_CHUNK_SIZE, M, q.shape[-1]), torch.bfloat16),
            )[0]
        for chunk_idx in range(num_chunks):
"""
        new = """        kv = None
        if not sm120_dynamic_compressed_n:
            kv = workspace_manager.get_simultaneous(
                ((PREFILL_CHUNK_SIZE, M, q.shape[-1]), torch.bfloat16),
            )[0]
        sm120_dynamic_chunk_ns = None
        if sm120_dynamic_compressed_n:
            sm120_dynamic_chunk_ns = getattr(
                swa_metadata, "_sm120_dynamic_c128_chunk_ns", None
            )
            if sm120_dynamic_chunk_ns is None or len(sm120_dynamic_chunk_ns) != num_chunks:
                chunk_seq_lens_cpu = getattr(
                    swa_metadata, "prefill_seq_lens_cpu", None
                )
                chunk_ns = []
                for _chunk_idx in range(num_chunks):
                    _chunk_start = _chunk_idx * PREFILL_CHUNK_SIZE
                    _chunk_end = min(_chunk_start + PREFILL_CHUNK_SIZE, num_prefills)
                    if chunk_seq_lens_cpu is not None:
                        _max_seq_len = int(
                            chunk_seq_lens_cpu[_chunk_start:_chunk_end].max().item()
                        )
                    else:
                        _max_seq_len = int(
                            seq_lens[_chunk_start:_chunk_end].max().item()
                        )
                    _compressed = int(
                        (_max_seq_len + self.compress_ratio - 1)
                        // self.compress_ratio
                    )
                    chunk_ns.append(max(1, min(N, _compressed)))
                sm120_dynamic_chunk_ns = tuple(chunk_ns)
                setattr(
                    swa_metadata,
                    "_sm120_dynamic_c128_chunk_ns",
                    sm120_dynamic_chunk_ns,
                )
        for chunk_idx in range(num_chunks):
"""
        if old in source:
            source = source.replace(old, new, 1)
        old = """                chunk_seq_lens_cpu = getattr(
                    swa_metadata, "prefill_seq_lens_cpu", None
                )
                if chunk_seq_lens_cpu is not None:
                    chunk_max_seq_len = int(
                        chunk_seq_lens_cpu[chunk_start:chunk_end].max().item()
                    )
                else:
                    chunk_max_seq_len = int(
                        seq_lens[chunk_start:chunk_end].max().item()
                    )
                chunk_compressed = int(
                    (chunk_max_seq_len + self.compress_ratio - 1)
                    // self.compress_ratio
                )
                chunk_N = max(1, min(N, chunk_compressed))
"""
        new = """                assert sm120_dynamic_chunk_ns is not None
                chunk_N = sm120_dynamic_chunk_ns[chunk_idx]
"""
        if old in source:
            source = source.replace(old, new, 1)
        path.write_text(source)
        return

    old = """        M = N + self.window_size + self.max_num_batched_tokens
        num_chunks = (num_prefills + PREFILL_CHUNK_SIZE - 1) // PREFILL_CHUNK_SIZE

        workspace_manager = current_workspace_manager()
        kv = workspace_manager.get_simultaneous(
            ((PREFILL_CHUNK_SIZE, M, q.shape[-1]), torch.bfloat16),
        )[0]
        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * PREFILL_CHUNK_SIZE
            chunk_end = min(chunk_start + PREFILL_CHUNK_SIZE, num_prefills)
            chunk_size = chunk_end - chunk_start
            if not swa_only:
                # Gather compressed KV
                assert attn_metadata is not None
                block_table = attn_metadata.block_table[num_decodes:]
                dequantize_and_gather_k_cache(
                    kv[:chunk_size],
                    compressed_k_cache,
                    seq_lens=seq_lens[chunk_start:chunk_end] // self.compress_ratio,
                    gather_lens=None,
                    block_table=block_table[chunk_start:chunk_end],
                    block_size=attn_metadata.block_size // self.compress_ratio,
                    offset=0,
                )

            # Gather SWA KV
            swa_block_table = swa_metadata.block_table[num_decodes:]
            dequantize_and_gather_k_cache(
                kv[:chunk_size],
                swa_k_cache,
                seq_lens=seq_lens[chunk_start:chunk_end],
                gather_lens=gather_lens[chunk_start:chunk_end],
                block_table=swa_block_table[chunk_start:chunk_end],
                block_size=swa_metadata.block_size,
                offset=N,
            )

            # Combine the topk indices and SWA indices for gathered KV cache
            query_start = (
                query_start_loc_cpu[num_decodes + chunk_start] - prefill_token_base
            )
            query_end = (
                query_start_loc_cpu[num_decodes + chunk_end] - prefill_token_base
            )

            combined_indices, combined_lens = combine_topk_swa_indices(
                topk_indices[query_start:query_end],
                query_start_loc[
                    num_decodes + chunk_start : num_decodes + chunk_end + 1
                ],
                seq_lens[chunk_start:chunk_end],
                gather_lens[chunk_start:chunk_end],
                self.window_size,
                self.compress_ratio,
                top_k,
                M,
                N,
            )
"""
    new = """        M = N + self.window_size + self.max_num_batched_tokens
        num_chunks = (num_prefills + PREFILL_CHUNK_SIZE - 1) // PREFILL_CHUNK_SIZE

        workspace_manager = current_workspace_manager()
        sm120_dynamic_compressed_n = (
            (not swa_only)
            and self.compress_ratio == 128
            and q.is_cuda
            and torch.cuda.get_device_capability(q.device)[0] >= 12
            and __import__("os").environ.get(
                "DG_SM120_PREFILL_DYNAMIC_COMPRESSED_N", "1"
            )
            != "0"
        )
        kv = None
        if not sm120_dynamic_compressed_n:
            kv = workspace_manager.get_simultaneous(
                ((PREFILL_CHUNK_SIZE, M, q.shape[-1]), torch.bfloat16),
            )[0]
        sm120_dynamic_chunk_ns = None
        if sm120_dynamic_compressed_n:
            sm120_dynamic_chunk_ns = getattr(
                swa_metadata, "_sm120_dynamic_c128_chunk_ns", None
            )
            if sm120_dynamic_chunk_ns is None or len(sm120_dynamic_chunk_ns) != num_chunks:
                chunk_seq_lens_cpu = getattr(
                    swa_metadata, "prefill_seq_lens_cpu", None
                )
                chunk_ns = []
                for _chunk_idx in range(num_chunks):
                    _chunk_start = _chunk_idx * PREFILL_CHUNK_SIZE
                    _chunk_end = min(_chunk_start + PREFILL_CHUNK_SIZE, num_prefills)
                    if chunk_seq_lens_cpu is not None:
                        _max_seq_len = int(
                            chunk_seq_lens_cpu[_chunk_start:_chunk_end].max().item()
                        )
                    else:
                        _max_seq_len = int(
                            seq_lens[_chunk_start:_chunk_end].max().item()
                        )
                    _compressed = int(
                        (_max_seq_len + self.compress_ratio - 1)
                        // self.compress_ratio
                    )
                    chunk_ns.append(max(1, min(N, _compressed)))
                sm120_dynamic_chunk_ns = tuple(chunk_ns)
                setattr(
                    swa_metadata,
                    "_sm120_dynamic_c128_chunk_ns",
                    sm120_dynamic_chunk_ns,
                )
        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * PREFILL_CHUNK_SIZE
            chunk_end = min(chunk_start + PREFILL_CHUNK_SIZE, num_prefills)
            chunk_size = chunk_end - chunk_start

            # Combine/gather works on a chunk-local query window.
            query_start = (
                query_start_loc_cpu[num_decodes + chunk_start] - prefill_token_base
            )
            query_end = (
                query_start_loc_cpu[num_decodes + chunk_end] - prefill_token_base
            )

            chunk_N = N
            chunk_M = M
            chunk_top_k = top_k
            chunk_topk_indices = topk_indices[query_start:query_end]
            if sm120_dynamic_compressed_n:
                # Per-chunk actual C128 compressed capacity.  Round the sparse
                # top-k width to the same 128-entry alignment used by FlashMLA
                # sparse prefill, but keep the SWA offset tight so the gathered
                # workspace does not reserve the whole 128k-context C128 pool.
                assert sm120_dynamic_chunk_ns is not None
                chunk_N = sm120_dynamic_chunk_ns[chunk_idx]
                chunk_top_k = max(1, min(top_k, ((chunk_N + 127) // 128) * 128))
                chunk_topk_indices = topk_indices[
                    query_start:query_end, :chunk_top_k
                ]
                chunk_M = chunk_N + self.window_size + self.max_num_batched_tokens
                kv = workspace_manager.get_simultaneous(
                    ((PREFILL_CHUNK_SIZE, chunk_M, q.shape[-1]), torch.bfloat16),
                )[0]
            assert kv is not None

            if not swa_only:
                # Gather compressed KV
                assert attn_metadata is not None
                block_table = attn_metadata.block_table[num_decodes:]
                dequantize_and_gather_k_cache(
                    kv[:chunk_size],
                    compressed_k_cache,
                    seq_lens=seq_lens[chunk_start:chunk_end] // self.compress_ratio,
                    gather_lens=None,
                    block_table=block_table[chunk_start:chunk_end],
                    block_size=attn_metadata.block_size // self.compress_ratio,
                    offset=0,
                )

            # Gather SWA KV
            swa_block_table = swa_metadata.block_table[num_decodes:]
            dequantize_and_gather_k_cache(
                kv[:chunk_size],
                swa_k_cache,
                seq_lens=seq_lens[chunk_start:chunk_end],
                gather_lens=gather_lens[chunk_start:chunk_end],
                block_table=swa_block_table[chunk_start:chunk_end],
                block_size=swa_metadata.block_size,
                offset=chunk_N,
            )

            combined_indices, combined_lens = combine_topk_swa_indices(
                chunk_topk_indices,
                query_start_loc[
                    num_decodes + chunk_start : num_decodes + chunk_end + 1
                ],
                seq_lens[chunk_start:chunk_end],
                gather_lens[chunk_start:chunk_end],
                self.window_size,
                self.compress_ratio,
                chunk_top_k,
                chunk_M,
                chunk_N,
            )
"""
    if old not in source:
        raise RuntimeError(
            f"Could not patch DeepSeek V4 dynamic prefill C128 workspace in {path}"
        )
    source = source.replace(old, new, 1)
    path.write_text(source)


def patch_deepseek_v4_combine_topk_swa_empty_indices() -> None:
    """Avoid clearing unused sparse-prefill combined-index tails on SM120.

    combine_topk_swa_indices allocates a padded [tokens, topk + window] matrix
    with torch.full(..., -1), then the Triton kernel overwrites only the valid
    prefix and separately returns combined_lens. The SM120 prefill bridge masks
    by combined_lens/topk_length before attention, so clearing the unused tail is
    wasted per-layer work. Keep an env-gated fallback to the upstream fill for
    quick correctness isolation.
    """
    path = Path(
        "/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/ops/"
        "deepseek_v4_ops/cache_utils.py"
    )
    source = path.read_text()
    if "DG_SM120_PREFILL_EMPTY_COMBINED_INDICES" in source:
        return

    old = """    combined_indices = torch.full(
        (num_tokens, combined_topk),
        fill_value=-1,
        dtype=torch.int32,
        device=topk_indices.device,
    )
"""
    new = """    sm120_empty_combined_indices = (
        topk_indices.is_cuda
        and torch.cuda.get_device_capability(topk_indices.device)[0] >= 12
        and __import__("os").environ.get(
            "DG_SM120_PREFILL_EMPTY_COMBINED_INDICES", "1"
        )
        != "0"
    )
    if sm120_empty_combined_indices:
        combined_indices = torch.empty(
            (num_tokens, combined_topk),
            dtype=torch.int32,
            device=topk_indices.device,
        )
    else:
        combined_indices = torch.full(
            (num_tokens, combined_topk),
            fill_value=-1,
            dtype=torch.int32,
            device=topk_indices.device,
        )
"""
    if old not in source:
        raise RuntimeError(f"Could not patch combine_topk_swa_indices in {path}")
    source = source.replace(old, new, 1)
    path.write_text(source)


def patch_deepseek_v4_direct_fp8_prefill_map() -> None:
    """Add an opt-in direct FP8 sparse-prefill path for SM120.

    The default path gathers compressed + SWA FP8 KV cache rows into a BF16
    workspace before sparse attention.  This patch keeps the proven BF16 path
    as default, but adds a larger-boundary experiment that builds a compact
    int32 workspace-row -> physical-cache-row map and runs sparse prefill
    directly from the FP8 caches.  It is intentionally env-gated because the
    first scalar direct kernel may lose to tensor-core BMM on some shapes; the
    value is validating/removing the BF16 materialization boundary.
    """
    path = Path(
        "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/"
        "layers/deepseek_v4_attention.py"
    )
    source = path.read_text()
    if "DG_SM120_PREFILL_DIRECT_FP8_MAP" in source:
        return

    old = """            output_chunk, _, _ = flash_mla_sparse_fwd(
                q=q[query_start:query_end],
                kv=kv.view(-1, 1, q.shape[-1]),
                indices=combined_indices.unsqueeze(1),
                sm_scale=self.scale,
                attn_sink=self.attn_sink,
                topk_length=combined_lens,
                out=output[query_start:query_end],
            )
"""
    new = """            output_chunk = None
            sm120_direct_fp8_prefill = (
                (not swa_only)
                and attn_metadata is not None
                and q.is_cuda
                and torch.cuda.get_device_capability(q.device)[0] >= 12
                and __import__("os").environ.get(
                    "DG_SM120_PREFILL_DIRECT_FP8_MAP", "0"
                )
                != "0"
            )
            if sm120_direct_fp8_prefill:
                try:
                    import deep_gemm

                    dg_c = getattr(deep_gemm, "_C", None)
                    build_strided_map = getattr(
                        dg_c, "sm120_build_prefill_strided_workspace_map", None
                    )
                    direct_prefill = getattr(
                        dg_c, "sm120_sparse_mla_prefill_from_two_fp8_workspace_map", None
                    )
                    if build_strided_map is not None and direct_prefill is not None:
                        rows = int(chunk_size * chunk_M)
                        map_cache = getattr(self, "_dg_sm120_prefill_direct_map", None)
                        if (
                            map_cache is None
                            or map_cache.device != q.device
                            or map_cache.numel() < rows
                        ):
                            map_cache = torch.empty(
                                (rows,), device=q.device, dtype=torch.int32
                            )
                            setattr(self, "_dg_sm120_prefill_direct_map", map_cache)
                        workspace_map = map_cache[:rows]
                        workspace_map.fill_(-1)
                        compressed_block_table = attn_metadata.block_table[num_decodes:][
                            chunk_start:chunk_end
                        ]
                        build_strided_map(
                            workspace_map,
                            compressed_block_table,
                            seq_lens[chunk_start:chunk_end] // self.compress_ratio,
                            None,
                            attn_metadata.block_size // self.compress_ratio,
                            chunk_M,
                            0,
                            False,
                        )
                        build_strided_map(
                            workspace_map,
                            swa_block_table[chunk_start:chunk_end],
                            seq_lens[chunk_start:chunk_end],
                            gather_lens[chunk_start:chunk_end],
                            swa_metadata.block_size,
                            chunk_M,
                            chunk_N,
                            True,
                        )
                        output_chunk, _, _ = direct_prefill(
                            q[query_start:query_end],
                            compressed_k_cache.view(torch.uint8).unsqueeze(-2),
                            swa_k_cache.view(torch.uint8).unsqueeze(-2),
                            workspace_map,
                            combined_indices.unsqueeze(1),
                            combined_lens,
                            self.attn_sink,
                            attn_metadata.block_size // self.compress_ratio,
                            swa_metadata.block_size,
                            q.shape[-1],
                            self.scale,
                            output[query_start:query_end],
                        )
                except Exception:
                    if __import__("os").environ.get(
                        "DG_SM120_PREFILL_DIRECT_FP8_MAP_STRICT", "0"
                    ) != "0":
                        raise
                    output_chunk = None

            if output_chunk is None:
                output_chunk, _, _ = flash_mla_sparse_fwd(
                    q=q[query_start:query_end],
                    kv=kv.view(-1, 1, q.shape[-1]),
                    indices=combined_indices.unsqueeze(1),
                    sm_scale=self.scale,
                    attn_sink=self.attn_sink,
                    topk_length=combined_lens,
                    out=output[query_start:query_end],
                )
"""
    if old not in source:
        raise RuntimeError(f"Could not patch direct SM120 FP8 prefill map in {path}")
    source = source.replace(old, new, 1)
    path.write_text(source)


def patch_flashmla_sparse_prefill() -> None:
    paths = (
        Path(
            "/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/"
            "ops/flashmla.py"
        ),
        Path(
            "/usr/local/lib/python3.12/dist-packages/vllm/third_party/"
            "flashmla/flash_mla_interface.py"
        ),
    )
    marker = "def flash_mla_sparse_fwd(\n"
    helper = '''_DG_SM120_PREFILL_COMPILED_BMM = None

def _dg_sm120_prefill_torch_bmm(q_chunk, workspace, valid, valid_any, sink, sm_scale):
    scores = torch.bmm(q_chunk, workspace.transpose(1, 2)).to(torch.float32)
    scores.mul_(float(sm_scale))
    scores.masked_fill_(~valid.unsqueeze(1), float("-inf"))
    scores = torch.where(
        valid_any.reshape(-1, 1, 1), scores, torch.zeros_like(scores)
    )
    chunk_lse = torch.logsumexp(scores, dim=-1)
    probs = torch.softmax(scores, dim=-1).to(q_chunk.dtype)
    probs.masked_fill_(~valid_any.reshape(-1, 1, 1), 0)
    chunk_out = torch.bmm(probs, workspace)
    gate = torch.sigmoid(chunk_lse - sink.to(chunk_lse.dtype).reshape(1, -1))
    chunk_out = chunk_out * gate.to(chunk_out.dtype).unsqueeze(-1)
    return chunk_out, chunk_lse

def _dg_sm120_get_prefill_compiled_bmm():
    global _DG_SM120_PREFILL_COMPILED_BMM
    if _DG_SM120_PREFILL_COMPILED_BMM is None:
        mode = __import__("os").environ.get(
            "DG_SM120_PREFILL_TORCH_COMPILE_MODE", "reduce-overhead"
        )
        _DG_SM120_PREFILL_COMPILED_BMM = torch.compile(
            _dg_sm120_prefill_torch_bmm, mode=mode, fullgraph=True
        )
    return _DG_SM120_PREFILL_COMPILED_BMM

def _sm120_flash_mla_sparse_prefill_fwd(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_v: int = 512,
    attn_sink: Optional[torch.Tensor] = None,
    topk_length: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Correctness fallback for RTX PRO 6000 Blackwell sparse MLA prefill.

    The bundled FlashMLA sparse prefill kernel only admits SM90a/SM100f. This
    keeps DeepSeek V4 functional on SM120 while native support is implemented.
    """
    if kv.shape[1] != 1 or indices.shape[1] != 1:
        raise RuntimeError(
            "SM120 sparse prefill fallback currently expects single-KV-head MLA"
        )
    if out is None:
        out = torch.empty(
            (q.shape[0], q.shape[1], d_v), device=q.device, dtype=q.dtype
        )

    try:
        import deep_gemm
        import os

        workspace_decode = getattr(
            getattr(deep_gemm, "_C", None),
            "sm120_sparse_mla_decode_from_bf16_workspace_split",
            None,
        )
        workspace_gather = getattr(
            getattr(deep_gemm, "_C", None),
            "sm120_gather_bf16_workspace",
            None,
        )
        indexed_prefill_split = getattr(
            getattr(deep_gemm, "_C", None),
            "sm120_sparse_mla_prefill_from_bf16_workspace_split",
            None,
        )
        chunk_size = int(os.environ.get("DG_SM120_PREFILL_WORKSPACE_CHUNK", "16"))
        if (
            os.environ.get("DG_SM120_PREFILL_INDEXED_SPLIT", "0") == "1"
            and indexed_prefill_split is not None
            and chunk_size > 0
        ):
            lse = torch.empty((q.shape[0], q.shape[1]), device=q.device,
                              dtype=torch.float32)
            max_logits = torch.empty_like(lse)
            for start in range(0, q.shape[0], chunk_size):
                end = min(start + chunk_size, q.shape[0])
                chunk_topk = None
                chunk_width = indices.shape[-1]
                if topk_length is not None:
                    chunk_topk = topk_length[start:end]
                    trim_min_width = int(
                        os.environ.get("DG_SM120_PREFILL_TRIM_TOPK_MIN_WIDTH", "2048")
                    )
                    if (
                        os.environ.get("DG_SM120_PREFILL_TRIM_TOPK", "1") == "1"
                        and indices.shape[-1] >= trim_min_width
                    ):
                        chunk_width = max(
                            1,
                            min(
                                indices.shape[-1],
                                int(chunk_topk.max().item()),
                            ),
                        )
                _chunk_out, _chunk_max, chunk_lse = indexed_prefill_split(
                    q[start:end],
                    kv,
                    indices[start:end, :, :chunk_width],
                    chunk_topk,
                    attn_sink,
                    d_v,
                    sm_scale,
                    out[start:end],
                )
                lse[start:end].copy_(chunk_lse)
            max_logits.fill_(float("nan"))
            return out, max_logits, lse

        if workspace_decode is not None and chunk_size > 0:
            kv_2d = kv[:, 0, :]
            lse = torch.empty((q.shape[0], q.shape[1]), device=q.device,
                              dtype=torch.float32)
            max_logits = torch.empty_like(lse)
            cudnn_attention_enabled = (
                os.environ.get("DG_SM120_PREFILL_CUDNN", "0") == "1"
                and os.environ.get("DG_SM120_PREFILL_CUDNN_UNMASKED", "0") == "1"
                and d_v == q.shape[-1]
                and hasattr(torch.ops.aten, "_scaled_dot_product_cudnn_attention")
            )
            torch_bmm_enabled = (
                os.environ.get("DG_SM120_PREFILL_TORCH_BMM", "1") == "1"
                and d_v == q.shape[-1]
            )
            trust_indices = (
                os.environ.get("DG_SM120_PREFILL_TRUST_INDICES", "1") == "1"
            )
            gather_enabled = (
                os.environ.get("DG_SM120_PREFILL_GATHER_WORKSPACE", "0") == "1"
                and workspace_gather is not None
                and indices.is_contiguous()
                and kv_2d.is_contiguous()
            )
            index_select_enabled = (
                os.environ.get("DG_SM120_PREFILL_INDEX_SELECT", "0") == "1"
                and indices.is_contiguous()
                and kv_2d.is_contiguous()
            )
            safe_tail_indices = (
                os.environ.get("DG_SM120_PREFILL_EMPTY_COMBINED_INDICES", "0") == "1"
            )
            workspace_cache = getattr(
                _sm120_flash_mla_sparse_prefill_fwd,
                "_dg_sm120_workspace_cache",
                {},
            )
            for start in range(0, q.shape[0], chunk_size):
                end = min(start + chunk_size, q.shape[0])
                chunk_topk = None
                chunk_width = indices.shape[-1]
                if topk_length is not None:
                    chunk_topk = topk_length[start:end]
                    trim_min_width = int(
                        os.environ.get("DG_SM120_PREFILL_TRIM_TOPK_MIN_WIDTH", "2048")
                    )
                    if (
                        os.environ.get("DG_SM120_PREFILL_TRIM_TOPK", "1") == "1"
                        and indices.shape[-1] >= trim_min_width
                    ):
                        chunk_width = max(
                            1,
                            min(
                                indices.shape[-1],
                                int(chunk_topk.max().item()),
                            ),
                        )
                chunk_indices = indices[start:end, 0, :chunk_width]
                if gather_enabled:
                    rows = end - start
                    key = (
                        q.device.index,
                        str(kv_2d.dtype),
                        chunk_size,
                        chunk_width,
                        kv_2d.shape[-1],
                    )
                    workspace = workspace_cache.get(key)
                    if workspace is None or workspace.shape != (
                        chunk_size,
                        chunk_width,
                        kv_2d.shape[-1],
                    ):
                        workspace = torch.empty(
                            (chunk_size, chunk_width, kv_2d.shape[-1]),
                            device=q.device,
                            dtype=kv_2d.dtype,
                        )
                        workspace_cache[key] = workspace
                        setattr(
                            _sm120_flash_mla_sparse_prefill_fwd,
                            "_dg_sm120_workspace_cache",
                            workspace_cache,
                        )
                    workspace = workspace[:rows]
                    workspace_gather(kv_2d, chunk_indices, workspace)
                elif index_select_enabled:
                    rows = end - start
                    key = (
                        q.device.index,
                        str(kv_2d.dtype),
                        chunk_size,
                        chunk_width,
                        kv_2d.shape[-1],
                    )
                    workspace = workspace_cache.get(key)
                    if workspace is None or workspace.shape != (
                        chunk_size,
                        chunk_width,
                        kv_2d.shape[-1],
                    ):
                        workspace = torch.empty(
                            (chunk_size, chunk_width, kv_2d.shape[-1]),
                            device=q.device,
                            dtype=kv_2d.dtype,
                        )
                        workspace_cache[key] = workspace
                        setattr(
                            _sm120_flash_mla_sparse_prefill_fwd,
                            "_dg_sm120_workspace_cache",
                            workspace_cache,
                        )
                    workspace = workspace[:rows]
                    select_indices = chunk_indices.reshape(-1).clamp(
                        0, kv_2d.shape[0] - 1
                    )
                    torch.index_select(
                        kv_2d,
                        0,
                        select_indices,
                        out=workspace.reshape(-1, kv_2d.shape[-1]),
                    )
                else:
                    idx = chunk_indices
                    if safe_tail_indices and chunk_topk is not None:
                        pos_cache = getattr(
                            _sm120_flash_mla_sparse_prefill_fwd,
                            "_dg_sm120_pos_cache",
                            {},
                        )
                        pos_key = (
                            q.device.index,
                            idx.shape[1],
                        )
                        pos = pos_cache.get(pos_key)
                        if pos is None:
                            pos = torch.arange(idx.shape[1], device=q.device)
                            pos_cache[pos_key] = pos
                            setattr(
                                _sm120_flash_mla_sparse_prefill_fwd,
                                "_dg_sm120_pos_cache",
                                pos_cache,
                            )
                        idx = torch.where(
                            pos.reshape(1, -1)
                            < chunk_topk.to(torch.long).reshape(-1, 1),
                            idx,
                            torch.zeros((), device=idx.device, dtype=idx.dtype),
                        )
                    elif (not trust_indices) or safe_tail_indices:
                        idx = idx.to(torch.long).clamp(0, kv_2d.shape[0] - 1)
                    workspace = kv_2d[idx]
                if torch_bmm_enabled:
                    active_heads = min(
                        q.shape[1],
                        int(os.environ.get("DG_SM120_ACTIVE_HEADS", q.shape[1])),
                    )
                    q_chunk = q[start:end, :active_heads, :]
                    idx_chunk = chunk_indices[:, : workspace.shape[1]]
                    valid = (idx_chunk >= 0) & (idx_chunk < kv_2d.shape[0])
                    if chunk_topk is not None:
                        pos_cache = getattr(
                            _sm120_flash_mla_sparse_prefill_fwd,
                            "_dg_sm120_pos_cache",
                            {},
                        )
                        pos_key = (
                            q.device.index,
                            workspace.shape[1],
                        )
                        pos = pos_cache.get(pos_key)
                        if pos is None:
                            pos = torch.arange(workspace.shape[1], device=q.device)
                            pos_cache[pos_key] = pos
                            setattr(
                                _sm120_flash_mla_sparse_prefill_fwd,
                                "_dg_sm120_pos_cache",
                                pos_cache,
                            )
                        valid = valid & (
                            pos.reshape(1, -1)
                            < chunk_topk.to(torch.long).reshape(-1, 1)
                        )
                    valid_any = valid.any(dim=-1)
                    if cudnn_attention_enabled:
                        bias_cache = getattr(
                            _sm120_flash_mla_sparse_prefill_fwd,
                            "_dg_sm120_cudnn_bias_cache",
                            {},
                        )
                        bias_key = (
                            q.device.index,
                            chunk_size,
                            workspace.shape[1],
                        )
                        bias = bias_cache.get(bias_key)
                        if bias is None or bias.shape != (
                            chunk_size,
                            1,
                            1,
                            workspace.shape[1],
                        ):
                            bias = torch.empty(
                                (chunk_size, 1, 1, workspace.shape[1]),
                                device=q.device,
                                dtype=torch.float32,
                            )
                            bias_cache[bias_key] = bias
                            setattr(
                                _sm120_flash_mla_sparse_prefill_fwd,
                                "_dg_sm120_cudnn_bias_cache",
                                bias_cache,
                            )
                        bias = bias[: end - start]
                        safe_valid = valid | ~valid_any.reshape(end - start, 1)
                        bias.zero_()
                        bias.masked_fill_(
                            ~safe_valid.reshape(end - start, 1, 1, workspace.shape[1]),
                            float("-inf"),
                        )
                        chunk_out4, chunk_lse4, *_ = (
                            torch.ops.aten._scaled_dot_product_cudnn_attention(
                                q_chunk.unsqueeze(2),
                                workspace.unsqueeze(1),
                                workspace.unsqueeze(1),
                                bias,
                                True,
                                0.0,
                                False,
                                False,
                                scale=float(sm_scale),
                            )
                        )
                        chunk_out = chunk_out4.squeeze(2)
                        chunk_lse = chunk_lse4.squeeze(-1).squeeze(-1)
                        chunk_lse = chunk_lse.masked_fill(
                            ~valid_any.unsqueeze(1), float("-inf")
                        )
                        chunk_out = chunk_out.masked_fill(
                            ~valid_any.reshape(end - start, 1, 1), 0
                        )
                        if attn_sink is not None:
                            gate = torch.sigmoid(
                                chunk_lse
                                - attn_sink[:active_heads]
                                .to(chunk_lse.dtype)
                                .reshape(1, -1)
                            )
                            chunk_out = (
                                chunk_out * gate.to(chunk_out.dtype).unsqueeze(-1)
                            )
                    else:
                        use_compiled_bmm = (
                            os.environ.get("DG_SM120_PREFILL_TORCH_COMPILE", "0") == "1"
                            and attn_sink is not None
                            and (end - start) >= int(
                                os.environ.get(
                                    "DG_SM120_PREFILL_TORCH_COMPILE_MIN_ROWS", "64"
                                )
                            )
                        )
                        if use_compiled_bmm:
                            chunk_out, chunk_lse = _dg_sm120_get_prefill_compiled_bmm()(
                                q_chunk,
                                workspace,
                                valid,
                                valid_any,
                                attn_sink[:active_heads],
                                float(sm_scale),
                            )
                        else:
                            scores = torch.bmm(
                                q_chunk, workspace.transpose(1, 2)
                            ).to(torch.float32)
                            scores.mul_(float(sm_scale))
                            scores.masked_fill_(~valid.unsqueeze(1), float("-inf"))
                            scores = torch.where(
                                valid_any.reshape(-1, 1, 1),
                                scores,
                                torch.zeros_like(scores),
                            )
                            chunk_lse = torch.logsumexp(scores, dim=-1)
                            probs = torch.softmax(scores, dim=-1).to(q.dtype)
                            probs.masked_fill_(~valid_any.reshape(-1, 1, 1), 0)
                            chunk_out = torch.bmm(probs, workspace)
                            if attn_sink is not None:
                                gate = torch.sigmoid(
                                    chunk_lse
                                    - attn_sink[:active_heads]
                                    .to(chunk_lse.dtype)
                                    .reshape(1, -1)
                                )
                                chunk_out = (
                                    chunk_out * gate.to(chunk_out.dtype).unsqueeze(-1)
                                )
                    out[start:end, :active_heads, :].copy_(chunk_out)
                    if active_heads < q.shape[1]:
                        out[start:end, active_heads:, :].zero_()
                    lse[start:end, :active_heads].copy_(
                        chunk_lse.masked_fill(
                            ~valid_any.unsqueeze(1), float("-inf")
                        )
                    )
                    if active_heads < q.shape[1]:
                        lse[start:end, active_heads:].fill_(float("-inf"))
                    continue
                chunk_out, chunk_lse = workspace_decode(
                    q[start:end].unsqueeze(1),
                    workspace,
                    chunk_topk,
                    None,
                    attn_sink,
                    workspace.shape[1],
                    0,
                    d_v,
                    sm_scale,
                    out[start:end].unsqueeze(1),
                )
                lse[start:end].copy_(chunk_lse.squeeze(-1))
            max_logits.fill_(float("nan"))
            return out, max_logits, lse

        native_prefill = getattr(
            getattr(deep_gemm, "_C", None),
            "sm120_sparse_mla_prefill_from_bf16_workspace",
            None,
        )
        if native_prefill is not None:
            return native_prefill(
                q, kv, indices, topk_length, attn_sink, d_v, sm_scale, out
            )
    except Exception:
        # Keep the correctness fallback usable while the extension is rebuilt.
        pass

    max_logits = torch.empty((q.shape[0], q.shape[1]), device=q.device,
                             dtype=torch.float32)
    lse = torch.empty_like(max_logits)
    q_f = q.to(torch.float32)
    kv_f = kv[:, 0, :].to(torch.float32)
    idx = indices[:, 0, :]
    sink = attn_sink.to(torch.float32) if attn_sink is not None else None

    for token_idx in range(q.shape[0]):
        limit = idx.shape[-1]
        if topk_length is not None:
            limit = int(topk_length[token_idx].item())
        token_indices = idx[token_idx, :limit].to(torch.long)
        valid = (token_indices >= 0) & (token_indices < kv.shape[0])
        token_indices = token_indices[valid]

        if token_indices.numel() == 0:
            out[token_idx].zero_()
            max_logits[token_idx].fill_(float("-inf"))
            lse[token_idx].fill_(float("-inf"))
            continue

        selected = kv_f.index_select(0, token_indices)
        scores = torch.matmul(q_f[token_idx], selected[:, : q.shape[-1]].T)
        scores = scores * sm_scale
        token_max = torch.max(scores, dim=-1).values
        token_lse = torch.logsumexp(scores, dim=-1)
        probs = torch.softmax(scores, dim=-1)
        token_out = torch.matmul(probs, selected[:, :d_v])
        if sink is not None:
            token_out = token_out * torch.sigmoid(token_lse - sink).unsqueeze(-1)
        out[token_idx].copy_(token_out.to(out.dtype))
        max_logits[token_idx].copy_(token_max)
        lse[token_idx].copy_(token_lse)

    return out, max_logits, lse


'''
    path = None
    source = None
    for candidate in paths:
        candidate_source = candidate.read_text()
        if marker in candidate_source:
            path = candidate
            source = candidate_source
            break
    if path is None or source is None:
        raise RuntimeError("Could not find flash_mla_sparse_fwd marker")

    if "_DG_SM120_PREFILL_COMPILED_BMM = None" in source:
        source = re.sub(
            r"_DG_SM120_PREFILL_COMPILED_BMM = None\n\n.*?\n\n(?=def flash_mla_sparse_fwd\()",
            helper,
            source,
            count=1,
            flags=re.S,
        )
    elif "def _sm120_flash_mla_sparse_prefill_fwd" in source:
        source = re.sub(
            r"def _sm120_flash_mla_sparse_prefill_fwd\(.*?\n\n(?=def flash_mla_sparse_fwd\()",
            helper,
            source,
            count=1,
            flags=re.S,
        )
    else:
        source = source.replace(marker, helper + marker, 1)

    old = """    results = flash_mla_cuda.sparse_prefill_fwd(
        q, kv, indices, sm_scale, d_v, attn_sink, topk_length, out
    )
    return results
"""
    new = """    if q.is_cuda and torch.cuda.get_device_capability(q.device)[0] >= 12:
        return _sm120_flash_mla_sparse_prefill_fwd(
            q, kv, indices, sm_scale, d_v, attn_sink, topk_length, out
        )
    results = flash_mla_cuda.sparse_prefill_fwd(
        q, kv, indices, sm_scale, d_v, attn_sink, topk_length, out
    )
    return results
"""
    if old in source:
        source = source.replace(old, new, 1)
    elif "return _sm120_flash_mla_sparse_prefill_fwd(" not in source:
        raise RuntimeError(f"Could not patch flash_mla_sparse_fwd body in {path}")
    path.write_text(source)


def patch_flashmla_sparse_prefill_workspace_factor() -> None:
    path = Path(
        "/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/"
        "mla/flashmla_sparse.py"
    )
    source = path.read_text()
    if "DG_SM120_FLASHMLA_PREFILL_WORKSPACE_FACTOR" in source:
        return
    if "import os\n" not in source:
        source = source.replace(
            "from dataclasses import dataclass\n",
            "from dataclasses import dataclass\nimport os\n",
            1,
        )
    old = "    return max_model_len * 5\n"
    new = """    factor = int(os.environ.get("DG_SM120_FLASHMLA_PREFILL_WORKSPACE_FACTOR", "1"))
    return max_model_len * max(1, factor)
"""
    if old not in source:
        raise RuntimeError(f"Could not patch FlashMLA sparse prefill workspace in {path}")
    path.write_text(source.replace(old, new, 1))


def patch_vllm_memory_breakdown_logging() -> None:
    path = Path("/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_worker.py")
    source = path.read_text()
    if "DG_SM120_LOG_MEMORY" in source:
        return
    old = "            logger.debug(msg)\n\n        if self.use_v2_model_runner:\n"
    new = """            if os.environ.get("DG_SM120_LOG_MEMORY", "1").lower() in (
                "1",
                "true",
                "yes",
                "on",
            ):
                logger.info(msg)
            else:
                logger.debug(msg)

        if self.use_v2_model_runner:
"""
    if old not in source:
        raise RuntimeError(f"Could not patch vLLM memory breakdown logging in {path}")
    path.write_text(source.replace(old, new, 1))


def patch_flashmla_sparse_decode() -> None:
    path = Path(
        "/usr/local/lib/python3.12/dist-packages/vllm/third_party/"
        "flashmla/flash_mla_interface.py"
    )
    source = path.read_text()
    marker = "def flash_mla_with_kvcache(\n"
    helper = '''def _sm120_dequant_deepseek_v4_mla_cache(
    cache: torch.Tensor,
    linear_indices: torch.Tensor,
    head_dim: int,
) -> torch.Tensor:
    if cache.shape[2] != 1:
        raise RuntimeError("SM120 decode fallback expects single-KV-head MLA cache")
    if head_dim != 512 or cache.shape[-1] < 584:
        raise RuntimeError(
            f"Unsupported SM120 MLA cache layout: head_dim={head_dim}, "
            f"token_bytes={cache.shape[-1]}"
        )

    fp8_dim = 448
    bf16_dim = 64
    quant_block = 64
    n_quant_blocks = 7
    token_data_size = fp8_dim + bf16_dim * 2
    scale_dim = 8
    block_size = cache.shape[1]
    cache_flat = cache.squeeze(2).reshape(cache.shape[0], -1)
    rows = []

    block_ids = torch.div(linear_indices, block_size, rounding_mode="floor")
    block_offsets = linear_indices % block_size
    for block_id, block_offset in zip(block_ids.tolist(), block_offsets.tolist()):
        token_start = block_offset * token_data_size
        scale_start = block_size * token_data_size + block_offset * scale_dim
        token_bytes = cache_flat[block_id, token_start: token_start + token_data_size]
        scale_bytes = cache_flat[block_id, scale_start: scale_start + n_quant_blocks]

        fp8_values = token_bytes[:fp8_dim].contiguous().view(torch.float8_e4m3fn)
        fp8_values = fp8_values.to(torch.float32)
        exponents = scale_bytes.to(torch.float32)
        scales = torch.exp2(exponents - 127.0)
        scales = torch.where(exponents == 0, torch.zeros_like(scales), scales)
        nope = fp8_values * scales.repeat_interleave(quant_block, dim=-1)
        rope = token_bytes[fp8_dim:token_data_size].contiguous().view(torch.bfloat16)
        rows.append(torch.cat((nope, rope.to(torch.float32)), dim=-1))

    if not rows:
        return torch.empty((0, head_dim), device=cache.device, dtype=torch.float32)
    return torch.stack(rows, dim=0)


def _sm120_flash_mla_sparse_decode_fwd(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    indices: torch.Tensor,
    topk_length: Optional[torch.Tensor],
    attn_sink: Optional[torch.Tensor],
    extra_k_cache: Optional[torch.Tensor],
    extra_indices_in_kvcache: Optional[torch.Tensor],
    extra_topk_length: Optional[torch.Tensor],
    head_dim_v: int,
    softmax_scale: float,
    out: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    if q.shape[1] != 1:
        raise RuntimeError("SM120 sparse decode fallback expects seq_len_q == 1")
    if out is None:
        out = torch.empty(
            (q.shape[0], q.shape[1], q.shape[2], head_dim_v),
            device=q.device,
            dtype=q.dtype,
        )
    import deep_gemm
    import os
    gather_indexed = getattr(
        getattr(deep_gemm, "_C", None),
        "sm120_dequantize_and_gather_indexed_k_cache",
        None,
    )
    workspace_decode = getattr(
        getattr(deep_gemm, "_C", None),
        "sm120_sparse_mla_decode_from_bf16_workspace",
        None,
    )
    workspace_split_decode = getattr(
        getattr(deep_gemm, "_C", None),
        "sm120_sparse_mla_decode_from_bf16_workspace_split",
        None,
    )
    if (
        os.environ.get("DG_SM120_INDEXED_BF16_BMM", "1") == "1"
        and gather_indexed is not None
        and head_dim_v == q.shape[-1]
    ):
        main_topk_full = indices.shape[-1]
        main_topk = min(
            main_topk_full,
            int(os.environ.get("DG_SM120_MAIN_TOPK_CAP", main_topk_full)),
        )
        indices_for_bmm = indices[..., :main_topk]

        extra_topk_full = (
            extra_indices_in_kvcache.shape[-1]
            if extra_k_cache is not None and extra_indices_in_kvcache is not None
            else 0
        )
        extra_topk = extra_topk_full
        if extra_topk_full:
            if extra_k_cache.shape[1] <= 2:
                default_extra_cap = 4
            elif extra_k_cache.shape[1] <= 64:
                default_extra_cap = 128
            else:
                default_extra_cap = extra_topk_full
            extra_topk = min(
                extra_topk_full,
                int(os.environ.get("DG_SM120_EXTRA_TOPK_CAP", default_extra_cap)),
            )
            extra_indices_for_bmm = extra_indices_in_kvcache[..., :extra_topk]
        else:
            extra_indices_for_bmm = None
        total_topk = main_topk + extra_topk
        try:
            from vllm.v1.worker.workspace import current_workspace_manager

            (kv_workspace,) = current_workspace_manager().get_simultaneous(
                ((q.shape[0], total_topk, q.shape[-1]), q.dtype),
            )
        except Exception:
            kv_workspace = torch.empty(
                (q.shape[0], total_topk, q.shape[-1]),
                device=q.device,
                dtype=q.dtype,
            )

        gather_indexed(
            kv_workspace, k_cache, indices_for_bmm, topk_length, k_cache.shape[1], 0
        )
        if extra_topk:
            gather_indexed(
                kv_workspace,
                extra_k_cache,
                extra_indices_for_bmm,
                extra_topk_length,
                extra_k_cache.shape[1],
                main_topk,
            )

        if (
            os.environ.get("DG_SM120_WORKSPACE_FUSED_ATTENTION", "1") == "1"
            and workspace_decode is not None
        ):
            selected_workspace_decode = workspace_decode
            if (
                os.environ.get("DG_SM120_WORKSPACE_SPLIT_ATTENTION", "1") == "1"
                and workspace_split_decode is not None
            ):
                selected_workspace_decode = workspace_split_decode
            return selected_workspace_decode(
                q,
                kv_workspace,
                topk_length,
                extra_topk_length if extra_topk else None,
                attn_sink,
                main_topk,
                extra_topk,
                head_dim_v,
                softmax_scale,
                out,
            )

        batch = q.shape[0]
        main_pos = torch.arange(main_topk, device=q.device)
        if topk_length is None:
            valid_main = torch.ones(
                (batch, main_topk), device=q.device, dtype=torch.bool
            )
        else:
            main_lens = topk_length.to(torch.long).clamp(0, main_topk)
            valid_main = main_pos.unsqueeze(0) < main_lens.reshape(-1, 1)
        main_idx = (
            indices_for_bmm[:, 0, :]
            if indices_for_bmm.dim() == 3
            else indices_for_bmm
        )
        valid_main = valid_main & (main_idx >= 0)

        if extra_topk:
            extra_pos = torch.arange(extra_topk, device=q.device)
            if extra_topk_length is None:
                valid_extra = torch.ones(
                    (batch, extra_topk), device=q.device, dtype=torch.bool
                )
            else:
                extra_lens = extra_topk_length.to(torch.long).clamp(0, extra_topk)
                valid_extra = extra_pos.unsqueeze(0) < extra_lens.reshape(-1, 1)
            extra_idx = (
                extra_indices_for_bmm[:, 0, :]
                if extra_indices_for_bmm.dim() == 3
                else extra_indices_for_bmm
            )
            valid_extra = valid_extra & (extra_idx >= 0)
            valid = torch.cat((valid_main, valid_extra), dim=-1)
        else:
            valid = valid_main

        active_heads = min(
            q.shape[2], int(os.environ.get("DG_SM120_ACTIVE_HEADS", q.shape[2]))
        )
        q_decode = q[:, 0, :active_heads, :]
        scores = torch.matmul(q_decode, kv_workspace.transpose(1, 2))
        scores = scores.to(torch.float32).mul_(float(softmax_scale))
        valid_any = valid.any(dim=-1)
        scores.masked_fill_(~valid.unsqueeze(1), float("-inf"))
        scores = torch.where(
            valid_any.reshape(batch, 1, 1),
            scores,
            torch.zeros_like(scores),
        )
        row_lse = torch.logsumexp(scores, dim=-1)
        probs = torch.softmax(scores, dim=-1).to(q.dtype)
        probs.masked_fill_(~valid_any.reshape(batch, 1, 1), 0)
        attn_out = torch.matmul(probs, kv_workspace)
        if attn_sink is not None:
            gate = torch.sigmoid(
                row_lse
                - attn_sink[:active_heads].to(row_lse.dtype).reshape(1, -1)
            )
            attn_out = attn_out * gate.to(attn_out.dtype).unsqueeze(-1)
        row_lse = row_lse.masked_fill(~valid_any.unsqueeze(1), float("-inf"))
        attn_out = attn_out.unsqueeze(1)
        if out is None:
            if active_heads == q.shape[2]:
                out = attn_out
            else:
                full_out = q.new_zeros(
                    (q.shape[0], q.shape[1], q.shape[2], head_dim_v)
                )
                full_out[:, :, :active_heads, :] = attn_out
                out = full_out
        else:
            out[:, :, :active_heads, :].copy_(attn_out)
            if active_heads < q.shape[2]:
                out[:, :, active_heads:, :].zero_()
        if active_heads == q.shape[2]:
            lse = row_lse.unsqueeze(-1)
        else:
            lse = q.new_full(
                (q.shape[0], q.shape[2], q.shape[1]),
                float("-inf"),
                dtype=torch.float32,
            )
            lse[:, :active_heads, :] = row_lse.unsqueeze(-1)
        return out, lse

    return deep_gemm._C.sm120_sparse_mla_decode(
        q, k_cache, indices, topk_length, attn_sink, extra_k_cache,
        extra_indices_in_kvcache, extra_topk_length, head_dim_v,
        softmax_scale, out
    )

    lse = torch.empty((q.shape[0], q.shape[2], q.shape[1]), device=q.device,
                      dtype=torch.float32)
    q_f = q.to(torch.float32)
    sink = attn_sink.to(torch.float32) if attn_sink is not None else None

    for batch_idx in range(q.shape[0]):
        pieces = []
        main_limit = indices.shape[-1]
        if topk_length is not None:
            main_limit = int(topk_length[batch_idx].item())
        main_indices = indices[batch_idx, 0, :main_limit].to(torch.long)
        main_valid = (main_indices >= 0) & (
            main_indices < k_cache.shape[0] * k_cache.shape[1]
        )
        main_indices = main_indices[main_valid]
        if main_indices.numel() > 0:
            pieces.append(
                _sm120_dequant_deepseek_v4_mla_cache(
                    k_cache, main_indices, q.shape[-1]
                )
            )

        if extra_k_cache is not None and extra_indices_in_kvcache is not None:
            extra_limit = extra_indices_in_kvcache.shape[-1]
            if extra_topk_length is not None:
                extra_limit = int(extra_topk_length[batch_idx].item())
            extra_indices = extra_indices_in_kvcache[batch_idx, 0, :extra_limit]
            extra_indices = extra_indices.to(torch.long)
            extra_valid = (extra_indices >= 0) & (
                extra_indices < extra_k_cache.shape[0] * extra_k_cache.shape[1]
            )
            extra_indices = extra_indices[extra_valid]
            if extra_indices.numel() > 0:
                pieces.append(
                    _sm120_dequant_deepseek_v4_mla_cache(
                        extra_k_cache, extra_indices, q.shape[-1]
                    )
                )

        if not pieces:
            out[batch_idx].zero_()
            lse[batch_idx].fill_(float("-inf"))
            continue

        selected = torch.cat(pieces, dim=0)
        scores = torch.matmul(q_f[batch_idx, 0], selected[:, : q.shape[-1]].T)
        scores = scores * softmax_scale
        token_lse = torch.logsumexp(scores, dim=-1)
        probs = torch.softmax(scores, dim=-1)
        token_out = torch.matmul(probs, selected[:, :head_dim_v])
        if sink is not None:
            token_out = token_out * torch.sigmoid(token_lse - sink).unsqueeze(-1)
        out[batch_idx, 0].copy_(token_out.to(out.dtype))
        lse[batch_idx, :, 0].copy_(token_lse)

    return out, lse


'''
    if helper not in source:
        if marker not in source:
            raise RuntimeError(f"Could not find flash_mla_with_kvcache marker in {path}")
        source = source.replace(marker, helper + marker, 1)

    old = """        out, lse, new_tile_scheduler_metadata, new_num_splits = flash_mla_cuda.sparse_decode_fwd(
            q, k_cache, indices_in_kvcache, topk_length, attn_sink,
            sched_meta.tile_scheduler_metadata, sched_meta.num_splits,
            extra_k_cache, extra_indices_in_kvcache, extra_topk_length,
            head_dim_v, softmax_scale, out
        )
"""
    new = """        if q.is_cuda and torch.cuda.get_device_capability(q.device)[0] >= 12:
            return _sm120_flash_mla_sparse_decode_fwd(
                q, k_cache, indices_in_kvcache, topk_length, attn_sink,
                extra_k_cache, extra_indices_in_kvcache, extra_topk_length,
                head_dim_v, softmax_scale, out
            )
        out, lse, new_tile_scheduler_metadata, new_num_splits = flash_mla_cuda.sparse_decode_fwd(
            q, k_cache, indices_in_kvcache, topk_length, attn_sink,
            sched_meta.tile_scheduler_metadata, sched_meta.num_splits,
            extra_k_cache, extra_indices_in_kvcache, extra_topk_length,
            head_dim_v, softmax_scale, out
        )
"""
    if old not in source:
        raise RuntimeError(f"Could not patch sparse_decode_fwd body in {path}")
    path.write_text(source.replace(old, new, 1))


def patch_flashmla_sparse_full_context_decode() -> None:
    """Use DeepGEMM's direct SM120 full-context sparse MLA decode path.

    The small-VRAM DeepSeek V4 Flash profile caps max_model_len so sparse top-k
    covers the entire decode context. In that case the selected-token list is
    logically dense and can be reconstructed from vLLM's block table and actual
    sequence lengths inside the CUDA extension. Keep this branch narrow so
    mixed prefill/decode, speculative decode, and larger contexts fall back to
    the existing sparse-index conversion path.
    """
    path = Path(
        "/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/"
        "backends/mla/flashmla_sparse.py"
    )
    source = path.read_text()
    if "        return capability.major in [9, 10, 12]\n" not in source:
        source = source.replace(
            "        return capability.major in [9, 10]\n",
            "        return capability.major in [9, 10, 12]\n",
            1,
        )
    if "current_platform.is_device_capability(120)" not in source:
        source = source.replace(
            "current_platform.is_device_capability_family(100)",
            "(current_platform.is_device_capability_family(100) or current_platform.is_device_capability(120))",
        )

    old_field = "    req_id_per_token: torch.Tensor\n"
    new_field = "    req_id_per_token: torch.Tensor\n    seq_lens: torch.Tensor\n"
    if new_field not in source:
        if old_field not in source:
            raise RuntimeError(f"Could not add seq_lens metadata field in {path}")
        source = source.replace(old_field, new_field, 1)

    if "            seq_lens=cm.seq_lens,\n" not in source:
        old_init = "            req_id_per_token=req_id_per_token,\n"
        new_init = (
            "            req_id_per_token=req_id_per_token,\n"
            "            seq_lens=cm.seq_lens,\n"
        )
        if old_init not in source:
            raise RuntimeError(f"Could not add seq_lens metadata init in {path}")
        source = source.replace(old_init, new_init, 1)

    if "sm120_direct_full_context_decode" in source:
        old_full_context_bmm = '''                    q_decode = reshape_query_for_spec_decode(q, num_decodes)
                    q_decode = q_decode[:, 0, :, : self.kv_lora_rank]
                    scores = torch.matmul(q_decode, kv_workspace.transpose(1, 2))
                    scores = scores.to(torch.float32).mul_(self.softmax_scale)
                    positions = torch.arange(
                        max_decode_context,
                        device=q.device,
                        dtype=attn_metadata.seq_lens.dtype,
                    )
                    valid_tokens = (
                        positions.unsqueeze(0)
                        < attn_metadata.seq_lens[:num_decodes].unsqueeze(1)
                    )
                    scores.masked_fill_(~valid_tokens.unsqueeze(1), float("-inf"))
                    probs = torch.softmax(scores, dim=-1).to(q.dtype)
                    attn_out = torch.matmul(probs, kv_workspace).unsqueeze(1)
                    return reshape_attn_output_for_spec_decode(attn_out)
'''
        new_full_context_bmm = '''                    workspace_decode = getattr(
                        getattr(deep_gemm, "_C", None),
                        "sm120_sparse_mla_decode_from_bf16_workspace",
                        None,
                    )
                    workspace_split_decode = getattr(
                        getattr(deep_gemm, "_C", None),
                        "sm120_sparse_mla_decode_from_bf16_workspace_split",
                        None,
                    )
                    if (
                        os.environ.get("DG_SM120_WORKSPACE_FUSED_ATTENTION", "1") == "1"
                        and workspace_decode is not None
                    ):
                        selected_workspace_decode = workspace_decode
                        if (
                            os.environ.get("DG_SM120_WORKSPACE_SPLIT_ATTENTION", "1") == "1"
                            and workspace_split_decode is not None
                        ):
                            selected_workspace_decode = workspace_split_decode
                        logger.warning_once(
                            "Using SM120 full-context fused workspace sparse MLA decode"
                        )
                        q_decode = reshape_query_for_spec_decode(q, num_decodes)
                        q_decode = q_decode[:, :, :, : self.kv_lora_rank]
                        attn_out, _ = selected_workspace_decode(
                            q_decode,
                            kv_workspace,
                            attn_metadata.seq_lens[:num_decodes],
                            None,
                            None,
                            max_decode_context,
                            0,
                            self.kv_lora_rank,
                            self.softmax_scale,
                            None,
                        )
                        return reshape_attn_output_for_spec_decode(attn_out)

                    q_decode = reshape_query_for_spec_decode(q, num_decodes)
                    q_decode = q_decode[:, 0, :, : self.kv_lora_rank]
                    scores = torch.matmul(q_decode, kv_workspace.transpose(1, 2))
                    scores = scores.to(torch.float32).mul_(self.softmax_scale)
                    positions = torch.arange(
                        max_decode_context,
                        device=q.device,
                        dtype=attn_metadata.seq_lens.dtype,
                    )
                    valid_tokens = (
                        positions.unsqueeze(0)
                        < attn_metadata.seq_lens[:num_decodes].unsqueeze(1)
                    )
                    scores.masked_fill_(~valid_tokens.unsqueeze(1), float("-inf"))
                    probs = torch.softmax(scores, dim=-1).to(q.dtype)
                    attn_out = torch.matmul(probs, kv_workspace).unsqueeze(1)
                    return reshape_attn_output_for_spec_decode(attn_out)
'''
        if (
            old_full_context_bmm in source
            and "sm120_sparse_mla_decode_from_bf16_workspace" not in source
        ):
            source = source.replace(old_full_context_bmm, new_full_context_bmm, 1)
        source = source.replace(
            """                if gather_full_context is not None:
                    max_decode_context = min(
""",
            """                if gather_full_context is not None:
                    logger.warning_once(
                        "Using SM120 full-context workspace sparse MLA decode"
                    )
                    max_decode_context = min(
""",
            1,
        )
        source = source.replace(
            """            if full_context_decode is not None:
                q_decode = reshape_query_for_spec_decode(q, num_decodes)
""",
            """            if full_context_decode is not None:
                logger.warning_once(
                    "Using SM120 direct full-context sparse MLA decode"
                )
                q_decode = reshape_query_for_spec_decode(q, num_decodes)
""",
            1,
        )
        path.write_text(source)
        return

    anchor = """        # Convert per-request indices to global slots (decode) or workspace
        # offsets (prefill).
        # For FP8 cache: prefill uses workspace mapping (upconverted to BF16)
        # For BF16 cache: always use global cache slots (no workspace)
        # prefill_workspace_starts has been adjusted in-place per chunk so
        # prefill indices automatically come out chunk-local
"""
    insert = '''        sm120_direct_full_context_decode = (
            q.is_cuda
            and torch.cuda.get_device_capability(q.device)[0] >= 12
            and fp8_metadata.num_decode_tokens > 0
            and fp8_metadata.num_prefill_tokens == 0
            and fp8_metadata.decode is not None
            and fp8_metadata.decode.decode_query_len == 1
            and topk_indices.shape[1] >= max(1, attn_metadata.max_seq_len)
            and attn_metadata.max_seq_len <= 768
        )
        if sm120_direct_full_context_decode:
            import deep_gemm
            import os

            if os.environ.get("DG_SM120_FULL_CONTEXT_BF16_BMM", "1") == "1":
                gather_full_context = getattr(
                    getattr(deep_gemm, "_C", None),
                    "sm120_dequantize_and_gather_k_cache",
                    None,
                )
                if gather_full_context is not None:
                    logger.warning_once(
                        "Using SM120 full-context workspace sparse MLA decode"
                    )
                    max_decode_context = min(
                        topk_indices.shape[1],
                        max(1, attn_metadata.max_seq_len),
                    )
                    workspace_manager = current_workspace_manager()
                    (kv_workspace,) = workspace_manager.get_simultaneous(
                        ((num_decodes, max_decode_context, self.kv_lora_rank), q.dtype),
                    )
                    gather_full_context(
                        kv_workspace,
                        kv_c_and_k_pe_cache.view(torch.uint8).unsqueeze(-2),
                        attn_metadata.seq_lens[:num_decodes],
                        None,
                        attn_metadata.block_table[:num_decodes],
                        attn_metadata.block_size,
                        0,
                    )
                    workspace_decode = getattr(
                        getattr(deep_gemm, "_C", None),
                        "sm120_sparse_mla_decode_from_bf16_workspace",
                        None,
                    )
                    workspace_split_decode = getattr(
                        getattr(deep_gemm, "_C", None),
                        "sm120_sparse_mla_decode_from_bf16_workspace_split",
                        None,
                    )
                    if (
                        os.environ.get("DG_SM120_WORKSPACE_FUSED_ATTENTION", "1") == "1"
                        and workspace_decode is not None
                    ):
                        selected_workspace_decode = workspace_decode
                        if (
                            os.environ.get("DG_SM120_WORKSPACE_SPLIT_ATTENTION", "1") == "1"
                            and workspace_split_decode is not None
                        ):
                            selected_workspace_decode = workspace_split_decode
                        logger.warning_once(
                            "Using SM120 full-context fused workspace sparse MLA decode"
                        )
                        q_decode = reshape_query_for_spec_decode(q, num_decodes)
                        q_decode = q_decode[:, :, :, : self.kv_lora_rank]
                        attn_out, _ = selected_workspace_decode(
                            q_decode,
                            kv_workspace,
                            attn_metadata.seq_lens[:num_decodes],
                            None,
                            None,
                            max_decode_context,
                            0,
                            self.kv_lora_rank,
                            self.softmax_scale,
                            None,
                        )
                        return reshape_attn_output_for_spec_decode(attn_out)

                    q_decode = reshape_query_for_spec_decode(q, num_decodes)
                    q_decode = q_decode[:, 0, :, : self.kv_lora_rank]
                    scores = torch.matmul(q_decode, kv_workspace.transpose(1, 2))
                    scores = scores.to(torch.float32).mul_(self.softmax_scale)
                    positions = torch.arange(
                        max_decode_context,
                        device=q.device,
                        dtype=attn_metadata.seq_lens.dtype,
                    )
                    valid_tokens = (
                        positions.unsqueeze(0)
                        < attn_metadata.seq_lens[:num_decodes].unsqueeze(1)
                    )
                    scores.masked_fill_(~valid_tokens.unsqueeze(1), float("-inf"))
                    probs = torch.softmax(scores, dim=-1).to(q.dtype)
                    attn_out = torch.matmul(probs, kv_workspace).unsqueeze(1)
                    return reshape_attn_output_for_spec_decode(attn_out)

            full_context_decode = getattr(
                getattr(deep_gemm, "_C", None),
                "sm120_sparse_mla_decode_full_context",
                None,
            )
            if full_context_decode is not None:
                logger.warning_once(
                    "Using SM120 direct full-context sparse MLA decode"
                )
                q_decode = reshape_query_for_spec_decode(q, num_decodes)
                attn_out, _ = full_context_decode(
                    q_decode,
                    kv_c_and_k_pe_cache.view(torch.uint8).unsqueeze(-2),
                    attn_metadata.block_table,
                    attn_metadata.seq_lens,
                    attn_metadata.req_id_per_token,
                    None,
                    self.kv_lora_rank,
                    self.softmax_scale,
                    None,
                )
                return reshape_attn_output_for_spec_decode(attn_out)

'''
    if anchor not in source:
        raise RuntimeError(f"Could not patch SM120 direct sparse MLA decode in {path}")
    source = source.replace(anchor, insert + anchor, 1)
    path.write_text(source)


def patch_sparse_topk_for_small_context() -> None:
    """Keep sparse MLA top-k legal when memory forces a small max_model_len.

    DeepSeek V4 Flash advertises index_topk=2048. That is fine for normal
    contexts, but an FP8-KV smoke profile with max_model_len < 2048 cannot ask
    persistent_topk/FlashInfer to select more entries than exist. Cap top-k at
    max_model_len consistently across the model indexer and MLA backends.
    """
    model_paths = (
        Path("/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/deepseek_v4.py"),
        Path("/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/deepseek_v2.py"),
    )
    for path in model_paths:
        if not path.exists():
            continue
        source = path.read_text()
        source = source.replace(
            "self.topk_tokens = config.index_topk\n",
            "self.topk_tokens = min(config.index_topk, vllm_config.model_config.max_model_len)\n",
        )
        source = source.replace(
            "topk_tokens = config.index_topk\n",
            "topk_tokens = min(config.index_topk, vllm_config.model_config.max_model_len)\n",
        )
        path.write_text(source)

    backend_paths = (
        Path(
            "/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/"
            "backends/mla/flashinfer_mla_sparse.py"
        ),
        Path(
            "/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/"
            "backends/mla/flashmla_sparse.py"
        ),
    )
    for path in backend_paths:
        source = path.read_text()
        old = "        self.topk_tokens = vllm_config.model_config.hf_config.index_topk\n"
        new = (
            "        self.topk_tokens = min(\n"
            "            vllm_config.model_config.hf_config.index_topk,\n"
            "            vllm_config.model_config.max_model_len,\n"
            "        )\n"
        )
        if old in source:
            source = source.replace(old, new, 1)
        elif "vllm_config.model_config.max_model_len" not in source:
            raise RuntimeError(f"Could not patch sparse MLA top-k in {path}")
        path.write_text(source)

    path = Path(
        "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/"
        "layers/sparse_attn_indexer.py"
    )
    source = path.read_text()
    old_prefill = """            topk_indices = topk_indices_buffer[
                chunk.token_start : chunk.token_end, :topk_tokens
            ]
"""
    new_prefill = """            effective_topk_tokens = min(
                topk_tokens,
                logits.shape[-1],
                max(1, int((chunk.cu_seqlen_ke - chunk.cu_seqlen_ks).max().item())),
            )
            topk_indices = topk_indices_buffer[
                chunk.token_start : chunk.token_end, :effective_topk_tokens
            ]
"""
    if old_prefill in source:
        source = source.replace(old_prefill, new_prefill, 1)

    source = source.replace(
        "                    topk_tokens,\n"
        "                )\n",
        "                    effective_topk_tokens,\n"
        "                )\n",
        1,
    )
    source = source.replace(
        "                    topk_tokens,\n"
        "                )\n",
        "                    effective_topk_tokens,\n"
        "                )\n",
        1,
    )

    old_decode = """        topk_indices = topk_indices_buffer[:num_padded_tokens, :topk_tokens]

        if current_platform.is_cuda():
"""
    new_decode = """        effective_topk_tokens = min(
            topk_tokens,
            logits.shape[-1],
            max(1, int(seq_lens.max().item())),
        )
        topk_indices = topk_indices_buffer[:num_padded_tokens, :effective_topk_tokens]

        if current_platform.is_cuda():
"""
    if old_decode in source:
        source = source.replace(old_decode, new_decode, 1)

    old_decode_image = """        topk_indices = topk_indices_buffer[:num_padded_tokens, :topk_tokens]

        if current_platform.is_cuda() and topk_tokens in (512, 1024, 2048):
            workspace_manager = current_workspace_manager()
            (topk_workspace,) = workspace_manager.get_simultaneous(
                ((RADIX_TOPK_WORKSPACE_SIZE,), torch.uint8),
            )
            torch.ops._C.persistent_topk(
                logits,
                seq_lens,
                topk_indices,
                topk_workspace,
                topk_tokens,
                attn_metadata.max_seq_len,
            )
        else:
"""
    new_decode_image = """        effective_topk_tokens = min(
            topk_tokens,
            logits.shape[-1],
            max(1, int(seq_lens.max().item())),
        )
        topk_indices = topk_indices_buffer[:num_padded_tokens, :effective_topk_tokens]

        if current_platform.is_cuda():
            use_sm120_topk_fallback = (
                hidden_states.is_cuda
                and torch.cuda.get_device_capability(hidden_states.device)[0] >= 12
            )
            if use_sm120_topk_fallback or effective_topk_tokens not in (512, 1024, 2048):
                # Correctness fallback for small-context SM120 FP8 KV profiles.
                logits_2d = logits.reshape(num_rows, -1)
                seq_lens_1d = seq_lens.reshape(-1)
                topk_indices.fill_(-1)
                for row_idx in range(num_rows):
                    row_len = min(int(seq_lens_1d[row_idx].item()), logits_2d.shape[-1])
                    row_topk = min(effective_topk_tokens, row_len)
                    if row_topk <= 0:
                        continue
                    row_indices = torch.topk(
                        logits_2d[row_idx, :row_len], row_topk, dim=-1
                    ).indices.to(torch.int32)
                    topk_indices[row_idx, :row_topk] = row_indices
            else:
                workspace_manager = current_workspace_manager()
                (topk_workspace,) = workspace_manager.get_simultaneous(
                    ((RADIX_TOPK_WORKSPACE_SIZE,), torch.uint8),
                )
                torch.ops._C.persistent_topk(
                    logits,
                    seq_lens,
                    topk_indices,
                    topk_workspace,
                    effective_topk_tokens,
                    min(attn_metadata.max_seq_len, logits.shape[-1]),
                )
        else:
"""
    if old_decode_image in source:
        source = source.replace(old_decode_image, new_decode_image, 1)

    source = source.replace(
        "                topk_tokens,\n"
        "                attn_metadata_narrowed.max_seq_len,\n"
        "            )\n",
        "                effective_topk_tokens,\n"
        "                min(attn_metadata_narrowed.max_seq_len, logits.shape[-1]),\n"
        "            )\n",
        1,
    )
    old_cuda_decode_topk = """        if current_platform.is_cuda():
            workspace_manager = current_workspace_manager()
            (topk_workspace,) = workspace_manager.get_simultaneous(
                ((RADIX_TOPK_WORKSPACE_SIZE,), torch.uint8),
            )
            torch.ops._C.persistent_topk(
                logits,
                seq_lens,
                topk_indices,
                topk_workspace,
                effective_topk_tokens,
                min(attn_metadata_narrowed.max_seq_len, logits.shape[-1]),
            )
        else:
"""
    new_cuda_decode_topk = """        if current_platform.is_cuda():
            if hidden_states.is_cuda and torch.cuda.get_device_capability(hidden_states.device)[0] >= 12:
                logits_2d = logits.reshape(num_rows, -1)
                seq_lens_1d = seq_lens.reshape(-1)
                topk_indices.fill_(-1)
                for row_idx in range(num_rows):
                    row_len = min(int(seq_lens_1d[row_idx].item()), logits_2d.shape[-1])
                    row_topk = min(effective_topk_tokens, row_len)
                    if row_topk <= 0:
                        continue
                    row_indices = torch.topk(
                        logits_2d[row_idx, :row_len], row_topk, dim=-1
                    ).indices.to(torch.int32)
                    topk_indices[row_idx, :row_topk] = row_indices
            else:
                workspace_manager = current_workspace_manager()
                (topk_workspace,) = workspace_manager.get_simultaneous(
                    ((RADIX_TOPK_WORKSPACE_SIZE,), torch.uint8),
                )
                torch.ops._C.persistent_topk(
                    logits,
                    seq_lens,
                    topk_indices,
                    topk_workspace,
                    effective_topk_tokens,
                    min(attn_metadata_narrowed.max_seq_len, logits.shape[-1]),
                )
        else:
"""
    if old_cuda_decode_topk in source:
        source = source.replace(old_cuda_decode_topk, new_cuda_decode_topk, 1)
    source = source.replace(
        "                    topk_tokens,\n"
        "                )\n",
        "                    effective_topk_tokens,\n"
        "                )\n",
        1,
    )
    source = source.replace(
        "                    topk_tokens,\n"
        "                )\n",
        "                    effective_topk_tokens,\n"
        "                )\n",
        1,
    )
    path.write_text(source)


def patch_sm120_sparse_indexer_graph_safe_topk() -> None:
    """Make the SM120 sparse-indexer top-k path CUDA-graph safe.

    The earlier correctness fallback used Python .item() and torch.topk per row.
    That works eagerly, but vLLM captures decode with CUDA graphs and host reads
    from CUDA tensors invalidate capture. The generic CUDA top_k_per_row_decode
    kernel already accepts ragged seq_lens, so use it for SM120 instead of the
    Python fallback.
    """
    path = Path(
        "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/"
        "layers/sparse_attn_indexer.py"
    )
    source = path.read_text()

    source = source.replace(
        """        effective_topk_tokens = min(
            topk_tokens,
            logits.shape[-1],
            max(1, int(seq_lens.max().item())),
        )
""",
        """        effective_topk_tokens = min(
            topk_tokens,
            logits.shape[-1],
            max(1, attn_metadata.max_seq_len),
        )
""",
    )

    source = source.replace(
        """        effective_topk_tokens = min(
            topk_tokens,
            logits.shape[-1],
            max(1, int(seq_lens.max().item())),
        )
""",
        """        effective_topk_tokens = min(
            topk_tokens,
            logits.shape[-1],
            max(1, attn_metadata_narrowed.max_seq_len),
        )
""",
    )

    old_sm120_fallback = """        if current_platform.is_cuda():
            use_sm120_topk_fallback = (
                hidden_states.is_cuda
                and torch.cuda.get_device_capability(hidden_states.device)[0] >= 12
            )
            if use_sm120_topk_fallback or effective_topk_tokens not in (512, 1024, 2048):
                # Correctness fallback for small-context SM120 FP8 KV profiles.
                logits_2d = logits.reshape(num_rows, -1)
                seq_lens_1d = seq_lens.reshape(-1)
                topk_indices.fill_(-1)
                for row_idx in range(num_rows):
                    row_len = min(int(seq_lens_1d[row_idx].item()), logits_2d.shape[-1])
                    row_topk = min(effective_topk_tokens, row_len)
                    if row_topk <= 0:
                        continue
                    row_indices = torch.topk(
                        logits_2d[row_idx, :row_len], row_topk, dim=-1
                    ).indices.to(torch.int32)
                    topk_indices[row_idx, :row_topk] = row_indices
            else:
                workspace_manager = current_workspace_manager()
                (topk_workspace,) = workspace_manager.get_simultaneous(
                    ((RADIX_TOPK_WORKSPACE_SIZE,), torch.uint8),
                )
                torch.ops._C.persistent_topk(
                    logits,
                    seq_lens,
                    topk_indices,
                    topk_workspace,
                    effective_topk_tokens,
                    min(attn_metadata.max_seq_len, logits.shape[-1]),
                )
"""
    new_sm120_fallback = """        if current_platform.is_cuda():
            use_sm120_decode_topk = (
                hidden_states.is_cuda
                and torch.cuda.get_device_capability(hidden_states.device)[0] >= 12
            )
            if use_sm120_decode_topk or effective_topk_tokens not in (512, 1024, 2048):
                torch.ops._C.top_k_per_row_decode(
                    logits,
                    next_n,
                    seq_lens,
                    topk_indices,
                    num_rows,
                    logits.stride(0),
                    logits.stride(1),
                    effective_topk_tokens,
                )
            else:
                workspace_manager = current_workspace_manager()
                (topk_workspace,) = workspace_manager.get_simultaneous(
                    ((RADIX_TOPK_WORKSPACE_SIZE,), torch.uint8),
                )
                torch.ops._C.persistent_topk(
                    logits,
                    seq_lens,
                    topk_indices,
                    topk_workspace,
                    effective_topk_tokens,
                    min(attn_metadata.max_seq_len, logits.shape[-1]),
                )
"""
    if old_sm120_fallback in source:
        source = source.replace(old_sm120_fallback, new_sm120_fallback, 1)

    old_sm120_narrowed_fallback = """        if current_platform.is_cuda():
            if hidden_states.is_cuda and torch.cuda.get_device_capability(hidden_states.device)[0] >= 12:
                logits_2d = logits.reshape(num_rows, -1)
                seq_lens_1d = seq_lens.reshape(-1)
                topk_indices.fill_(-1)
                for row_idx in range(num_rows):
                    row_len = min(int(seq_lens_1d[row_idx].item()), logits_2d.shape[-1])
                    row_topk = min(effective_topk_tokens, row_len)
                    if row_topk <= 0:
                        continue
                    row_indices = torch.topk(
                        logits_2d[row_idx, :row_len], row_topk, dim=-1
                    ).indices.to(torch.int32)
                    topk_indices[row_idx, :row_topk] = row_indices
            else:
                workspace_manager = current_workspace_manager()
                (topk_workspace,) = workspace_manager.get_simultaneous(
                    ((RADIX_TOPK_WORKSPACE_SIZE,), torch.uint8),
                )
                torch.ops._C.persistent_topk(
                    logits,
                    seq_lens,
                    topk_indices,
                    topk_workspace,
                    effective_topk_tokens,
                    min(attn_metadata_narrowed.max_seq_len, logits.shape[-1]),
                )
"""
    new_sm120_narrowed_fallback = """        if current_platform.is_cuda():
            use_sm120_decode_topk = (
                hidden_states.is_cuda
                and torch.cuda.get_device_capability(hidden_states.device)[0] >= 12
            )
            if use_sm120_decode_topk or effective_topk_tokens not in (512, 1024, 2048):
                torch.ops._C.top_k_per_row_decode(
                    logits,
                    next_n,
                    seq_lens,
                    topk_indices,
                    num_rows,
                    logits.stride(0),
                    logits.stride(1),
                    effective_topk_tokens,
                )
            else:
                workspace_manager = current_workspace_manager()
                (topk_workspace,) = workspace_manager.get_simultaneous(
                    ((RADIX_TOPK_WORKSPACE_SIZE,), torch.uint8),
                )
                torch.ops._C.persistent_topk(
                    logits,
                    seq_lens,
                    topk_indices,
                    topk_workspace,
                    effective_topk_tokens,
                    min(attn_metadata_narrowed.max_seq_len, logits.shape[-1]),
                )
"""
    if old_sm120_narrowed_fallback in source:
        source = source.replace(old_sm120_narrowed_fallback, new_sm120_narrowed_fallback, 1)

    if ".item()" in source and "use_sm120_decode_topk" not in source:
        raise RuntimeError(f"Could not patch SM120 graph-safe sparse top-k in {path}")

    path.write_text(source)


def patch_sm120_sparse_indexer_full_context_decode() -> None:
    """Skip sparse-indexer scoring when top-k already covers full context.

    In the 2x RTX PRO 6000 FP8-KV profile we cap max_model_len to fit VRAM,
    which also caps top-k to max_model_len. Once top-k >= current max seq len,
    the learned sparse-indexer logits cannot exclude any valid token, so the
    decode path can fill logical token indices directly and avoid a full paged
    MQA logits pass plus top-k on every sparse-attention layer.
    """
    path = Path(
        "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/"
        "layers/sparse_attn_indexer.py"
    )
    source = path.read_text()
    if "sm120_full_context_decode" in source:
        source = source.replace(
            "attn_metadata_narrowed.max_seq_len", "attn_metadata.max_seq_len"
        )
        source = source.replace("                topk_indices_buffer[:num_padded_tokens] = -1\n", "")
        path.write_text(source)
        return

    anchor = "        seq_lens = decode_metadata.seq_lens[:batch_size]\n"
    insert = '''        sm120_full_context_decode = (
            hidden_states.is_cuda
            and torch.cuda.get_device_capability(hidden_states.device)[0] >= 12
            and topk_tokens >= max(1, attn_metadata.max_seq_len)
        )
        if sm120_full_context_decode:
            import deep_gemm

            fill_all_indices = getattr(
                getattr(deep_gemm, "_C", None),
                "sm120_fill_decode_all_indices",
                None,
            )
            if fill_all_indices is not None:
                topk_indices = topk_indices_buffer[
                    :num_padded_tokens, :topk_tokens
                ]
                fill_all_indices(
                    topk_indices, seq_lens, num_padded_tokens, next_n, topk_tokens
                )
                if decode_metadata.requires_padding:
                    topk_indices = unpack_seq_triton(
                        topk_indices.reshape(batch_size, -1, topk_indices.shape[-1]),
                        decode_lens,
                    )
                    topk_indices_buffer[
                        : topk_indices.shape[0], : topk_indices.shape[-1]
                    ] = topk_indices
                return topk_indices_buffer

'''
    if anchor not in source:
        raise RuntimeError(f"Could not patch SM120 full-context sparse decode in {path}")
    source = source.replace(anchor, anchor + insert, 1)
    path.write_text(source)



def patch_deepseek_v4_compressor_graph_native_metadata() -> None:
    """Build compressor metadata on device for SM120.

    Upstream builds token_to_req_indices with CPU repeat_interleave + pinned
    H2D copy and separately launches a block-table clamp. The SM120 path fuses
    token-to-request fill and block-table nonnegative clamp into one graph-safe
    device launch, matching SGLang-style graph-native metadata prep.
    """
    path = Path(
        "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/"
        "layers/deepseek_compressor.py"
    )
    source = path.read_text()

    new_build = """    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> CompressorMetadata:
        num_reqs = common_attn_metadata.num_reqs
        num_tokens = common_attn_metadata.num_actual_tokens
        token_to_req_indices = self.token_to_req_indices[:num_tokens]
        block_table = common_attn_metadata.block_table_tensor
        build_metadata = None
        if token_to_req_indices.is_cuda and common_attn_metadata.query_start_loc.is_cuda:
            try:
                import deep_gemm

                build_metadata = getattr(
                    getattr(deep_gemm, "_C", None),
                    "sm120_build_compressor_metadata",
                    None,
                )
            except Exception:
                build_metadata = None
        if (
            build_metadata is not None
            and torch.cuda.get_device_capability(token_to_req_indices.device)[0] >= 12
        ):
            build_metadata(
                token_to_req_indices,
                common_attn_metadata.query_start_loc,
                block_table,
                num_reqs,
            )
        else:
            query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
            query_lens = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]
            x = torch.repeat_interleave(torch.arange(num_reqs), query_lens).pin_memory()
            token_to_req_indices = self.token_to_req_indices[: x.shape[0]]
            token_to_req_indices.copy_(x, non_blocking=True)
            block_table = block_table.clamp_(min=0)
        return CompressorMetadata(
            block_table=block_table,
            slot_mapping=common_attn_metadata.slot_mapping,
            block_size=self.block_size,
            token_to_req_indices=token_to_req_indices,
        )
"""

    old_upstream = """    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> CompressorMetadata:
        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
        num_reqs = common_attn_metadata.num_reqs
        query_lens = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]
        x = torch.repeat_interleave(torch.arange(num_reqs), query_lens).pin_memory()
        token_to_req_indices = self.token_to_req_indices[: x.shape[0]]
        token_to_req_indices.copy_(x, non_blocking=True)
        return CompressorMetadata(
            block_table=common_attn_metadata.block_table_tensor.clamp_(min=0),
            slot_mapping=common_attn_metadata.slot_mapping,
            block_size=self.block_size,
            token_to_req_indices=token_to_req_indices,
        )
"""

    old_fill_only = """    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> CompressorMetadata:
        num_reqs = common_attn_metadata.num_reqs
        num_tokens = common_attn_metadata.num_actual_tokens
        token_to_req_indices = self.token_to_req_indices[:num_tokens]
        fill_token_to_req = None
        if token_to_req_indices.is_cuda and common_attn_metadata.query_start_loc.is_cuda:
            try:
                import deep_gemm

                fill_token_to_req = getattr(
                    getattr(deep_gemm, "_C", None),
                    "sm120_fill_token_to_req_indices",
                    None,
                )
            except Exception:
                fill_token_to_req = None
        if (
            fill_token_to_req is not None
            and torch.cuda.get_device_capability(token_to_req_indices.device)[0] >= 12
        ):
            fill_token_to_req(
                token_to_req_indices, common_attn_metadata.query_start_loc, num_reqs
            )
        else:
            query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
            query_lens = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]
            x = torch.repeat_interleave(torch.arange(num_reqs), query_lens).pin_memory()
            token_to_req_indices = self.token_to_req_indices[: x.shape[0]]
            token_to_req_indices.copy_(x, non_blocking=True)
        return CompressorMetadata(
            block_table=common_attn_metadata.block_table_tensor.clamp_(min=0),
            slot_mapping=common_attn_metadata.slot_mapping,
            block_size=self.block_size,
            token_to_req_indices=token_to_req_indices,
        )
"""

    if "sm120_build_compressor_metadata" in source:
        path.write_text(source)
        return
    if old_fill_only in source:
        source = source.replace(old_fill_only, new_build, 1)
    elif old_upstream in source:
        source = source.replace(old_upstream, new_build, 1)
    else:
        raise RuntimeError(f"Could not patch SM120 compressor metadata in {path}")
    path.write_text(source)



def patch_flashmla_sparse_req_id_graph_native_metadata() -> None:
    """Build FlashMLA sparse req_id_per_token on device for SM120.

    The FlashMLA sparse metadata builder still used NumPy repeat on the host,
    then copied a freshly-built request-id vector into a persistent GPU buffer.
    This is another per-build metadata expansion on the scheduler/Python path.
    Reuse the SM120 token-to-request CUDA kernel so CUDA graph replay consumes a
    stable device buffer without host repeat/copy work.
    """
    path = Path(
        "/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/"
        "backends/mla/flashmla_sparse.py"
    )
    source = path.read_text()
    if "sm120_fill_token_to_req_indices" in source:
        path.write_text(source)
        return

    old = """        starts = np.asarray(cm.query_start_loc_cpu, dtype=np.int32)
        seg_lengths = np.diff(starts)
        req_id_per_token = np.repeat(
            np.arange(seg_lengths.shape[0], dtype=np.int32), seg_lengths
        )
        # Zero-fill for cudagraphs
        self.req_id_per_token_buffer.fill_(0)
        self.req_id_per_token_buffer[: req_id_per_token.shape[0]].copy_(
            torch.from_numpy(req_id_per_token), non_blocking=True
        )
        req_id_per_token = self.req_id_per_token_buffer[:num_tokens]
"""
    new = """        req_id_per_token = self.req_id_per_token_buffer[:num_tokens]
        fill_token_to_req = None
        if req_id_per_token.is_cuda and cm.query_start_loc.is_cuda:
            try:
                import deep_gemm

                fill_token_to_req = getattr(
                    getattr(deep_gemm, "_C", None),
                    "sm120_fill_token_to_req_indices",
                    None,
                )
            except Exception:
                fill_token_to_req = None
        if (
            fill_token_to_req is not None
            and torch.cuda.get_device_capability(req_id_per_token.device)[0] >= 12
        ):
            fill_token_to_req(req_id_per_token, cm.query_start_loc, cm.num_reqs)
        else:
            starts = np.asarray(cm.query_start_loc_cpu, dtype=np.int32)
            seg_lengths = np.diff(starts)
            req_id_per_token_cpu = np.repeat(
                np.arange(seg_lengths.shape[0], dtype=np.int32), seg_lengths
            )
            # Zero-fill for cudagraphs
            self.req_id_per_token_buffer.fill_(0)
            self.req_id_per_token_buffer[: req_id_per_token_cpu.shape[0]].copy_(
                torch.from_numpy(req_id_per_token_cpu), non_blocking=True
            )
            req_id_per_token = self.req_id_per_token_buffer[:num_tokens]
"""
    if old not in source:
        raise RuntimeError(f"Could not patch SM120 FlashMLA sparse req-id metadata in {path}")
    source = source.replace(old, new, 1)
    path.write_text(source)

def patch_deepseek_v4_sparse_swa_graph_native_metadata() -> None:
    """Build sparse-SWA token metadata on device for SM120.

    Sparse SWA used the same CPU repeat_interleave + pinned H2D token-to-request
    construction as the compressor metadata path, plus separate CUDA launches
    for slot validity and decode-lens tail clearing. Fuse those graph-replay
    metadata chores into one SM120 extension call so decode/spec replay avoids
    host-side metadata expansion.
    """
    path = Path(
        "/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/"
        "backends/mla/sparse_swa.py"
    )
    source = path.read_text()

    # Keep a CPU view of prefill seq_lens so SM120 prefill workspace sizing can
    # avoid a per-layer GPU max().item() sync while still using the exact
    # context+query lengths that the metadata builder already materialized.
    source = source.replace(
        "    prefill_seq_lens: torch.Tensor | None = None\n"
        "    prefill_gather_lens: torch.Tensor | None = None\n",
        "    prefill_seq_lens: torch.Tensor | None = None\n"
        "    prefill_seq_lens_cpu: torch.Tensor | None = None\n"
        "    prefill_gather_lens: torch.Tensor | None = None\n",
        1,
    )
    source = source.replace(
        "            query_start_loc,\n"
        "        )\n\n"
        "        # Per-layer-type tile-scheduler plan holders.",
        "            query_start_loc,\n"
        "            common_attn_metadata.seq_lens_cpu,\n"
        "        )\n\n"
        "        # Per-layer-type tile-scheduler plan holders.",
        1,
    )
    source = source.replace(
        "        query_start_loc: torch.Tensor,\n"
        "    ) -> dict[str, torch.Tensor | None]:\n",
        "        query_start_loc: torch.Tensor,\n"
        "        seq_lens_cpu: torch.Tensor,\n"
        "    ) -> dict[str, torch.Tensor | None]:\n",
        1,
    )
    source = source.replace(
        "            result[\"prefill_seq_lens\"] = seq_lens[num_decodes:]\n"
        "            result[\"prefill_gather_lens\"] = pfx_gather_lens\n",
        "            result[\"prefill_seq_lens\"] = seq_lens[num_decodes:]\n"
        "            result[\"prefill_seq_lens_cpu\"] = seq_lens_cpu[num_decodes:]\n"
        "            result[\"prefill_gather_lens\"] = pfx_gather_lens\n",
        1,
    )
    decode_metadata_already_patched = "sm120_build_sparse_swa_decode_metadata" in source

    if not decode_metadata_already_patched:
        old = """        # NOTE: Ensure all metadata tensors maintain fixed memory addresses
        # for CUDA graph compatibility.
        query_lens = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]
        x = torch.repeat_interleave(torch.arange(num_reqs), query_lens).pin_memory()
        token_to_req_indices = self.token_to_req_indices[: x.shape[0]]
        token_to_req_indices.copy_(x, non_blocking=True)

        is_valid_token = self.is_valid_token[: slot_mapping.shape[0]]
        is_valid_token.copy_(slot_mapping >= 0)

        if num_decode_tokens > 0:
            self.decode_swa_lens[num_decode_tokens:] = 0
            _compute_swa_indices_and_lens_kernel[(num_decode_tokens,)](
"""
        new = """        # NOTE: Ensure all metadata tensors maintain fixed memory addresses
        # for CUDA graph compatibility. On SM120, keep this on-device so replay
        # avoids CPU repeat_interleave, pinned copies, and small fill kernels.
        token_to_req_indices = self.token_to_req_indices[: slot_mapping.shape[0]]
        is_valid_token = self.is_valid_token[: slot_mapping.shape[0]]
        sm120_sparse_swa_decode_metadata_built = False
        build_sparse_swa_decode_metadata = None
        build_sparse_swa_metadata = None
        if token_to_req_indices.is_cuda and query_start_loc.is_cuda:
            try:
                import deep_gemm

                _dg_c = getattr(deep_gemm, "_C", None)
                build_sparse_swa_decode_metadata = getattr(
                    _dg_c,
                    "sm120_build_sparse_swa_decode_metadata",
                    None,
                )
                build_sparse_swa_metadata = getattr(
                    _dg_c,
                    "sm120_build_sparse_swa_metadata",
                    None,
                )
            except Exception:
                build_sparse_swa_decode_metadata = None
                build_sparse_swa_metadata = None
        if (
            build_sparse_swa_decode_metadata is not None
            and torch.cuda.get_device_capability(token_to_req_indices.device)[0] >= 12
        ):
            build_sparse_swa_decode_metadata(
                token_to_req_indices,
                is_valid_token,
                query_start_loc,
                slot_mapping,
                self.decode_swa_lens,
                self.decode_swa_indices,
                seq_lens,
                block_table,
                num_reqs,
                num_decode_tokens,
                self.window_size,
                self.block_size,
            )
            sm120_sparse_swa_decode_metadata_built = True
        elif (
            build_sparse_swa_metadata is not None
            and torch.cuda.get_device_capability(token_to_req_indices.device)[0] >= 12
        ):
            build_sparse_swa_metadata(
                token_to_req_indices,
                is_valid_token,
                query_start_loc,
                slot_mapping,
                self.decode_swa_lens,
                num_reqs,
                num_decode_tokens,
            )
        else:
            query_lens = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]
            x = torch.repeat_interleave(torch.arange(num_reqs), query_lens).pin_memory()
            token_to_req_indices = self.token_to_req_indices[: x.shape[0]]
            token_to_req_indices.copy_(x, non_blocking=True)
            is_valid_token.copy_(slot_mapping >= 0)
            if num_decode_tokens > 0:
                self.decode_swa_lens[num_decode_tokens:] = 0

        if num_decode_tokens > 0 and not sm120_sparse_swa_decode_metadata_built:
            _compute_swa_indices_and_lens_kernel[(num_decode_tokens,)](
"""
        if old in source:
            source = source.replace(old, new, 1)
        else:
            old_patched = """        build_sparse_swa_metadata = None
        if token_to_req_indices.is_cuda and query_start_loc.is_cuda:
            try:
                import deep_gemm

                build_sparse_swa_metadata = getattr(
                    getattr(deep_gemm, "_C", None),
                    "sm120_build_sparse_swa_metadata",
                    None,
                )
            except Exception:
                build_sparse_swa_metadata = None
        if (
            build_sparse_swa_metadata is not None
            and torch.cuda.get_device_capability(token_to_req_indices.device)[0] >= 12
        ):
            build_sparse_swa_metadata(
                token_to_req_indices,
                is_valid_token,
                query_start_loc,
                slot_mapping,
                self.decode_swa_lens,
                num_reqs,
                num_decode_tokens,
            )
        else:
            query_lens = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]
            x = torch.repeat_interleave(torch.arange(num_reqs), query_lens).pin_memory()
            token_to_req_indices = self.token_to_req_indices[: x.shape[0]]
            token_to_req_indices.copy_(x, non_blocking=True)
            is_valid_token.copy_(slot_mapping >= 0)
            if num_decode_tokens > 0:
                self.decode_swa_lens[num_decode_tokens:] = 0

        if num_decode_tokens > 0:
            _compute_swa_indices_and_lens_kernel[(num_decode_tokens,)](
"""
            if old_patched not in source:
                raise RuntimeError(f"Could not patch SM120 sparse SWA metadata in {path}")
            source = source.replace(old_patched, new, 1)

    if "sm120_build_sparse_swa_prefill_metadata" not in source:
        old = """            _compute_prefill_metadata_kernel[(1,)](
                pfx_gather_lens,
                seq_lens,
                query_start_loc,
                num_prefills,
                num_decodes,
                self.window_size,
                BLOCK_SIZE=triton.next_power_of_2(num_prefills),
            )
"""
        new = """            build_sparse_swa_prefill_metadata = None
            if pfx_gather_lens.is_cuda and query_start_loc.is_cuda:
                try:
                    import deep_gemm

                    build_sparse_swa_prefill_metadata = getattr(
                        getattr(deep_gemm, "_C", None),
                        "sm120_build_sparse_swa_prefill_metadata",
                        None,
                    )
                except Exception:
                    build_sparse_swa_prefill_metadata = None
            if (
                build_sparse_swa_prefill_metadata is not None
                and torch.cuda.get_device_capability(pfx_gather_lens.device)[0] >= 12
            ):
                build_sparse_swa_prefill_metadata(
                    pfx_gather_lens,
                    seq_lens,
                    query_start_loc,
                    num_prefills,
                    num_decodes,
                    self.window_size,
                )
            else:
                _compute_prefill_metadata_kernel[(1,)](
                    pfx_gather_lens,
                    seq_lens,
                    query_start_loc,
                    num_prefills,
                    num_decodes,
                    self.window_size,
                    BLOCK_SIZE=triton.next_power_of_2(num_prefills),
                )
"""
        if old not in source:
            raise RuntimeError(f"Could not patch SM120 sparse SWA prefill metadata in {path}")
        source = source.replace(old, new, 1)
    path.write_text(source)


def patch_deepseek_v4_sm120_compressor_overlap() -> None:
    """Avoid peak-memory OOM from overlapping compressor and KV insert on SM120.

    The FP8 KV smoke profile leaves very little scratch space after loading
    DeepSeek V4 Flash on two 96 GB cards. vLLM normally overlaps the C128
    compressor with KV insertion; on SM120 this can OOM before the first token.
    Running these two pieces sequentially preserves the fp8_ds_mla KV cache
    path while we work on lower-footprint production kernels.
    """
    path = Path(
        "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/"
        "layers/deepseek_v4_attention.py"
    )
    source = path.read_text()
    if "sm120_sequential_compressor = hidden_states.is_cuda" not in source:
        marker = "        # Overlap kv_insert with whichever of indexer/compressor is present.\n"
        insert = (
            "        sm120_sequential_compressor = (\n"
            "            __import__(\"os\").environ.get(\"DG_SM120_SEQUENTIAL_COMPRESSOR\", \"1\") != \"0\"\n"
            "            and hidden_states.is_cuda\n"
            "            and torch.cuda.get_device_capability(hidden_states.device)[0] >= 12\n"
            "        )\n\n"
        )
        if marker not in source:
            raise RuntimeError(f"Could not find compressor overlap marker in {path}")
        source = source.replace(marker, insert + marker, 1)

    old_indexer = """            maybe_execute_in_parallel(
                lambda: self.indexer(
                    hidden_states, qr, positions, self.indexer_rotary_emb
                ),
                kv_insert_and_compress,
                self.ln_events[0],
                self.ln_events[1],
                self.aux_stream,
            )
"""
    new_indexer = """            if sm120_sequential_compressor:
                self.indexer(hidden_states, qr, positions, self.indexer_rotary_emb)
                kv_insert_and_compress()
            else:
                maybe_execute_in_parallel(
                    lambda: self.indexer(
                        hidden_states, qr, positions, self.indexer_rotary_emb
                    ),
                    kv_insert_and_compress,
                    self.ln_events[0],
                    self.ln_events[1],
                    self.aux_stream,
                )
"""
    if old_indexer in source:
        source = source.replace(old_indexer, new_indexer, 1)

    old_compressor = """            # Compressor on default, kv_insert on aux.
            maybe_execute_in_parallel(
                lambda: self.compressor(hidden_states, positions, self.rotary_emb),
                lambda: self._fused_qnorm_rope_kv_insert(
                    q, kv, positions, attn_metadata
                ),
                self.ln_events[0],
                self.ln_events[1],
                self.aux_stream,
            )
"""
    new_compressor = """            # Compressor on default, kv_insert on aux.
            if sm120_sequential_compressor:
                self.compressor(hidden_states, positions, self.rotary_emb)
                self._fused_qnorm_rope_kv_insert(q, kv, positions, attn_metadata)
            else:
                maybe_execute_in_parallel(
                    lambda: self.compressor(hidden_states, positions, self.rotary_emb),
                    lambda: self._fused_qnorm_rope_kv_insert(
                        q, kv, positions, attn_metadata
                    ),
                    self.ln_events[0],
                    self.ln_events[1],
                    self.aux_stream,
                )
"""
    if old_compressor in source:
        source = source.replace(old_compressor, new_compressor, 1)
    elif "if sm120_sequential_compressor:" not in source:
        raise RuntimeError(f"Could not patch compressor overlap in {path}")
    source = source.replace(
        "        sm120_sequential_compressor = (\n"
        "            hidden_states.is_cuda\n"
        "            and torch.cuda.get_device_capability(hidden_states.device)[0] >= 12\n"
        "        )\n",
        "        sm120_sequential_compressor = (\n"
        "            __import__(\"os\").environ.get(\"DG_SM120_SEQUENTIAL_COMPRESSOR\", \"1\") != \"0\"\n"
        "            and hidden_states.is_cuda\n"
        "            and torch.cuda.get_device_capability(hidden_states.device)[0] >= 12\n"
        "        )\n",
    )
    while source.count("DG_SM120_SEQUENTIAL_COMPRESSOR") > 1:
        first = source.find("        sm120_sequential_compressor = (\n")
        second = source.find("        sm120_sequential_compressor = (\n", first + 1)
        if second < 0:
            break
        end = source.find("\n\n", second)
        if end < 0:
            break
        source = source[:second] + source[end + 2 :]

    old_q_pad = '''        # Pad q to FlashMLA-required head count (64 or 128)
        if self.n_local_heads < self.padded_heads:
            pad_size = self.padded_heads - self.n_local_heads
            q = F.pad(q, (0, 0, 0, pad_size), value=0.0)
'''
    new_q_pad = '''        # Native FlashMLA requires 64/128 heads. The SM120 fallback kernels
        # use DG_SM120_ACTIVE_HEADS and can write only the real local heads into
        # the preallocated padded output, so avoid a per-layer F.pad launch.
        if self.n_local_heads < self.padded_heads and not (
            q.is_cuda and torch.cuda.get_device_capability(q.device)[0] >= 12
        ):
            pad_size = self.padded_heads - self.n_local_heads
            q = F.pad(q, (0, 0, 0, pad_size), value=0.0)
'''
    if old_q_pad in source:
        source = source.replace(old_q_pad, new_q_pad, 1)
    elif (
        "DG_SM120_ACTIVE_HEADS" not in source
        and "avoid a per-layer F.pad launch" not in source
    ):
        raise RuntimeError(f"Could not patch SM120 q padding in {path}")

    old_output_assert = '''        assert output.shape == q.shape, (
            f"output buffer shape {output.shape} must match q shape {q.shape}"
        )
'''
    new_output_assert = '''        sm120_padded_output = (
            q.is_cuda
            and torch.cuda.get_device_capability(q.device)[0] >= 12
            and output.dim() == q.dim()
            and output.shape[0] == q.shape[0]
            and output.shape[1] >= q.shape[1]
            and output.shape[2:] == q.shape[2:]
        )
        assert output.shape == q.shape or sm120_padded_output, (
            f"output buffer shape {output.shape} must match q shape {q.shape}"
        )
'''
    if old_output_assert in source:
        source = source.replace(old_output_assert, new_output_assert, 1)
    elif (
        "sm120_padded_output" not in source
        and "must match q shape" in source
    ):
        raise RuntimeError(f"Could not patch SM120 output assert in {path}")
    path.write_text(source)


def patch_deepseek_v4_cache_gather_bounds() -> None:
    path = Path(
        "/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/ops/"
        "deepseek_v4_ops/cache_utils.py"
    )
    source = path.read_text()
    old_unpatched = """        # Get physical block index from block table
        block_table_row_ptr = block_table_ptr + batch_idx * max_blocks_per_seq
        physical_block_idx = tl.load(block_table_row_ptr + block_in_seq)  # int32

        # int64: physical_block_idx * block_stride can exceed 2^31 with many
"""
    old_continue_patch = """        # Get physical block index from block table. Small SM120 smoke
        # profiles can expose unused sparse/SWA entries; skip them instead of
        # forming invalid byte pointers into the packed fp8_ds_mla cache.
        if block_in_seq >= max_blocks_per_seq:
            continue
        block_table_row_ptr = block_table_ptr + batch_idx * max_blocks_per_seq
        physical_block_idx = tl.load(block_table_row_ptr + block_in_seq)  # int32
        if physical_block_idx < 0:
            continue

        # int64: physical_block_idx * block_stride can exceed 2^31 with many
"""
    new = """        # Get physical block index from block table. Small SM120 smoke
        # profiles can expose unused sparse/SWA entries; mask them instead of
        # forming invalid byte pointers into the packed fp8_ds_mla cache.
        valid_block = block_in_seq < max_blocks_per_seq
        block_table_row_ptr = block_table_ptr + batch_idx * max_blocks_per_seq
        physical_block_idx = tl.load(
            block_table_row_ptr + block_in_seq, mask=valid_block, other=-1
        )  # int32
        valid_block = valid_block & (physical_block_idx >= 0)

        # int64: physical_block_idx * block_stride can exceed 2^31 with many
"""
    if old_continue_patch in source:
        source = source.replace(old_continue_patch, new, 1)
    elif old_unpatched in source:
        source = source.replace(old_unpatched, new, 1)
    elif "valid_block = valid_block & (physical_block_idx >= 0)" not in source:
        raise RuntimeError(f"Could not patch DeepSeek V4 gather bounds in {path}")

    source = source.replace(
        "                x_uint8 = tl.load(token_fp8_ptr + offsets, mask=mask, other=0)\n",
        "                x_uint8 = tl.load(token_fp8_ptr + offsets, mask=valid_block & mask, other=0)\n",
    )
    source = source.replace(
        "                encoded_scale = tl.load(token_scale_ptr + qblock_idx)\n",
        "                encoded_scale = tl.load(token_scale_ptr + qblock_idx, mask=valid_block, other=0)\n",
    )
    source = source.replace(
        "                tl.store(output_row_ptr + offsets, x_dequant.to(tl.bfloat16), mask=mask)\n",
        "                tl.store(output_row_ptr + offsets, x_dequant.to(tl.bfloat16), mask=valid_block & mask)\n",
    )
    source = source.replace(
        "            bf16_vals = tl.load(bf16_cache_ptr + chunk_offsets)\n",
        "            bf16_vals = tl.load(bf16_cache_ptr + chunk_offsets, mask=valid_block, other=0.0)\n",
    )
    source = source.replace(
        "            tl.store(output_row_ptr + bf16_output_offset + chunk_offsets, bf16_vals)\n",
        "            tl.store(output_row_ptr + bf16_output_offset + chunk_offsets, bf16_vals, mask=valid_block)\n",
    )

    marker = "def dequantize_and_gather_k_cache(\n"
    helper = '''def _sm120_dequantize_and_gather_k_cache_torch(
    out: torch.Tensor,
    k_cache: torch.Tensor,
    seq_lens: torch.Tensor,
    gather_lens: torch.Tensor | None,
    block_table: torch.Tensor,
    block_size: int,
    offset: int,
) -> None:
    fp8_dim = 448
    bf16_dim = 64
    token_data_size = fp8_dim + bf16_dim * 2
    scale_dim = 8
    quant_block = 64
    n_quant_blocks = 7
    cache_flat = k_cache.reshape(k_cache.shape[0], -1)

    for batch_idx in range(seq_lens.shape[0]):
        seq_len = int(seq_lens[batch_idx].item())
        gather_len = seq_len
        if gather_lens is not None:
            gather_len = int(gather_lens[batch_idx].item())
        start_pos = max(0, seq_len - gather_len)

        for out_idx in range(gather_len):
            pos = start_pos + out_idx
            block_in_seq = pos // block_size
            if block_in_seq < 0 or block_in_seq >= block_table.shape[-1]:
                continue
            physical_block = int(block_table[batch_idx, block_in_seq].item())
            if physical_block < 0 or physical_block >= k_cache.shape[0]:
                continue
            pos_in_block = pos % block_size
            token_start = pos_in_block * token_data_size
            scale_start = block_size * token_data_size + pos_in_block * scale_dim

            token_bytes = cache_flat[
                physical_block, token_start : token_start + token_data_size
            ]
            scale_bytes = cache_flat[
                physical_block, scale_start : scale_start + n_quant_blocks
            ]
            fp8_values = token_bytes[:fp8_dim].contiguous().view(torch.float8_e4m3fn)
            fp8_values = fp8_values.to(torch.float32)
            exponents = scale_bytes.to(torch.float32)
            scales = torch.exp2(exponents - 127.0)
            scales = torch.where(exponents == 0, torch.zeros_like(scales), scales)
            nope = fp8_values * scales.repeat_interleave(quant_block, dim=-1)
            rope = token_bytes[fp8_dim:token_data_size].contiguous().view(torch.bfloat16)
            row = torch.cat((nope, rope.to(torch.float32)), dim=-1).to(out.dtype)
            out[batch_idx, offset + out_idx, : row.shape[0]].copy_(row)


'''
    if helper not in source:
        if marker not in source:
            raise RuntimeError(f"Could not find gather function marker in {path}")
        source = source.replace(marker, helper + marker, 1)

    old_call = """    num_reqs = seq_lens.shape[0]
    NUM_WORKERS = 128
    _dequantize_and_gather_k_kernel[(num_reqs, NUM_WORKERS)](
"""
    new_call = """    if k_cache.is_cuda and torch.cuda.get_device_capability(k_cache.device)[0] >= 12:
        import deep_gemm
        native_gather = getattr(
            getattr(deep_gemm, "_C", None),
            "sm120_dequantize_and_gather_k_cache",
            None,
        )
        if native_gather is not None:
            native_gather(
                out, k_cache, seq_lens, gather_lens, block_table, block_size, offset
            )
            return
        _sm120_dequantize_and_gather_k_cache_torch(
            out, k_cache, seq_lens, gather_lens, block_table, block_size, offset
        )
        return

    num_reqs = seq_lens.shape[0]
    NUM_WORKERS = 128
    _dequantize_and_gather_k_kernel[(num_reqs, NUM_WORKERS)](
"""
    if old_call in source:
        source = source.replace(old_call, new_call, 1)
    elif "_sm120_dequantize_and_gather_k_cache_torch(" not in source:
        raise RuntimeError(f"Could not patch gather dispatch in {path}")
    path.write_text(source)


def patch_deepseek_v4_layer_profiler() -> None:
    path = Path(
        "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/"
        "models/deepseek_v4.py"
    )
    source = path.read_text()
    if "DG_SM120_PROFILE_LAYER" in source:
        old_guard = "        if not _DG_SM120_PROFILE_LAYER_ENABLED:\n"
        new_guard = """        if (
            not _DG_SM120_PROFILE_LAYER_ENABLED
            or torch.cuda.is_current_stream_capturing()
        ):
"""
        if old_guard in source and "is_current_stream_capturing()" not in source:
            path.write_text(source.replace(old_guard, new_guard, 1))
        return
    old = '''    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        input_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        residual = x
        x, post, comb = self.hc_pre(
            x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base
        )
        x = self.attn_norm(x)
        x = self.attn(positions, x, None)
        x = self.hc_post(x, residual, post, comb)

        residual = x
        x, post, comb = self.hc_pre(
            x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base
        )
        x = self.ffn_norm(x)
        x = self.ffn(x, input_ids)
        x = self.hc_post(x, residual, post, comb)
        return x
'''
    new = '''    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        input_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        if (
            not _DG_SM120_PROFILE_LAYER_ENABLED
            or torch.cuda.is_current_stream_capturing()
        ):
            residual = x
            x, post, comb = self.hc_pre(
                x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base
            )
            x = self.attn_norm(x)
            x = self.attn(positions, x, None)
            x = self.hc_post(x, residual, post, comb)

            residual = x
            x, post, comb = self.hc_pre(
                x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base
            )
            x = self.ffn_norm(x)
            x = self.ffn(x, input_ids)
            x = self.hc_post(x, residual, post, comb)
            return x

        import os
        import time

        cls = type(self)
        prof = getattr(cls, "_dg_sm120_profile", None)
        if prof is None:
            prof = {
                "calls": 0,
                "hc_attn_pre": 0.0,
                "attn_norm": 0.0,
                "attn": 0.0,
                "hc_attn_post": 0.0,
                "hc_ffn_pre": 0.0,
                "ffn_norm": 0.0,
                "ffn": 0.0,
                "hc_ffn_post": 0.0,
            }
            setattr(cls, "_dg_sm120_profile", prof)

        def timed(name, fn):
            torch.cuda.synchronize(x.device)
            start = time.perf_counter()
            value = fn()
            torch.cuda.synchronize(x.device)
            prof[name] += (time.perf_counter() - start) * 1000.0
            return value

        residual = x
        x, post, comb = timed(
            "hc_attn_pre",
            lambda: self.hc_pre(
                x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base
            ),
        )
        x = timed("attn_norm", lambda: self.attn_norm(x))
        x = timed("attn", lambda: self.attn(positions, x, None))
        x = timed("hc_attn_post", lambda: self.hc_post(x, residual, post, comb))

        residual = x
        x, post, comb = timed(
            "hc_ffn_pre",
            lambda: self.hc_pre(
                x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base
            ),
        )
        x = timed("ffn_norm", lambda: self.ffn_norm(x))
        x = timed("ffn", lambda: self.ffn(x, input_ids))
        x = timed("hc_ffn_post", lambda: self.hc_post(x, residual, post, comb))

        prof["calls"] += 1
        if prof["calls"] % _DG_SM120_PROFILE_EVERY == 0:
            keys = [k for k in prof.keys() if k != "calls"]
            total = sum(prof[k] for k in keys)
            line = (
                f"pid={os.getpid()} calls={prof['calls']} total_ms={total:.3f} "
                + " ".join(f"{k}_ms={prof[k]:.3f}" for k in keys)
                + "\\n"
            )
            with open(_DG_SM120_PROFILE_PATH, "a", encoding="utf-8") as f:
                f.write(line)
                f.flush()
        return x
'''
    profile_globals = '''
_DG_SM120_PROFILE_OS = __import__("os")
_DG_SM120_PROFILE_LAYER_ENABLED = (
    _DG_SM120_PROFILE_OS.environ.get("DG_SM120_PROFILE_LAYER", "0") == "1"
)
_DG_SM120_PROFILE_EVERY = int(
    _DG_SM120_PROFILE_OS.environ.get("DG_SM120_PROFILE_EVERY", "2048")
)
_DG_SM120_PROFILE_PATH = _DG_SM120_PROFILE_OS.environ.get(
    "DG_SM120_PROFILE_PATH", "/tmp/dg_sm120_profile.txt"
)
'''
    if "_DG_SM120_PROFILE_LAYER_ENABLED" not in source:
        if "logger = init_logger(__name__)\n" in source:
            source = source.replace(
                "logger = init_logger(__name__)\n",
                "logger = init_logger(__name__)\n" + profile_globals + "\n",
                1,
            )
        else:
            source = profile_globals + "\n" + source
    if old not in source:
        raise RuntimeError(f"Could not patch DeepSeek V4 layer profiler in {path}")
    path.write_text(source.replace(old, new, 1))


def patch_mhc_reusable_buffers() -> None:
    path = Path("/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/mhc.py")
    source = path.read_text()

    if "import os\n" not in source:
        source = source.replace("import math\n", "import math\nimport os\n", 1)

    pre_marker = "\n\ndef mhc_pre(\n"
    pre_helper = r'''

_DG_SM120_MHC_PRE_BUFFERS: dict[tuple, tuple[torch.Tensor, ...]] = {}


def _dg_sm120_mhc_pre_buffers(
    num_tokens: int,
    hc_mult: int,
    hc_mult2: int,
    hc_mult3: int,
    hidden_size: int,
    n_splits: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    key = (
        device.type,
        device.index,
        num_tokens,
        hc_mult,
        hidden_size,
        n_splits,
    )
    cached = _DG_SM120_MHC_PRE_BUFFERS.get(key)
    if cached is not None:
        return cached  # type: ignore[return-value]

    post_mix = torch.empty(num_tokens, hc_mult, dtype=torch.float32, device=device)
    comb_mix = torch.empty(num_tokens, hc_mult2, dtype=torch.float32, device=device)
    layer_input = torch.empty(
        num_tokens, hidden_size, dtype=torch.bfloat16, device=device
    )
    gemm_out_mul = torch.empty(
        n_splits, num_tokens, hc_mult3, dtype=torch.float32, device=device
    )
    gemm_out_sqrsum = torch.empty(
        n_splits, num_tokens, dtype=torch.float32, device=device
    )
    cached = (post_mix, comb_mix, layer_input, gemm_out_mul, gemm_out_sqrsum)
    _DG_SM120_MHC_PRE_BUFFERS[key] = cached
    return cached
'''
    if "_DG_SM120_MHC_PRE_BUFFERS" not in source:
        if pre_marker not in source:
            raise RuntimeError(f"Could not find mhc_pre marker in {path}")
        source = source.replace(pre_marker, pre_helper + pre_marker, 1)

    old_pre_alloc = '''    post_mix = torch.empty(
        num_tokens,
        hc_mult,
        dtype=torch.float32,
        device=residual.device,
    )
    comb_mix = torch.empty(
        num_tokens,
        hc_mult2,
        dtype=torch.float32,
        device=residual.device,
    )
    layer_input = torch.empty(
        num_tokens,
        hidden_size,
        dtype=torch.bfloat16,
        device=residual.device,
    )

    gemm_out_mul = torch.empty(
        n_splits,
        num_tokens,
        hc_mult3,
        dtype=torch.float32,
        device=residual.device,
    )
    gemm_out_sqrsum = torch.empty(
        n_splits,
        num_tokens,
        dtype=torch.float32,
        device=residual.device,
    )
'''
    new_pre_alloc = '''    if os.environ.get("DG_SM120_MHC_REUSE_BUFFERS", "0") != "0":
        post_mix, comb_mix, layer_input, gemm_out_mul, gemm_out_sqrsum = (
            _dg_sm120_mhc_pre_buffers(
                num_tokens,
                hc_mult,
                hc_mult2,
                hc_mult3,
                hidden_size,
                n_splits,
                residual.device,
            )
        )
    else:
        post_mix = torch.empty(
            num_tokens,
            hc_mult,
            dtype=torch.float32,
            device=residual.device,
        )
        comb_mix = torch.empty(
            num_tokens,
            hc_mult2,
            dtype=torch.float32,
            device=residual.device,
        )
        layer_input = torch.empty(
            num_tokens,
            hidden_size,
            dtype=torch.bfloat16,
            device=residual.device,
        )

        gemm_out_mul = torch.empty(
            n_splits,
            num_tokens,
            hc_mult3,
            dtype=torch.float32,
            device=residual.device,
        )
        gemm_out_sqrsum = torch.empty(
            n_splits,
            num_tokens,
            dtype=torch.float32,
            device=residual.device,
        )
'''
    if old_pre_alloc in source:
        source = source.replace(old_pre_alloc, new_pre_alloc, 1)

    post_marker = "\n\ndef mhc_post(\n"
    post_helper = r'''

_DG_SM120_MHC_POST_BUFFERS: dict[tuple, list[torch.Tensor]] = {}


def _dg_sm120_mhc_post_buffer(residual: torch.Tensor) -> torch.Tensor:
    key = (
        residual.device.type,
        residual.device.index,
        tuple(residual.shape),
        tuple(residual.stride()),
        residual.dtype,
    )
    buffers = _DG_SM120_MHC_POST_BUFFERS.get(key)
    if buffers is None:
        buffers = []
        _DG_SM120_MHC_POST_BUFFERS[key] = buffers

    # Avoid returning the same allocation as the input residual. In decode the
    # previous layer output can become the next layer residual; aliasing would
    # turn mhc_post into an accidental in-place read/write kernel.
    residual_ptr = residual.data_ptr()
    for buf in buffers:
        if buf.data_ptr() != residual_ptr:
            return buf

    buf = torch.empty_strided(
        tuple(residual.shape),
        tuple(residual.stride()),
        dtype=residual.dtype,
        device=residual.device,
    )
    buffers.append(buf)
    # Two buffers are enough for the sequential decode path: one may alias the
    # current residual, and the other is safe as the next output destination.
    if len(buffers) > 2:
        del buffers[:-2]
    return buf
'''
    if "_DG_SM120_MHC_POST_BUFFERS" not in source:
        if post_marker not in source:
            raise RuntimeError(f"Could not find mhc_post marker in {path}")
        source = source.replace(post_marker, post_helper + post_marker, 1)

    old_post_alloc = "    out = torch.empty_like(residual)\n"
    new_post_alloc = '''    if os.environ.get("DG_SM120_MHC_REUSE_BUFFERS", "0") != "0":
        out = _dg_sm120_mhc_post_buffer(residual)
    else:
        out = torch.empty_like(residual)
'''
    if "_dg_sm120_mhc_post_buffer(residual)" not in source:
        if old_post_alloc not in source:
            raise RuntimeError(f"Could not patch MHC post allocation in {path}")
        source = source.replace(old_post_alloc, new_post_alloc, 1)

    source = source.replace(
        'os.environ.get("DG_SM120_MHC_REUSE_BUFFERS", "1")',
        'os.environ.get("DG_SM120_MHC_REUSE_BUFFERS", "0")',
    )
    path.write_text(source)


def patch_tp_allreduce_diagnostic_bypass() -> None:
    path = Path("/usr/local/lib/python3.12/dist-packages/vllm/distributed/communication_op.py")
    source = path.read_text()
    if "DG_SM120_BYPASS_TP_ALLREDUCE" in source:
        return
    if "import os\n" not in source:
        source = source.replace(
            "from typing import Any\n",
            "from typing import Any\n\nimport os\n",
            1,
        )
    old = '''def tensor_model_parallel_all_reduce(input_: torch.Tensor) -> torch.Tensor:
    """All-reduce the input tensor across model parallel group."""
    return get_tp_group().all_reduce(input_)
'''
    new = '''def tensor_model_parallel_all_reduce(input_: torch.Tensor) -> torch.Tensor:
    """All-reduce the input tensor across model parallel group."""
    if os.environ.get("DG_SM120_BYPASS_TP_ALLREDUCE", "0") != "0":
        return input_
    return get_tp_group().all_reduce(input_)
'''
    if old not in source:
        raise RuntimeError(f"Could not patch tensor_model_parallel_all_reduce in {path}")
    path.write_text(source.replace(old, new, 1))


def patch_deep_gemm_wrapper_with_starts() -> None:
    path = Path("/usr/local/lib/python3.12/dist-packages/vllm/utils/deep_gemm.py")
    source = path.read_text()
    if "_grouped_with_starts_impl: Callable[..., Any] | None = None" not in source:
        source = source.replace(
            "_grouped_impl: Callable[..., Any] | None = None\n",
            "_grouped_impl: Callable[..., Any] | None = None\n"
            "_grouped_with_starts_impl: Callable[..., Any] | None = None\n",
            1,
        )
    source = source.replace(
        "    global _fp8_gemm_nt_impl, _grouped_impl, _grouped_masked_impl\n"
        "    global _grouped_with_starts_impl\n",
        "    global _fp8_gemm_nt_impl, _grouped_impl, _grouped_masked_impl, _grouped_with_starts_impl\n",
    )
    if "global _fp8_gemm_nt_impl, _grouped_impl, _grouped_masked_impl, _grouped_with_starts_impl" not in source:
        source = source.replace(
            "    global _fp8_gemm_nt_impl, _grouped_impl, _grouped_masked_impl\n",
            "    global _fp8_gemm_nt_impl, _grouped_impl, _grouped_masked_impl, _grouped_with_starts_impl\n",
            1,
        )
    if "global _grouped_impl, _grouped_with_starts_impl, _grouped_masked_impl" not in source:
        source = source.replace(
            "    global _grouped_impl, _grouped_masked_impl, _grouped_fp4_impl\n",
            "    global _grouped_impl, _grouped_with_starts_impl, _grouped_masked_impl, _grouped_fp4_impl\n",
            1,
        )
    if "or _grouped_with_starts_impl is not None" not in source:
        source = source.replace(
            "        or _grouped_impl is not None\n",
            "        or _grouped_impl is not None\n"
            "        or _grouped_with_starts_impl is not None\n",
            1,
        )
    if 'getattr(_dg, "m_grouped_fp8_gemm_nt_contiguous_with_starts", None)' not in source:
        source = source.replace(
            '    _grouped_impl = getattr(_dg, "m_grouped_fp8_gemm_nt_contiguous", None)\n',
            '    _grouped_impl = getattr(_dg, "m_grouped_fp8_gemm_nt_contiguous", None)\n'
            '    _grouped_with_starts_impl = getattr(\n'
            '        _dg, "m_grouped_fp8_gemm_nt_contiguous_with_starts", None\n'
            '    )\n',
            1,
        )
    if "dg_c = getattr(_dg, \"_C\", None)" not in source:
        source = source.replace(
            '    _transform_sf_into_required_layout_impl = getattr(\n'
            '        _dg, "transform_sf_into_required_layout", None\n'
            '    )\n',
            '    _transform_sf_into_required_layout_impl = getattr(\n'
            '        _dg, "transform_sf_into_required_layout", None\n'
            '    )\n'
            '    dg_c = getattr(_dg, "_C", None)\n'
            '    if dg_c is not None:\n'
            '        _cublaslt_gemm_nt_impl = _cublaslt_gemm_nt_impl or getattr(dg_c, "cublaslt_gemm_nt", None)\n'
            '        _fp8_gemm_nt_impl = _fp8_gemm_nt_impl or getattr(dg_c, "fp8_gemm_nt", None)\n'
            '        _fp8_einsum_impl = _fp8_einsum_impl or getattr(dg_c, "fp8_einsum", None)\n'
            '        _grouped_impl = _grouped_impl or getattr(dg_c, "m_grouped_fp8_gemm_nt_contiguous", None)\n'
            '        _grouped_with_starts_impl = _grouped_with_starts_impl or getattr(\n'
            '            dg_c, "m_grouped_fp8_gemm_nt_contiguous_with_starts", None\n'
            '        )\n'
            '        _grouped_masked_impl = _grouped_masked_impl or getattr(dg_c, "fp8_m_grouped_gemm_nt_masked", None)\n'
            '        _grouped_fp4_impl = _grouped_fp4_impl or getattr(dg_c, "m_grouped_fp8_fp4_gemm_nt_contiguous", None)\n'
            '        _fp8_fp4_mqa_logits_impl = _fp8_fp4_mqa_logits_impl or getattr(dg_c, "fp8_fp4_mqa_logits", None)\n'
            '        _fp8_fp4_paged_mqa_logits_impl = _fp8_fp4_paged_mqa_logits_impl or getattr(dg_c, "fp8_fp4_paged_mqa_logits", None)\n'
            '        _get_paged_mqa_logits_metadata_impl = _get_paged_mqa_logits_metadata_impl or getattr(\n'
            '            dg_c, "get_paged_mqa_logits_metadata", None\n'
            '        )\n'
            '        _tf32_hc_prenorm_gemm_impl = _tf32_hc_prenorm_gemm_impl or getattr(dg_c, "tf32_hc_prenorm_gemm", None)\n'
            '        _get_mn_major_tma_aligned_tensor_impl = _get_mn_major_tma_aligned_tensor_impl or getattr(\n'
            '            dg_c, "get_mn_major_tma_aligned_tensor", None\n'
            '        )\n'
            '        _get_mk_alignment_for_contiguous_layout_impl = _get_mk_alignment_for_contiguous_layout_impl or getattr(\n'
            '            dg_c, "get_mk_alignment_for_contiguous_layout", None\n'
            '        )\n'
            '        _transform_sf_into_required_layout_impl = _transform_sf_into_required_layout_impl or getattr(\n'
            '            dg_c, "transform_sf_into_required_layout", None\n'
            '        )\n',
            1,
        )
    wrapper = '''def m_grouped_fp8_gemm_nt_contiguous_with_starts(*args, **kwargs):
    _lazy_init()
    if _grouped_with_starts_impl is None:
        return _missing(*args, **kwargs)
    return _grouped_with_starts_impl(
        *args, disable_ue8m0_cast=not is_deep_gemm_e8m0_used(), **kwargs
    )


'''
    if "def m_grouped_fp8_gemm_nt_contiguous_with_starts(" not in source:
        source = source.replace(
            "\n\ndef fp8_m_grouped_gemm_nt_masked(*args, **kwargs):\n",
            "\n\n" + wrapper + "def fp8_m_grouped_gemm_nt_masked(*args, **kwargs):\n",
            1,
        )
    if '"m_grouped_fp8_gemm_nt_contiguous_with_starts",' not in source:
        source = source.replace(
            '    "m_grouped_fp8_gemm_nt_contiguous",\n',
            '    "m_grouped_fp8_gemm_nt_contiguous",\n'
            '    "m_grouped_fp8_gemm_nt_contiguous_with_starts",\n',
            1,
        )
    path.write_text(source)


def patch_deep_gemm_moe_permute_starts() -> None:
    path = Path(
        "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/"
        "layers/fused_moe/deep_gemm_utils.py"
    )
    source = path.read_text()
    if "ALIGN_EXPERTS: tl.constexpr" not in source:
        source = source.replace(
            "    BLOCK_EXPERT_NUM: tl.constexpr,\n):\n",
            "    BLOCK_EXPERT_NUM: tl.constexpr,\n"
            "    ALIGN_EXPERTS: tl.constexpr,\n"
            "):\n",
            1,
        )
    if "if ALIGN_EXPERTS:\n        tokens_per_expert = round_up_128(tokens_per_expert)" not in source:
        source = source.replace(
            "    tokens_per_expert = round_up_128(tokens_per_expert)\n",
            "    if ALIGN_EXPERTS:\n"
            "        tokens_per_expert = round_up_128(tokens_per_expert)\n",
            1,
        )
    if "    align_experts: bool = True,\n):\n" not in source:
        source = source.replace(
            "    output_index: torch.Tensor,\n):\n",
            "    output_index: torch.Tensor,\n"
            "    align_experts: bool = True,\n"
            "):\n",
            1,
        )
    if "if align_experts:\n        assert m_indices.shape[0] % BLOCK_E == 0" not in source:
        source = source.replace(
            "    assert m_indices.shape[0] % BLOCK_E == 0\n",
            "    if align_experts:\n"
            "        assert m_indices.shape[0] % BLOCK_E == 0\n",
            1,
        )
    if "ALIGN_EXPERTS=align_experts" not in source:
        source = source.replace(
            "        BLOCK_EXPERT_NUM=triton.next_power_of_2(num_experts),\n"
            "    )\n",
            "        BLOCK_EXPERT_NUM=triton.next_power_of_2(num_experts),\n"
            "        ALIGN_EXPERTS=align_experts,\n"
            "    )\n",
            1,
        )

    if "sm120_compact = aq.is_cuda and torch.cuda.get_device_capability(aq.device)[0] >= 12" not in source:
        source = source.replace(
            "    block_m, block_k = get_mk_alignment_for_contiguous_layout()\n\n"
            "    M_sum = compute_aligned_M(\n"
            "        M=topk_ids.size(0),\n"
            "        num_topk=topk_ids.size(1),\n"
            "        local_num_experts=local_num_experts,\n"
            "        alignment=block_m,\n"
            "        expert_tokens_meta=expert_tokens_meta,\n"
            "    )\n",
            "    block_m, block_k = get_mk_alignment_for_contiguous_layout()\n"
            "    sm120_compact = aq.is_cuda and torch.cuda.get_device_capability(aq.device)[0] >= 12\n\n"
            "    if sm120_compact:\n"
            "        # SM120 Cutlass accepts unaligned M. Avoid padding every expert\n"
            "        # to 128 rows during decode; it wastes VRAM and dominates latency.\n"
            "        M_sum = topk_ids.numel()\n"
            "    else:\n"
            "        M_sum = compute_aligned_M(\n"
            "            M=topk_ids.size(0),\n"
            "            num_topk=topk_ids.size(1),\n"
            "            local_num_experts=local_num_experts,\n"
            "            alignment=block_m,\n"
            "            expert_tokens_meta=expert_tokens_meta,\n"
            "        )\n",
            1,
        )
    if "align_experts=not sm120_compact" not in source:
        source = source.replace(
            "        output_index=inv_perm,\n"
            "    )\n",
            "        output_index=inv_perm,\n"
            "        align_experts=not sm120_compact,\n"
            "    )\n",
            1,
        )
    if "expert_start_loc.sub_(expert_num_tokens)" not in source:
        old_return = "    return aq_out, aq_scale_out, expert_ids, inv_perm\n"
        starts_return = (
            "    return aq_out, aq_scale_out, expert_ids, inv_perm, "
            "expert_start_loc, expert_num_tokens\n"
        )
        new_return = (
            "    expert_start_loc.sub_(expert_num_tokens)\n"
            "    return aq_out, aq_scale_out, expert_ids, inv_perm, "
            "expert_start_loc, expert_num_tokens\n"
        )
        if old_return in source:
            source = source.replace(old_return, new_return, 1)
        elif starts_return in source:
            source = source.replace(starts_return, new_return, 1)
    if (
        "ALIGN_EXPERTS=align_experts" not in source
        or "sm120_compact" not in source
        or "expert_start_loc.sub_(expert_num_tokens)" not in source
    ):
        raise RuntimeError(f"Could not patch compact SM120 MoE permute in {path}")
    path.write_text(source)


def patch_deep_gemm_moe_with_starts() -> None:
    path = Path(
        "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/"
        "layers/fused_moe/experts/deep_gemm_moe.py"
    )
    source = path.read_text()
    install_moe_profile = (
        __import__("os").environ.get("DG_SM120_INSTALL_MOE_PROFILE_PATCH", "0") == "1"
    )
    profile_helper = """
_DG_SM120_MOE_PROFILE_COUNTER = 0


def _dg_sm120_profile_moe_shape(stage, a, w, expert_counts):
    # Env-gated routed-shape sampling for SM120 DeepGEMM MoE calls.
    os_mod = __import__("os")
    if os_mod.environ.get("DG_SM120_PROFILE_MOE_SHAPES", "0") != "1":
        return
    if torch.cuda.is_current_stream_capturing():
        return
    try:
        global _DG_SM120_MOE_PROFILE_COUNTER
        _DG_SM120_MOE_PROFILE_COUNTER += 1
        every = max(1, int(os_mod.environ.get("DG_SM120_PROFILE_MOE_EVERY", "1")))
        if _DG_SM120_MOE_PROFILE_COUNTER % every != 0:
            return
        counts_cpu = expert_counts.detach().to("cpu", non_blocking=False)
        active = counts_cpu[counts_cpu > 0]
        w_shape = tuple(int(x) for x in w.shape)
        if len(w_shape) >= 3:
            n = w_shape[-2]
            k_storage = w_shape[-1]
        else:
            n = 0
            k_storage = 0
        record = {
            "call": _DG_SM120_MOE_PROFILE_COUNTER,
            "stage": str(stage),
            "m": int(a.shape[0]),
            "k": int(a.shape[-1]) if a.dim() else 0,
            "weight_shape": w_shape,
            "n": int(n),
            "k_storage": int(k_storage),
            "num_experts": int(counts_cpu.numel()),
            "active_experts": int(active.numel()),
            "max_tokens_per_expert": int(active.max().item()) if active.numel() else 0,
            "min_tokens_per_expert": int(active.min().item()) if active.numel() else 0,
            "sum_tokens": int(counts_cpu.sum().item()),
        }
        path = os_mod.environ.get("DG_SM120_MOE_PROFILE_PATH", "/tmp/dg_sm120_moe_shapes.jsonl")
        with open(path, "a", encoding="utf-8") as f:
            f.write(__import__("json").dumps(record, sort_keys=True) + "\\n")
    except Exception:
        return
"""
    if install_moe_profile and "def _dg_sm120_profile_moe_shape(" not in source:
        if "logger = init_logger(__name__)\n" in source:
            source = source.replace(
                "logger = init_logger(__name__)\n",
                "logger = init_logger(__name__)\n" + profile_helper + "\n",
                1,
            )
        else:
            source = profile_helper + "\n" + source
    # The stock DeepGEMM backend allocates and processes per-expert regions
    # rounded up to 128 rows. For DeepSeek V4 decode on SM120 this inflates
    # M from M*topk (typically 6-48 rows) to roughly local_experts*128 rows,
    # wasting VRAM and making the SM120 grouped Cutlass launch memory-bound.
    compact_workspace_replacements = [
        (
            "        block_m = self.block_shape[0]\n"
            "        M_sum = compute_aligned_M(\n"
            "            M, topk, local_num_experts, block_m, expert_tokens_meta\n"
            "        )\n"
            "        assert M_sum % block_m == 0\n",
            "        block_m = self.block_shape[0]\n"
            "        M_sum = M * topk\n",
        ),
        (
            "        block_m = get_mk_alignment_for_contiguous_layout()[0]\n"
            "        M_sum = compute_aligned_M(\n"
            "            M, topk, local_num_experts, block_m, expert_tokens_meta\n"
            "        )\n"
            "        assert M_sum % block_m == 0\n",
            "        block_m = get_mk_alignment_for_contiguous_layout()[0]\n"
            "        M_sum = M * topk\n",
        ),
    ]
    for old, new in compact_workspace_replacements:
        if old in source:
            source = source.replace(old, new)

    old_forward_msum = """        M_sum = compute_aligned_M(
            M=topk_ids.size(0),
            num_topk=topk_ids.size(1),
            local_num_experts=local_num_experts,
            alignment=get_mk_alignment_for_contiguous_layout()[0],
            expert_tokens_meta=expert_tokens_meta,
        )
"""
    new_forward_msum = """        if hidden_states.is_cuda and torch.cuda.get_device_capability(hidden_states.device)[0] >= 12:
            M_sum = topk_ids.numel()
        else:
            M_sum = compute_aligned_M(
                M=topk_ids.size(0),
                num_topk=topk_ids.size(1),
                local_num_experts=local_num_experts,
                alignment=get_mk_alignment_for_contiguous_layout()[0],
                expert_tokens_meta=expert_tokens_meta,
            )
"""
    if old_forward_msum in source:
        source = source.replace(old_forward_msum, new_forward_msum)
    if "m_grouped_fp8_gemm_nt_contiguous_with_starts" not in source:
        source = source.replace(
            "    m_grouped_fp8_gemm_nt_contiguous,\n",
            "    m_grouped_fp8_gemm_nt_contiguous,\n"
            "    m_grouped_fp8_gemm_nt_contiguous_with_starts,\n",
            1,
        )
    old_unpack = "        a1q, a1q_scale, expert_ids, inv_perm = deepgemm_moe_permute(\n"
    new_unpack = (
        "        a1q, a1q_scale, expert_ids, inv_perm, expert_starts, expert_counts = "
        "deepgemm_moe_permute(\n"
    )
    if old_unpack in source:
        source = source.replace(old_unpack, new_unpack)
    elif "expert_starts, expert_counts = deepgemm_moe_permute" not in source:
        raise RuntimeError(f"Could not patch deepgemm_moe_permute unpack in {path}")

    old_mm1 = """        m_grouped_fp8_gemm_nt_contiguous(
            (a1q, a1q_scale), (w1, self.w1_scale), mm1_out, expert_ids
        )
"""
    new_mm1 = """        m_grouped_fp8_gemm_nt_contiguous_with_starts(
            (a1q, a1q_scale),
            (w1, self.w1_scale),
            mm1_out,
            expert_ids,
            expert_starts,
            expert_counts,
        )
"""
    if old_mm1 in source:
        source = source.replace(old_mm1, new_mm1, 1)
    elif "m_grouped_fp8_gemm_nt_contiguous_with_starts(" not in source:
        raise RuntimeError(f"Could not patch first DeepGEMM MoE call in {path}")

    old_mm2 = """        m_grouped_fp8_gemm_nt_contiguous(
            (a2q, a2q_scale), (w2, self.w2_scale), mm2_out, expert_ids
        )
"""
    new_mm2 = """        m_grouped_fp8_gemm_nt_contiguous_with_starts(
            (a2q, a2q_scale),
            (w2, self.w2_scale),
            mm2_out,
            expert_ids,
            expert_starts,
            expert_counts,
        )
"""
    if old_mm2 in source:
        source = source.replace(old_mm2, new_mm2, 1)
    elif source.count("m_grouped_fp8_gemm_nt_contiguous_with_starts(") < 2:
        raise RuntimeError(f"Could not patch second DeepGEMM MoE call in {path}")

    old_fp4_mm1 = """        m_grouped_fp8_fp4_gemm_nt_contiguous(
            (a1q, a1q_scale),
            (w1.view(torch.int8), self.w1_scale),
            mm1_out,
            expert_ids,
            recipe_a=(1, self._ACT_BLOCK_K),
            recipe_b=(1, self._WEIGHT_BLOCK_K),
        )
"""
    new_fp4_mm1 = """        m_grouped_fp8_gemm_nt_contiguous_with_starts(
            (a1q, a1q_scale),
            (w1.view(torch.int8), self.w1_scale),
            mm1_out,
            expert_ids,
            expert_starts,
            expert_counts,
            recipe_a=(1, self._ACT_BLOCK_K),
            recipe_b=(1, self._WEIGHT_BLOCK_K),
        )
"""
    if old_fp4_mm1 in source:
        source = source.replace(old_fp4_mm1, new_fp4_mm1, 1)

    old_fp4_mm2 = """        m_grouped_fp8_fp4_gemm_nt_contiguous(
            (a2q, a2q_scale),
            (w2.view(torch.int8), self.w2_scale),
            mm2_out,
            expert_ids,
            recipe_a=(1, self._ACT_BLOCK_K),
            recipe_b=(1, self._WEIGHT_BLOCK_K),
        )
"""
    new_fp4_mm2 = """        m_grouped_fp8_gemm_nt_contiguous_with_starts(
            (a2q, a2q_scale),
            (w2.view(torch.int8), self.w2_scale),
            mm2_out,
            expert_ids,
            expert_starts,
            expert_counts,
            recipe_a=(1, self._ACT_BLOCK_K),
            recipe_b=(1, self._WEIGHT_BLOCK_K),
        )
"""
    if old_fp4_mm2 in source:
        source = source.replace(old_fp4_mm2, new_fp4_mm2, 1)

    # If the container was already patched before this version, with-starts
    # calls may be present without the diagnostic shape sampler. Add the sampler
    # idempotently so enabling DG_SM120_PROFILE_MOE_SHAPES does not require a
    # pristine vLLM install.
    profile_call_replacements = [
        (
            '_dg_sm120_profile_moe_shape("fc1",',
            '        m_grouped_fp8_gemm_nt_contiguous_with_starts(\n'
            '            (a1q, a1q_scale),\n'
            '            (w1, self.w1_scale),',
            '        _dg_sm120_profile_moe_shape("fc1", a1q, w1, expert_counts)\n'
            '        m_grouped_fp8_gemm_nt_contiguous_with_starts(\n'
            '            (a1q, a1q_scale),\n'
            '            (w1, self.w1_scale),',
        ),
        (
            '_dg_sm120_profile_moe_shape("fc2",',
            '        m_grouped_fp8_gemm_nt_contiguous_with_starts(\n'
            '            (a2q, a2q_scale),\n'
            '            (w2, self.w2_scale),',
            '        _dg_sm120_profile_moe_shape("fc2", a2q, w2, expert_counts)\n'
            '        m_grouped_fp8_gemm_nt_contiguous_with_starts(\n'
            '            (a2q, a2q_scale),\n'
            '            (w2, self.w2_scale),',
        ),
        (
            '_dg_sm120_profile_moe_shape("fc1_fp4",',
            '        m_grouped_fp8_gemm_nt_contiguous_with_starts(\n'
            '            (a1q, a1q_scale),\n'
            '            (w1.view(torch.int8), self.w1_scale),',
            '        _dg_sm120_profile_moe_shape("fc1_fp4", a1q, w1, expert_counts)\n'
            '        m_grouped_fp8_gemm_nt_contiguous_with_starts(\n'
            '            (a1q, a1q_scale),\n'
            '            (w1.view(torch.int8), self.w1_scale),',
        ),
        (
            '_dg_sm120_profile_moe_shape("fc2_fp4",',
            '        m_grouped_fp8_gemm_nt_contiguous_with_starts(\n'
            '            (a2q, a2q_scale),\n'
            '            (w2.view(torch.int8), self.w2_scale),',
            '        _dg_sm120_profile_moe_shape("fc2_fp4", a2q, w2, expert_counts)\n'
            '        m_grouped_fp8_gemm_nt_contiguous_with_starts(\n'
            '            (a2q, a2q_scale),\n'
            '            (w2.view(torch.int8), self.w2_scale),',
        ),
    ]
    if install_moe_profile:
        for marker, old_call, new_call in profile_call_replacements:
            if marker not in source and old_call in source:
                source = source.replace(old_call, new_call, 1)

    path.write_text(source)


def patch_deep_gemm_moe_fused_activation_quant() -> None:
    path = Path(
        "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/"
        "layers/fused_moe/experts/deep_gemm_moe.py"
    )
    source = path.read_text()
    if "import deep_gemm\n" not in source:
        source = source.replace("import torch\n", "import torch\nimport deep_gemm\n", 1)

    marker = 'deep_gemm._C.sm120_silu_mul_quant_fp8_packed'
    if marker not in source:
        old_two_step = """        # 1. DeepGemm UE8M0: use packed per-token-group quant
        if scale_fmt == DeepGemmQuantScaleFMT.UE8M0:
            act_out = torch.empty(
                (M_sum, activation_out_dim), dtype=input.dtype, device=input.device
            )
            self.activation(activation, act_out, input)
            a2q, a2q_scale = per_token_group_quant_fp8_packed_for_deepgemm(
                act_out,
                block_k,
                out_q=output,
            )
            return a2q, a2q_scale
"""
        old_triton_fused = """        # 1. DeepGemm UE8M0: fused SiLU+mul+clamp+quant+pack
        if scale_fmt == DeepGemmQuantScaleFMT.UE8M0:
            if activation == MoEActivation.SILU:
                return fused_silu_mul_fp8_quant_packed(
                    input=input,
                    output_q=output,
                    group_size=block_k,
                )
            act_out = torch.empty(
                (M_sum, activation_out_dim), dtype=input.dtype, device=input.device
            )
            self.activation(activation, act_out, input)
            a2q, a2q_scale = per_token_group_quant_fp8_packed_for_deepgemm(
                act_out,
                block_k,
                out_q=output,
            )
            return a2q, a2q_scale
"""
        new = """        # 1. DeepGemm UE8M0: fuse SiLU+mul and packed scale quantization on SM120.
        if scale_fmt == DeepGemmQuantScaleFMT.UE8M0:
            if (
                activation == MoEActivation.SILU
                and input.is_cuda
                and input.is_contiguous()
                and output.is_contiguous()
                and input.dtype == torch.bfloat16
                and output.dtype == torch.float8_e4m3fn
                and torch.cuda.get_device_capability(input.device)[0] >= 12
                and hasattr(deep_gemm._C, "sm120_silu_mul_quant_fp8_packed")
            ):
                a2q_scale = deep_gemm._C.sm120_silu_mul_quant_fp8_packed(
                    input, output, block_k, 0.0
                )
                return output.view(M_sum, activation_out_dim), a2q_scale

            act_out = torch.empty(
                (M_sum, activation_out_dim), dtype=input.dtype, device=input.device
            )
            self.activation(activation, act_out, input)
            a2q, a2q_scale = per_token_group_quant_fp8_packed_for_deepgemm(
                act_out,
                block_k,
                out_q=output,
            )
            return a2q, a2q_scale
"""
        if old_triton_fused in source:
            source = source.replace(old_triton_fused, new, 1)
        elif old_two_step in source:
            source = source.replace(old_two_step, new, 1)
        else:
            raise RuntimeError(f"Could not patch SM120 fused act+quant in {path}")

    fp4_triton = """        if scale_fmt == DeepGemmQuantScaleFMT.UE8M0:
            assert activation == MoEActivation.SILU
            return fused_silu_mul_fp8_quant_packed(
                input=input,
                output_q=output,
                group_size=block_k,
                clamp_limit=self.gemm1_clamp_limit,
            )
"""
    fp4_sm120 = """        if scale_fmt == DeepGemmQuantScaleFMT.UE8M0:
            if (
                activation == MoEActivation.SILU
                and input.is_cuda
                and input.is_contiguous()
                and output.is_contiguous()
                and input.dtype == torch.bfloat16
                and output.dtype == torch.float8_e4m3fn
                and torch.cuda.get_device_capability(input.device)[0] >= 12
                and hasattr(deep_gemm._C, "sm120_silu_mul_quant_fp8_packed")
            ):
                a2q_scale = deep_gemm._C.sm120_silu_mul_quant_fp8_packed(
                    input, output, block_k, float(self.gemm1_clamp_limit or 0.0)
                )
                return output.view(M_sum, activation_out_dim), a2q_scale

            assert activation == MoEActivation.SILU
            return fused_silu_mul_fp8_quant_packed(
                input=input,
                output_q=output,
                group_size=block_k,
                clamp_limit=self.gemm1_clamp_limit,
            )
"""
    if fp4_triton in source:
        source = source.replace(fp4_triton, fp4_sm120, 1)
    path.write_text(source)


def patch_deep_gemm_moe_unpermute_reduce() -> None:
    path = Path(
        "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/"
        "layers/fused_moe/deep_gemm_utils.py"
    )
    source = path.read_text()
    if "import deep_gemm\n" not in source:
        source = source.replace("import torch\n", "import torch\nimport deep_gemm\n", 1)

    marker = "sm120_moe_unpermute_reduce_bf16"
    if marker in source:
        source = source.replace("output.shape[1] >= 6144", "output.shape[1] >= 512")
        path.write_text(source)
        return

    old = """def deepgemm_unpermute_and_reduce(
    a: torch.Tensor,  # Grouped gemm output
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    inv_perm: torch.Tensor,
    expert_map: torch.Tensor | None,
    output: torch.Tensor,
):
    return ep_gather(
        input_tensor=a,
        recv_topk_ids=topk_ids,
        recv_topk_weight=topk_weights,
        input_index=inv_perm,
        expert_map=expert_map,
        output_tensor=output,
    )
"""
    new = """def deepgemm_unpermute_and_reduce(
    a: torch.Tensor,  # Grouped gemm output
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    inv_perm: torch.Tensor,
    expert_map: torch.Tensor | None,
    output: torch.Tensor,
):
    if (
        a.is_cuda
        and output.is_cuda
        and a.dtype == torch.bfloat16
        and output.dtype == torch.bfloat16
        and inv_perm.dtype == torch.int32
        and (expert_map is None or expert_map.dtype == torch.int32)
        and topk_ids.ndim == 2
        and topk_weights.ndim == 2
        and inv_perm.ndim == 2
        and topk_ids.shape == topk_weights.shape == inv_perm.shape
        and topk_ids.shape[0] == output.shape[0]
        and topk_ids.shape[1] <= 32
        and output.shape[1] == a.shape[1]
        and output.shape[1] >= 512
        and a.stride(1) == 1
        and output.stride(1) == 1
        and torch.cuda.get_device_capability(output.device)[0] >= 12
        and hasattr(deep_gemm._C, "sm120_moe_unpermute_reduce_bf16")
    ):
        if expert_map is None:
            deep_gemm._C.sm120_moe_unpermute_reduce_bf16(
                a, topk_ids, topk_weights, inv_perm, output
            )
        elif hasattr(deep_gemm._C, "sm120_moe_unpermute_reduce_bf16_mapped"):
            deep_gemm._C.sm120_moe_unpermute_reduce_bf16_mapped(
                a, topk_ids, topk_weights, inv_perm, expert_map, output
            )
        else:
            return ep_gather(
                input_tensor=a,
                recv_topk_ids=topk_ids,
                recv_topk_weight=topk_weights,
                input_index=inv_perm,
                expert_map=expert_map,
                output_tensor=output,
            )
        return

    return ep_gather(
        input_tensor=a,
        recv_topk_ids=topk_ids,
        recv_topk_weight=topk_weights,
        input_index=inv_perm,
        expert_map=expert_map,
        output_tensor=output,
    )
"""
    if old not in source:
        raise RuntimeError(f"Could not patch SM120 MoE unpermute/reduce in {path}")
    source = source.replace(old, new, 1)
    path.write_text(source)


def patch_deepseek_mtp_local_argmax_reduction() -> None:
    """Allow vLLM's spec decoder to avoid full-vocab all-gather for MTP.

    The DeepSeek V4 MTP draft model shares the target lm_head in this image, but
    upstream DeepSeekMTP does not expose ``get_top_tokens``.  When
    ``use_local_argmax_reduction`` is enabled in the speculative config, vLLM
    therefore falls back to gathering full vocab logits before argmax.  On TP=2
    this is unnecessary for greedy draft-token selection: each rank can compute
    its local argmax and all-gather only (value, index) pairs via the existing
    LogitsProcessor helper.
    """

    path = Path(
        "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/"
        "models/deepseek_mtp.py"
    )
    source = path.read_text()
    if "def get_top_tokens(" in source:
        source_patched = True
    else:
        source_patched = False

    predictor_old = '''    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        current_step_idx = spec_step_idx % self.num_mtp_layers
        mtp_layer = self.layers[str(self.mtp_start_layer_idx + current_step_idx)]
        logits = self.logits_processor(
            mtp_layer.shared_head.head, mtp_layer.shared_head(hidden_states)
        )
        return logits
'''
    predictor_new = predictor_old + '''
    def get_top_tokens(
        self,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        current_step_idx = spec_step_idx % self.num_mtp_layers
        mtp_layer = self.layers[str(self.mtp_start_layer_idx + current_step_idx)]
        return self.logits_processor.get_top_tokens(
            mtp_layer.shared_head.head, mtp_layer.shared_head(hidden_states)
        )
'''
    if not source_patched:
        if predictor_old not in source:
            raise RuntimeError(f"Could not patch DeepSeek MTP predictor local argmax in {path}")
        source = source.replace(predictor_old, predictor_new, 1)

    mtp_old = '''    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> torch.Tensor | None:
        return self.model.compute_logits(hidden_states, spec_step_idx)
'''
    mtp_new = mtp_old + '''
    def get_top_tokens(
        self,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        return self.model.get_top_tokens(hidden_states, spec_step_idx)
'''
    if not source_patched:
        if mtp_old not in source:
            raise RuntimeError(f"Could not patch DeepSeekMTP local argmax in {path}")
        source = source.replace(mtp_old, mtp_new, 1)
        path.write_text(source)

    v4_path = Path(
        "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/"
        "models/deepseek_v4_mtp.py"
    )
    if not v4_path.exists():
        return
    source = v4_path.read_text()
    if "def get_top_tokens(" in source:
        return

    v4_predictor_old = '''    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        current_step_idx = spec_step_idx % self.num_mtp_layers
        mtp_layer = self.layers[str(self.mtp_start_layer_idx + current_step_idx)]
        # MTP forward returns the pre-hc_head residual (T, hc_mult * D); apply
        # hc_head here so logits are computed from the dense hidden state.
        hidden_states = hidden_states.view(
            -1, mtp_layer.hc_mult, mtp_layer.config.hidden_size
        )
        hidden_states = hc_head(
            hidden_states,
            mtp_layer.hc_head_fn,
            mtp_layer.hc_head_scale,
            mtp_layer.hc_head_base,
            mtp_layer.rms_norm_eps,
            mtp_layer.hc_eps,
        )
        logits = self.logits_processor(
            mtp_layer.shared_head.head, mtp_layer.shared_head(hidden_states)
        )
        return logits
'''
    v4_predictor_new = v4_predictor_old + '''
    def get_top_tokens(
        self,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        current_step_idx = spec_step_idx % self.num_mtp_layers
        mtp_layer = self.layers[str(self.mtp_start_layer_idx + current_step_idx)]
        # Match compute_logits exactly up to vocab projection, then use vLLM's
        # vocab-parallel local argmax reduction instead of full-vocab gather.
        hidden_states = hidden_states.view(
            -1, mtp_layer.hc_mult, mtp_layer.config.hidden_size
        )
        hidden_states = hc_head(
            hidden_states,
            mtp_layer.hc_head_fn,
            mtp_layer.hc_head_scale,
            mtp_layer.hc_head_base,
            mtp_layer.rms_norm_eps,
            mtp_layer.hc_eps,
        )
        return self.logits_processor.get_top_tokens(
            mtp_layer.shared_head.head, mtp_layer.shared_head(hidden_states)
        )
'''
    if v4_predictor_old not in source:
        raise RuntimeError(f"Could not patch DeepSeek V4 MTP predictor local argmax in {v4_path}")
    source = source.replace(v4_predictor_old, v4_predictor_new, 1)

    v4_mtp_old = '''    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> torch.Tensor | None:
        return self.model.compute_logits(hidden_states, spec_step_idx)
'''
    v4_mtp_new = v4_mtp_old + '''
    def get_top_tokens(
        self,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        return self.model.get_top_tokens(hidden_states, spec_step_idx)
'''
    if v4_mtp_old not in source:
        raise RuntimeError(f"Could not patch DeepSeekV4MTP local argmax in {v4_path}")
    source = source.replace(v4_mtp_old, v4_mtp_new, 1)
    v4_path.write_text(source)


def patch_deepseek_v4_greedy_spec_argmax_fastpath() -> None:
    """Avoid full-vocab target-logit materialization for safe greedy MTP decode.

    vLLM's stock rejection sampler always computes and gathers full target
    logits for both draft-verification positions and bonus-token positions.  In
    the common greedy/no-logprobs path, the sampler only needs target argmax
    token ids.  DeepSeek V4 Flash on TP=2 pays this full-vocab cost every decode
    step, so expose the target model's vocab-parallel get_top_tokens() and route
    strictly safe greedy speculative batches through an argmax-only sampler path.

    The fast path is deliberately conservative: it only runs for all-greedy
    requests with no logprobs, no penalties/bad words/allowed-token masks, and no
    active non-argmax-invariant logits processors.  MinTokens processors are
    allowed only when their current mask is empty (e.g. ignore_eos=True).
    """

    model_path = Path(
        "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/"
        "models/deepseek_v4.py"
    )
    if model_path.exists():
        source = model_path.read_text()
        if "def get_top_tokens(" not in source:
            old = '''    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits
'''
            new = old + '''
    def get_top_tokens(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.logits_processor.get_top_tokens(self.lm_head, hidden_states)
'''
            if old not in source:
                raise RuntimeError(
                    f"Could not patch DeepSeek V4 target local argmax in {model_path}"
                )
            model_path.write_text(source.replace(old, new, 1))

    sampler_path = Path(
        "/usr/local/lib/python3.12/dist-packages/vllm/v1/sample/"
        "rejection_sampler.py"
    )
    source = sampler_path.read_text()
    if "def forward_greedy_argmax(" not in source:
        anchor = '''    def _get_logprobs_tensors(
        self,
        max_num_logprobs: int,
'''
        insert = '''    def forward_greedy_argmax(
        self,
        metadata: SpecDecodeMetadata,
        target_argmax: torch.Tensor,
        bonus_token_ids: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> SamplerOutput:
        """Greedy speculative sampler when target argmax ids are precomputed.

        This is equivalent to the all-greedy branch of rejection_sample(), but it
        avoids building full target logits/probabilities.  Callers must ensure no
        logits processors that can alter greedy argmax are active.
        """
        assert metadata.max_spec_len <= MAX_SPEC_LEN
        assert sampling_metadata.all_greedy
        assert sampling_metadata.max_num_logprobs is None
        assert not getattr(self, "synthetic_mode", False)

        batch_size = len(metadata.num_draft_tokens)
        output_token_ids = torch.full(
            (batch_size, metadata.max_spec_len + 1),
            PLACEHOLDER_TOKEN_ID,
            dtype=torch.int32,
            device=target_argmax.device,
        )
        if bonus_token_ids.ndim == 1:
            bonus_token_ids = bonus_token_ids.unsqueeze(-1)
        rejection_greedy_sample_kernel[(batch_size,)](
            output_token_ids,
            metadata.cu_num_draft_tokens,
            metadata.draft_token_ids,
            target_argmax,
            bonus_token_ids.to(torch.int32),
            None,
            metadata.max_spec_len,
        )
        return SamplerOutput(
            sampled_token_ids=output_token_ids,
            logprobs_tensors=None,
        )

'''
        if anchor not in source:
            raise RuntimeError(
                f"Could not patch greedy argmax rejection sampler in {sampler_path}"
            )
        source = source.replace(anchor, insert + anchor, 1)
        sampler_path.write_text(source)

    runner_path = Path(
        "/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/"
        "gpu_model_runner.py"
    )
    source = runner_path.read_text()
    if "def _sm120_can_use_greedy_spec_argmax(" not in source:
        anchor = '''    def _bookkeeping_sync(
        self,
        scheduler_output: "SchedulerOutput",
'''
        insert = '''    def _sm120_can_use_greedy_spec_argmax(self) -> bool:
        import os

        if os.environ.get("DG_SM120_SPEC_ARGMAX_FASTPATH", "1").lower() not in (
            "1", "true", "yes", "on"
        ):
            return False
        if not hasattr(self.model, "get_top_tokens"):
            return False
        sampling_metadata = self.input_batch.sampling_metadata
        if not sampling_metadata.all_greedy:
            return False
        if sampling_metadata.max_num_logprobs is not None:
            return False
        if sampling_metadata.logprob_token_ids:
            return False
        if sampling_metadata.allowed_token_ids_mask is not None:
            return False
        if not sampling_metadata.no_penalties:
            return False
        if sampling_metadata.bad_words_token_ids:
            return False

        # Any active processor that can alter argmax makes precomputed argmax
        # unsafe.  MinTokens is allowed only when it currently masks no tokens
        # (common with ignore_eos=True).
        for proc in sampling_metadata.logitsprocs.non_argmax_invariant:
            if proc.__class__.__name__ != "MinTokensLogitsProcessor":
                return False
            logits_slice = getattr(proc, "logits_slice", None)
            if logits_slice is None:
                return False
            req_slice = logits_slice[0]
            if req_slice is not None and req_slice.numel() != 0:
                return False
        return True

'''
        if anchor not in source:
            raise RuntimeError(
                f"Could not insert SM120 greedy spec guard in {runner_path}"
            )
        source = source.replace(anchor, insert + anchor, 1)

    old_sample = '''        sampler_output = self.rejection_sampler(
            spec_decode_metadata,
            None,  # draft_probs
            logits,
            sampling_metadata,
        )
        return sampler_output
'''
    new_sample = '''        sm120_argmax_fastpath = getattr(
            self, "_sm120_spec_argmax_fastpath", None
        )
        if sm120_argmax_fastpath is not None:
            self._sm120_spec_argmax_fastpath = None
            target_argmax, bonus_token_ids = sm120_argmax_fastpath
            return self.rejection_sampler.forward_greedy_argmax(
                spec_decode_metadata,
                target_argmax,
                bonus_token_ids,
                sampling_metadata,
            )

        sampler_output = self.rejection_sampler(
            spec_decode_metadata,
            None,  # draft_probs
            logits,
            sampling_metadata,
        )
        return sampler_output
'''
    if old_sample in source and "forward_greedy_argmax(" not in source[
        source.find("def _sample(") : source.find("def _bookkeeping_sync(")
    ]:
        source = source.replace(old_sample, new_sample, 1)

    old_logits = '''                sample_hidden_states = hidden_states[logits_indices]
                logits = self.model.compute_logits(sample_hidden_states)
'''
    new_logits = '''                sample_hidden_states = hidden_states[logits_indices]
                if (
                    spec_decode_metadata is not None
                    and self._sm120_can_use_greedy_spec_argmax()
                ):
                    sm120_top_token_ids = self.model.get_top_tokens(sample_hidden_states)
                    self._sm120_spec_argmax_fastpath = (
                        sm120_top_token_ids[spec_decode_metadata.target_logits_indices],
                        sm120_top_token_ids[spec_decode_metadata.bonus_logits_indices],
                    )
                    logits = None
                else:
                    self._sm120_spec_argmax_fastpath = None
                    logits = self.model.compute_logits(sample_hidden_states)
'''
    if old_logits in source and "sm120_top_token_ids = self.model.get_top_tokens" not in source:
        source = source.replace(old_logits, new_logits, 1)
    elif "sm120_top_token_ids = self.model.get_top_tokens" not in source:
        raise RuntimeError(
            f"Could not patch SM120 target argmax fast path in {runner_path}"
        )

    runner_path.write_text(source)


patch_cuda_platform_deep_gemm_sm120()
patch_triton_mxfp4_sm120()
patch_deep_gemm_fp4_sm120()
patch_deep_gemm_e8m0_oracle_sm120()
patch_deep_gemm_mxfp4_scale_prepack_sm120()
patch_sm120_b12x_mxfp4_load_transform()
patch_block_scaled_mm()
patch_deep_gemm_wrapper_with_starts()
patch_deep_gemm_moe_permute_starts()
patch_deep_gemm_moe_with_starts()
patch_deep_gemm_moe_fused_activation_quant()
patch_deep_gemm_moe_unpermute_reduce()
patch_deepseek_mtp_local_argmax_reduction()
patch_deepseek_v4_greedy_spec_argmax_fastpath()
patch_sm120_b12x_deep_gemm_moe()
patch_deepseek_v4_attention()
patch_deepseek_v4_prefill_dynamic_compressed_workspace()
patch_deepseek_v4_combine_topk_swa_empty_indices()
patch_deepseek_v4_direct_fp8_prefill_map()
patch_flashmla_sparse_decode()
patch_flashmla_sparse_full_context_decode()
patch_flashmla_sparse_prefill()
patch_flashmla_sparse_prefill_workspace_factor()
patch_vllm_memory_breakdown_logging()
patch_sparse_topk_for_small_context()
patch_sm120_sparse_indexer_graph_safe_topk()
patch_sm120_sparse_indexer_full_context_decode()
patch_deepseek_v4_compressor_graph_native_metadata()
patch_deepseek_v4_sparse_swa_graph_native_metadata()
patch_flashmla_sparse_req_id_graph_native_metadata()
patch_deepseek_v4_sm120_compressor_overlap()
patch_deepseek_v4_cache_gather_bounds()
patch_mhc_reusable_buffers()
patch_tp_allreduce_diagnostic_bypass()
if __import__("os").environ.get("DG_SM120_INSTALL_PROFILE_PATCH", "0") == "1":
    patch_deepseek_v4_layer_profiler()
