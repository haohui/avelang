"""FP8 block-scale g1u1 MoE regression tests adopted from petit-kernel.

The production entry point consumes AITER's pre-shuffled FP8 weights.  These
tests keep that contract but implement the small amount of packing locally so
the reference coverage does not depend on AITER being installed.
"""

import math

import pytest
import torch
import torch.nn.functional as F

from avelang.testing import has_rocm
from avelang_kernels.fused_moe import fused_moe_fp8_blockscale_g1u1


BLOCK = 128
PER_ELEMENT_ATOL = 5e-2
PER_ELEMENT_RTOL = 5e-2
REPLAY_LIKE_SENSITIVE_ATOL = 2.05e-2


def _fp8_dtype_or_skip() -> torch.dtype:
    if not torch.cuda.is_available():
        pytest.skip("requires an AMD GPU")
    arch = getattr(torch.cuda.get_device_properties("cuda"), "gcnArchName", "")
    if arch.startswith(("gfx950", "gfx1200", "gfx1201")) and hasattr(torch, "float8_e4m3fn"):
        return torch.float8_e4m3fn
    if hasattr(torch, "float8_e4m3fnuz"):
        return torch.float8_e4m3fnuz
    if hasattr(torch, "float8_e4m3fn"):
        return torch.float8_e4m3fn
    pytest.skip("a torch E4M3 FP8 dtype is required")


def _shuffle_weight(x: torch.Tensor) -> torch.Tensor:
    """Match aiter.ops.shuffle.shuffle_weight(x, layout=(16, 16))."""
    if x.shape[0] == 0:
        return x
    rows, cols = x.shape[-2:]
    assert rows % 16 == 0 and cols % 32 == 0
    return (
        x.view(-1, rows // 16, 16, cols // 32, 2, 16)
        .permute(0, 1, 3, 4, 2, 5)
        .contiguous()
        .view_as(x)
    )


def _quantize_input(x: torch.Tensor, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    tokens, dim = x.shape
    fp8_max = torch.finfo(dtype).max
    blocks = x.view(tokens, dim // BLOCK, BLOCK)
    scale = blocks.abs().amax(dim=-1).clamp_min(1e-6) / fp8_max
    return (blocks / scale.unsqueeze(-1)).to(dtype).reshape_as(x), scale


def _quantize_weight(x: torch.Tensor, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    experts, rows, cols = x.shape
    fp8_max = torch.finfo(dtype).max
    blocks = x.view(experts, rows // BLOCK, BLOCK, cols // BLOCK, BLOCK).permute(0, 1, 3, 2, 4)
    scale = blocks.abs().amax(dim=(-2, -1)).clamp_min(1e-6) / fp8_max
    q = (blocks / scale.unsqueeze(-1).unsqueeze(-1)).to(dtype)
    return q.permute(0, 1, 3, 2, 4).reshape_as(x), scale.reshape(experts, -1)


def _dequantize_input(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    tokens, dim = q.shape
    return (q.float().view(tokens, dim // BLOCK, BLOCK) * scale.float().unsqueeze(-1)).reshape_as(q)


def _dequantize_weight(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    experts, rows, cols = q.shape
    blocks = q.float().view(experts, rows // BLOCK, BLOCK, cols // BLOCK, BLOCK).permute(0, 1, 3, 2, 4)
    return (blocks * scale.float().view(experts, rows // BLOCK, cols // BLOCK, 1, 1)).permute(0, 1, 3, 2, 4).reshape_as(q)


def _build_sorted_routing(
    topk_ids: torch.Tensor, topk_weights: torch.Tensor, experts: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build the padded routing buffers consumed by the one-stage kernel."""
    tokens, topk = topk_ids.shape
    ids = topk_ids.cpu().to(torch.int32).reshape(-1)
    weights = topk_weights.cpu().to(torch.float32).reshape(-1)
    token_ids = torch.arange(tokens, dtype=torch.int32).unsqueeze(1).expand(tokens, topk).reshape(-1)
    slots = torch.arange(topk, dtype=torch.int32).unsqueeze(0).expand(tokens, topk).reshape(-1)
    order = torch.argsort(ids, stable=True)
    ids, weights, token_ids, slots = ids[order], weights[order], token_ids[order], slots[order]

    packed_ids: list[torch.Tensor] = []
    packed_weights: list[torch.Tensor] = []
    packed_experts: list[int] = []
    cursor = 0
    for expert in range(experts):
        count = int((ids == expert).sum())
        if count == 0:
            continue
        end = cursor + count
        routes = (slots[cursor:end] << 24) | token_ids[cursor:end]
        padded = math.ceil(count / 32) * 32
        packed_ids.append(torch.cat((routes, torch.full((padded - count,), tokens, dtype=torch.int32))))
        packed_weights.append(torch.cat((weights[cursor:end], torch.zeros(padded - count, dtype=torch.float32))))
        packed_experts.extend([expert] * (padded // 32))
        cursor = end

    device = topk_ids.device
    if not packed_ids:
        return (
            torch.empty(0, dtype=torch.int32, device=device),
            torch.empty(0, dtype=torch.float32, device=device),
            torch.empty(0, dtype=torch.int32, device=device),
            torch.tensor([0, tokens], dtype=torch.int32, device=device),
        )
    sorted_ids = torch.cat(packed_ids).to(device)
    return (
        sorted_ids,
        torch.cat(packed_weights).to(device),
        torch.tensor(packed_experts, dtype=torch.int32, device=device),
        torch.tensor([sorted_ids.numel(), tokens], dtype=torch.int32, device=device),
    )


def _reference(case: dict[str, torch.Tensor | int]) -> torch.Tensor:
    input_q = case["input_q"]
    w13_q = case["w13_q"]
    w2_q = case["w2_q"]
    assert isinstance(input_q, torch.Tensor) and isinstance(w13_q, torch.Tensor) and isinstance(w2_q, torch.Tensor)
    x = _dequantize_input(input_q, case["input_scale"])  # type: ignore[arg-type]
    tokens, dim = x.shape
    inter_dim = w2_q.shape[-1]
    result = torch.zeros((tokens, dim), device=x.device, dtype=torch.float32)
    w13 = _dequantize_weight(w13_q, case["fc1_scale"])  # type: ignore[arg-type]
    w2 = _dequantize_weight(w2_q, case["fc2_scale"])  # type: ignore[arg-type]
    topk_ids = case["topk_ids"]
    topk_weights = case["topk_weights"]
    assert isinstance(topk_ids, torch.Tensor) and isinstance(topk_weights, torch.Tensor)
    for expert in range(w2_q.shape[0]):
        token_ids, slot_ids = torch.where(topk_ids == expert)
        if token_ids.numel() == 0:
            continue
        stage1 = x[token_ids] @ w13[expert].T
        hidden = F.silu(stage1[:, :inter_dim]) * stage1[:, inter_dim:]
        result.index_add_(0, token_ids, (hidden @ w2[expert].T) * topk_weights[token_ids, slot_ids].float().unsqueeze(1))
    return result.to(torch.bfloat16)


def _build_case(tokens: int, dim: int, inter_dim: int, experts: int, topk: int, *, harsh: bool = False) -> dict[str, torch.Tensor | int]:
    dtype = _fp8_dtype_or_skip()
    device = torch.device("cuda")
    weight_std = 40.0 if harsh else 1.0 / 60.0
    input_f = torch.randn((tokens, dim), device=device)
    w13_f = torch.randn((experts, 2 * inter_dim, dim), device=device) * weight_std
    w2_f = torch.randn((experts, dim, inter_dim), device=device) * weight_std
    input_q, input_scale = _quantize_input(input_f, dtype)
    w13_q, fc1_scale = _quantize_weight(w13_f, dtype)
    w2_q, fc2_scale = _quantize_weight(w2_f, dtype)
    if harsh:
        # Mirror petit-kernel's ultra-harsh input: FP8 values and scales are
        # independently sampled, with tiny block scales plus a structured
        # stage-1 pattern that stresses DPP scale distribution.
        input_scale = (torch.randn_like(input_scale) * 2e-4 + 1e-3).clamp_min(1e-8)
        fc1_scale = (torch.randn_like(fc1_scale) * 2e-4 + 1e-3).clamp_min(1e-8)
        fc2_scale = (torch.randn_like(fc2_scale) * 2e-4 + 1e-3).clamp_min(1e-8)
        input_scale.mul_(torch.linspace(1.0, 4.0, tokens, device=device).unsqueeze(1))
        fc1_scale.mul_(torch.linspace(1.0, 8.0, fc1_scale.shape[1], device=device).unsqueeze(0))
    token_idx = torch.arange(tokens, device=device).unsqueeze(1)
    slot_idx = torch.arange(topk, device=device).unsqueeze(0)
    topk_ids = ((token_idx * 7 + slot_idx * 5) % experts).to(torch.int32)
    topk_weights = torch.rand((tokens, topk), device=device)
    topk_weights /= topk_weights.sum(dim=-1, keepdim=True)
    sorted_token_ids, sorted_weights, sorted_expert_ids, num_valid_ids = _build_sorted_routing(topk_ids, topk_weights, experts)
    return {
        "input_q": input_q,
        "w13_q": w13_q,
        "w2_q": w2_q,
        "w13_kernel": _shuffle_weight(w13_q),
        "w2_kernel": _shuffle_weight(w2_q),
        "input_scale": input_scale,
        "fc1_scale": fc1_scale,
        "fc2_scale": fc2_scale,
        "topk_ids": topk_ids,
        "topk_weights": topk_weights,
        "sorted_token_ids": sorted_token_ids,
        "sorted_weights": sorted_weights,
        "sorted_expert_ids": sorted_expert_ids,
        "num_valid_ids": num_valid_ids,
        "topk": topk,
    }


def _build_replay_like_case() -> dict[str, torch.Tensor | int]:
    """Port petit-kernel's replay-like sensitive FP8 distribution verbatim."""
    tokens, dim, inter_dim, experts, topk = 2, 7168, 2048, 32, 8
    dtype = _fp8_dtype_or_skip()
    device = torch.device("cuda")
    token_ramp = torch.linspace(-1.0, 1.0, tokens, device=device).unsqueeze(1)
    dim_ramp = torch.linspace(-1.0, 1.0, dim, device=device).unsqueeze(0)
    input_f = torch.randn((tokens, dim), device=device) + 0.1 + 0.35 * token_ramp + 0.15 * dim_ramp
    spike_mask = torch.rand((tokens, dim), device=device) < 0.002
    input_f += torch.randn((tokens, dim), device=device) * 16.0 * spike_mask
    input_q = input_f.to(dtype)
    w13_q = (1.2 * torch.randn((experts, 2 * inter_dim, dim), device=device)).to(dtype)
    w2_q = torch.randn((experts, dim, inter_dim), device=device).to(dtype)
    input_scale = (torch.randn((tokens, dim // BLOCK), device=device) * 0.02 + 0.10).clamp_min(1e-8)
    fc1_scale = (torch.randn((experts, 2 * (inter_dim // BLOCK) * (dim // BLOCK)), device=device) * 0.004 + 0.020).clamp_min(1e-8)
    fc2_scale = (torch.randn((experts, (dim // BLOCK) * (inter_dim // BLOCK)), device=device) * 0.004 + 0.025).clamp_min(1e-8)
    topk_ids = torch.tensor(
        ((9, 18, 29, 30, 25, 4, 11, 3), (15, 12, 20, 23, 31, 7, 6, 21)),
        dtype=torch.int32,
        device=device,
    )
    token_idx = torch.arange(tokens, dtype=torch.float32, device=device).unsqueeze(1)
    slot_idx = torch.arange(topk, dtype=torch.float32, device=device).unsqueeze(0)
    topk_weights = 2.0 * (0.25 + 0.025 * ((token_idx + 3 * slot_idx) % 9))
    sorted_token_ids, sorted_weights, sorted_expert_ids, num_valid_ids = _build_sorted_routing(topk_ids, topk_weights, experts)
    return {
        "input_q": input_q,
        "w13_q": w13_q,
        "w2_q": w2_q,
        "w13_kernel": _shuffle_weight(w13_q),
        "w2_kernel": _shuffle_weight(w2_q),
        "input_scale": input_scale,
        "fc1_scale": fc1_scale,
        "fc2_scale": fc2_scale,
        "topk_ids": topk_ids,
        "topk_weights": topk_weights,
        "sorted_token_ids": sorted_token_ids,
        "sorted_weights": sorted_weights,
        "sorted_expert_ids": sorted_expert_ids,
        "num_valid_ids": num_valid_ids,
        "topk": topk,
    }


def _run(
    case: dict[str, torch.Tensor | int],
    out: torch.Tensor | None = None,
    num_persistent_tgs: int = 0,
) -> torch.Tensor:
    w13_kernel = case.get("w13_kernel")
    w2_kernel = case.get("w2_kernel")
    if w13_kernel is None:
        w13_kernel = _shuffle_weight(case["w13_q"])  # type: ignore[arg-type]
    if w2_kernel is None:
        w2_kernel = _shuffle_weight(case["w2_q"])  # type: ignore[arg-type]
    return fused_moe_fp8_blockscale_g1u1(
        case["input_q"],  # type: ignore[arg-type]
        w13_kernel,  # type: ignore[arg-type]
        w2_kernel,  # type: ignore[arg-type]
        case["sorted_token_ids"],  # type: ignore[arg-type]
        case["sorted_weights"],  # type: ignore[arg-type]
        case["sorted_expert_ids"],  # type: ignore[arg-type]
        case["num_valid_ids"],  # type: ignore[arg-type]
        case["topk"],  # type: ignore[arg-type]
        case["input_scale"],  # type: ignore[arg-type]
        case["fc1_scale"],  # type: ignore[arg-type]
        case["fc2_scale"],  # type: ignore[arg-type]
        num_persistent_tgs=num_persistent_tgs,
        out=out,
    )


@pytest.mark.skipif(not has_rocm(), reason="requires ROCm/HIP with an AMD GPU")
def test_fused_moe_blockscale_fp8_matches_reference():
    torch.manual_seed(42)
    # DeepSeek V3.2-style routed expert configuration from petit-kernel.
    case = _build_case(tokens=8, dim=7168, inter_dim=2048, experts=33, topk=9)
    torch.testing.assert_close(_run(case), _reference(case), rtol=PER_ELEMENT_RTOL, atol=PER_ELEMENT_ATOL)


@pytest.mark.skipif(not has_rocm(), reason="requires ROCm/HIP with an AMD GPU")
def test_fused_moe_blockscale_fp8_replay_like_sensitive_matches_reference():
    # Match the upstream replay dimensions and expert count; the fixed seed and
    # sparse multi-expert routing catch W2 advancement/layout regressions.
    torch.manual_seed(1234)
    case = _build_replay_like_case()
    torch.testing.assert_close(_run(case), _reference(case), rtol=0.0, atol=REPLAY_LIKE_SENSITIVE_ATOL)


@pytest.mark.skipif(not has_rocm(), reason="requires ROCm/HIP with an AMD GPU")
def test_fused_moe_blockscale_fp8_harsh_scale_matches_reference():
    torch.manual_seed(7)
    case = _build_case(tokens=32, dim=512, inter_dim=512, experts=1, topk=1, harsh=True)
    torch.testing.assert_close(_run(case), _reference(case), rtol=5e-2, atol=1.0)


@pytest.mark.skipif(not has_rocm(), reason="requires ROCm/HIP with an AMD GPU")
def test_fused_moe_persistent_routes_match_reference():
    torch.manual_seed(17)
    case = _build_case(tokens=64, dim=512, inter_dim=512, experts=1, topk=1)
    torch.testing.assert_close(
        _run(case, num_persistent_tgs=2),
        _reference(case),
        rtol=PER_ELEMENT_RTOL,
        atol=PER_ELEMENT_ATOL,
    )


@pytest.mark.skipif(not has_rocm(), reason="requires ROCm/HIP with an AMD GPU")
def test_fused_moe_validates_preallocated_output():
    case = _build_case(tokens=32, dim=512, inter_dim=512, experts=1, topk=1)
    with pytest.raises(ValueError, match="out must be bfloat16"):
        _run(case, torch.empty((32, 512), dtype=torch.float32, device="cuda"))
    with pytest.raises(ValueError, match=r"out must be \[tokens, dim\]"):
        _run(case, torch.empty((32, 640), dtype=torch.bfloat16, device="cuda"))
    with pytest.raises(ValueError, match="out must be contiguous"):
        _run(case, torch.empty((512, 32), dtype=torch.bfloat16, device="cuda").T)


@pytest.mark.skipif(not has_rocm(), reason="requires ROCm/HIP with an AMD GPU")
def test_fused_moe_supports_preallocated_output_and_cuda_graph():
    torch.manual_seed(42)
    case = _build_case(tokens=32, dim=512, inter_dim=512, experts=2, topk=2)
    eager_out = torch.empty((32, 512), dtype=torch.bfloat16, device="cuda")
    assert _run(case, eager_out).data_ptr() == eager_out.data_ptr()
    expected = eager_out.clone()
    graph_out = torch.empty_like(eager_out)
    _run(case, graph_out)  # Compile and warm up before graph capture.
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = _run(case, graph_out)
    graph.replay()
    torch.cuda.synchronize()
    assert result.data_ptr() == graph_out.data_ptr()
    torch.testing.assert_close(graph_out, expected, rtol=PER_ELEMENT_RTOL, atol=PER_ELEMENT_ATOL)


@pytest.mark.skipif(not has_rocm(), reason="requires ROCm/HIP with an AMD GPU")
def test_fused_moe_zero_route_supports_cuda_graph():
    dtype = _fp8_dtype_or_skip()
    tokens, dim, inter_dim = 64, 7168, 2048
    case: dict[str, torch.Tensor | int] = {
        "input_q": torch.zeros((tokens, dim), dtype=dtype, device="cuda"),
        "w13_q": torch.empty((0, 2 * inter_dim, dim), dtype=dtype, device="cuda"),
        "w2_q": torch.empty((0, dim, inter_dim), dtype=dtype, device="cuda"),
        "sorted_token_ids": torch.empty(0, dtype=torch.int32, device="cuda"),
        "sorted_weights": torch.empty(0, dtype=torch.float32, device="cuda"),
        "sorted_expert_ids": torch.empty(0, dtype=torch.int32, device="cuda"),
        "num_valid_ids": torch.tensor([0, tokens], dtype=torch.int32, device="cuda"),
        "topk": 8,
        "input_scale": torch.ones((tokens, dim // BLOCK), dtype=torch.float32, device="cuda"),
        "fc1_scale": torch.empty((0, 2 * (inter_dim // BLOCK) * (dim // BLOCK)), dtype=torch.float32, device="cuda"),
        "fc2_scale": torch.empty((0, (dim // BLOCK) * (inter_dim // BLOCK)), dtype=torch.float32, device="cuda"),
    }
    out = torch.zeros((tokens, dim), dtype=torch.bfloat16, device="cuda")
    _run(case, out)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = _run(case, out)
    graph.replay()
    torch.cuda.synchronize()
    assert result.data_ptr() == out.data_ptr()
    torch.testing.assert_close(out, torch.zeros_like(out))


@pytest.mark.skipif(not has_rocm(), reason="requires ROCm/HIP with an AMD GPU")
def test_fused_moe_matches_aiter_when_available():
    aiter = pytest.importorskip("aiter")
    torch.manual_seed(42)
    case = _build_case(tokens=32, dim=512, inter_dim=512, experts=2, topk=2)
    expected = _reference(case)
    actual = _run(case)
    aiter_out = torch.zeros_like(actual)
    input_scale = torch.empty_like(case["input_scale"])  # type: ignore[arg-type]
    aiter.partial_transpose(input_scale, case["input_scale"], num_rows=torch.tensor([32], dtype=torch.int32, device="cuda"))  # type: ignore[arg-type]
    aiter.fmoe_fp8_blockscale_g1u1(
        aiter_out,
        case["input_q"],
        case["w13_kernel"], case["w2_kernel"],  # type: ignore[arg-type]
        case["sorted_token_ids"], case["sorted_weights"], case["sorted_expert_ids"], case["num_valid_ids"], case["topk"],  # type: ignore[arg-type]
        input_scale, case["fc1_scale"], case["fc2_scale"], "", BLOCK, BLOCK, None,  # type: ignore[arg-type]
    )
    torch.testing.assert_close(actual, expected, rtol=PER_ELEMENT_RTOL, atol=PER_ELEMENT_ATOL)
    torch.testing.assert_close(actual, aiter_out, rtol=PER_ELEMENT_RTOL, atol=PER_ELEMENT_ATOL)
