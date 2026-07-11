#!/usr/bin/env python3
import argparse
import math
from collections.abc import Callable

import torch

from avelang_kernels import fused_moe

try:
    import petit_kernel
except ImportError:
    petit_kernel = None

try:
    from aiter.ops.shuffle import shuffle_weight
except ImportError:
    shuffle_weight = None


ROUTE_BLOCK_SIZE = 32
SCALE_BLOCK_SIZE = 128
GROUP_DIM = 256


def _ensure_environment() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")
    if not hasattr(torch.version, "hip") or torch.version.hip is None:
        raise RuntimeError("HIP is not available; this benchmark requires ROCm.")
    if not hasattr(torch, "float8_e4m3fnuz"):
        raise RuntimeError("torch.float8_e4m3fnuz is required.")


def _validate_shape(tokens: int, dim: int, inter_dim: int, experts: int, topk: int) -> None:
    if tokens <= 0 or tokens % ROUTE_BLOCK_SIZE != 0:
        raise ValueError(f"tokens must be a positive multiple of {ROUTE_BLOCK_SIZE}, got {tokens}")
    for name, value in (("dim", dim), ("inter_dim", inter_dim)):
        if value <= 0 or value % SCALE_BLOCK_SIZE != 0:
            raise ValueError(
                f"{name} must be a positive multiple of {SCALE_BLOCK_SIZE}, got {value}"
            )
    if experts <= 0:
        raise ValueError(f"experts must be positive, got {experts}")
    if topk <= 0 or topk > experts:
        raise ValueError(f"topk must be in [1, experts], got topk={topk}, experts={experts}")


def _shuffle_weight_fallback(x: torch.Tensor) -> torch.Tensor:
    rows, cols = x.shape[-2:]
    if rows % 16 != 0 or cols % 32 != 0:
        raise ValueError(f"weight rows must divide by 16 and cols by 32, got {tuple(x.shape)}")
    return (
        x.view(-1, rows // 16, 16, cols // 32, 2, 16)
        .permute(0, 1, 3, 4, 2, 5)
        .contiguous()
        .view_as(x)
    )


def _shuffle_weight(x: torch.Tensor) -> torch.Tensor:
    if shuffle_weight is not None:
        return shuffle_weight(x, layout=(16, 16))
    return _shuffle_weight_fallback(x)


def _build_sorted_routing(
    tokens: int, experts: int, topk: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    token_idx = torch.arange(tokens, dtype=torch.int64, device="cuda").unsqueeze(1)
    slot_idx = torch.arange(topk, dtype=torch.int64, device="cuda").unsqueeze(0)
    topk_ids = ((token_idx * 7 + slot_idx * 5) % experts).to(torch.int32)
    topk_weights = torch.rand((tokens, topk), dtype=torch.float32, device="cuda")
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)

    ids = topk_ids.cpu().reshape(-1)
    weights = topk_weights.cpu().reshape(-1)
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
        padded = math.ceil(count / ROUTE_BLOCK_SIZE) * ROUTE_BLOCK_SIZE
        packed_ids.append(
            torch.cat((routes, torch.full((padded - count,), tokens, dtype=torch.int32)))
        )
        packed_weights.append(
            torch.cat((weights[cursor:end], torch.zeros(padded - count, dtype=torch.float32)))
        )
        packed_experts.extend([expert] * (padded // ROUTE_BLOCK_SIZE))
        cursor = end

    sorted_token_ids = torch.cat(packed_ids).to("cuda")
    sorted_weights = torch.cat(packed_weights).to("cuda")
    sorted_expert_ids = torch.tensor(packed_experts, dtype=torch.int32, device="cuda")
    num_valid_ids = torch.tensor([sorted_token_ids.numel(), tokens], dtype=torch.int32, device="cuda")
    return sorted_token_ids, sorted_weights, sorted_expert_ids, num_valid_ids


def _make_case(
    tokens: int, dim: int, inter_dim: int, experts: int, topk: int, seed: int
) -> tuple[torch.Tensor | int, ...]:
    torch.manual_seed(seed)

    input_q = (
        torch.randn((tokens, dim), dtype=torch.float32, device="cuda")
        .clamp(-6.0, 6.0)
        .to(torch.float8_e4m3fnuz)
    )
    w13_q = (
        torch.randn((experts, inter_dim * 2, dim), dtype=torch.float32, device="cuda")
        .clamp(-6.0, 6.0)
        .to(torch.float8_e4m3fnuz)
    )
    w2_q = (
        torch.randn((experts, dim, inter_dim), dtype=torch.float32, device="cuda")
        .clamp(-6.0, 6.0)
        .to(torch.float8_e4m3fnuz)
    )

    w13_q = _shuffle_weight(w13_q)
    w2_q = _shuffle_weight(w2_q)

    sorted_token_ids, sorted_weights, sorted_expert_ids, num_valid_ids = _build_sorted_routing(
        tokens, experts, topk
    )

    input_scale = (
        torch.rand(
            (tokens, dim // SCALE_BLOCK_SIZE), dtype=torch.float32, device="cuda"
        )
        * 0.25
        + 0.01
    )
    fc1_scale = (
        torch.rand(
            (experts, 2 * (inter_dim // SCALE_BLOCK_SIZE) * (dim // SCALE_BLOCK_SIZE)),
            dtype=torch.float32,
            device="cuda",
        )
        * 0.25
        + 0.01
    )
    fc2_scale = (
        torch.rand(
            (experts, (dim // SCALE_BLOCK_SIZE) * (inter_dim // SCALE_BLOCK_SIZE)),
            dtype=torch.float32,
            device="cuda",
        )
        * 0.25
        + 0.01
    )

    return (
        input_q,
        w13_q,
        w2_q,
        sorted_token_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        topk,
        input_scale,
        fc1_scale,
        fc2_scale,
    )


def _benchmark(
    fn: Callable[..., torch.Tensor],
    args: tuple[torch.Tensor | int, ...],
    warmup: int,
    iters: int,
) -> float:
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()

    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)
    start_evt.record()
    for _ in range(iters):
        fn(*args)
    end_evt.record()
    torch.cuda.synchronize()
    return start_evt.elapsed_time(end_evt) / iters


def _error_metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    diff = (actual.float() - expected.float()).abs()
    return {
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
        "p99_abs": float(torch.quantile(diff.flatten(), 0.99).item()),
    }


def _format_metrics(prefix: str, values: dict[str, float]) -> str:
    ordered = " ".join(f"{key}={value:.6f}" for key, value in values.items())
    return f"{prefix} {ordered}"


def run_benchmark(
    *,
    tokens: int,
    dim: int,
    inter_dim: int,
    experts: int,
    topk: int,
    warmup: int,
    iters: int,
    seed: int,
    validate: bool,
    compare_petit: bool,
) -> None:
    _ensure_environment()
    _validate_shape(tokens, dim, inter_dim, experts, topk)
    if compare_petit and petit_kernel is None:
        raise RuntimeError("--compare-petit requires petit_kernel to be importable.")
    args = _make_case(tokens, dim, inter_dim, experts, topk, seed)

    avelang_ms = _benchmark(
        fused_moe.fused_moe_fp8_blockscale_g1u1, args, warmup, iters
    )
    flops = 6.0 * tokens * topk * dim * inter_dim
    avelang_tflops = flops / (avelang_ms * 1e-3) / 1e12

    print(
        f"tokens={tokens} dim={dim} inter_dim={inter_dim} experts={experts} "
        f"topk={topk} warmup={warmup} iters={iters} seed={seed}"
    )
    if compare_petit:
        assert petit_kernel is not None
        petit_ms = _benchmark(
            petit_kernel.fused_moe_fp8_blockscale_g1u1, args, warmup, iters
        )
        petit_tflops = flops / (petit_ms * 1e-3) / 1e12
        print(
            f"petit_ms={petit_ms:.4f} petit_tflops={petit_tflops:.3f} "
            f"avelang_ms={avelang_ms:.4f} avelang_tflops={avelang_tflops:.3f} "
            f"ratio={avelang_ms / petit_ms:.3f}"
        )
    else:
        print(f"avelang_ms={avelang_ms:.4f} avelang_tflops={avelang_tflops:.3f}")

    if not validate:
        return

    expected = (
        petit_kernel.fused_moe_fp8_blockscale_g1u1(*args)
        if compare_petit and petit_kernel is not None
        else fused_moe.fused_moe_fp8_blockscale_g1u1(*args)
    )
    actual = fused_moe.fused_moe_fp8_blockscale_g1u1(*args)
    split_k = (inter_dim + GROUP_DIM - 1) // GROUP_DIM
    if compare_petit and split_k == 1:
        torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=1e-3, atol=1e-3)
        print(f"validation=max_abs_diff:{_error_metrics(actual, expected)['max_abs']:.6f}")
        return

    expected_2 = (
        petit_kernel.fused_moe_fp8_blockscale_g1u1(*args)
        if compare_petit and petit_kernel is not None
        else fused_moe.fused_moe_fp8_blockscale_g1u1(*args)
    )
    actual_2 = fused_moe.fused_moe_fp8_blockscale_g1u1(*args)
    baseline_name = "petit" if compare_petit else "avelang_baseline"
    print(_format_metrics(f"{baseline_name}_vs_{baseline_name}", _error_metrics(expected_2, expected)))
    print(_format_metrics("avelang_vs_avelang", _error_metrics(actual_2, actual)))
    print(_format_metrics(f"avelang_vs_{baseline_name}", _error_metrics(actual, expected)))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark AveLang fused_moe_fp8_blockscale_g1u1. This measures "
            "public Python wrapper calls, including output allocation. Use "
            "--compare-petit to also benchmark petit-kernel when installed."
        )
    )
    parser.add_argument("--tokens", type=int, required=True)
    parser.add_argument("--dim", type=int, required=True)
    parser.add_argument("--inter-dim", type=int, required=True)
    parser.add_argument("--experts", type=int, default=1)
    parser.add_argument("--topk", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--validate", action="store_true", default=False)
    parser.add_argument(
        "--compare-petit",
        action="store_true",
        default=False,
        help="Also benchmark/validate petit_kernel when it is installed.",
    )
    args = parser.parse_args()

    run_benchmark(
        tokens=args.tokens,
        dim=args.dim,
        inter_dim=args.inter_dim,
        experts=args.experts,
        topk=args.topk,
        warmup=args.warmup,
        iters=args.iters,
        seed=args.seed,
        validate=args.validate,
        compare_petit=args.compare_petit,
    )


if __name__ == "__main__":
    main()
