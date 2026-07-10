#!/usr/bin/env python3
import argparse
from collections.abc import Callable

import torch

from avelang_kernels import fused_moe

try:
    import petit_kernel
    from aiter.ops.shuffle import shuffle_weight
except ImportError:
    petit_kernel = None
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
    if petit_kernel is None or shuffle_weight is None:
        raise RuntimeError("petit_kernel and aiter.ops.shuffle must be importable.")


def _validate_shape(tokens: int, dim: int, inter_dim: int) -> None:
    if tokens <= 0 or tokens % ROUTE_BLOCK_SIZE != 0:
        raise ValueError(f"tokens must be a positive multiple of {ROUTE_BLOCK_SIZE}, got {tokens}")
    for name, value in (("dim", dim), ("inter_dim", inter_dim)):
        if value <= 0 or value % SCALE_BLOCK_SIZE != 0:
            raise ValueError(
                f"{name} must be a positive multiple of {SCALE_BLOCK_SIZE}, got {value}"
            )


def _make_case(
    tokens: int, dim: int, inter_dim: int, seed: int
) -> tuple[torch.Tensor | int, ...]:
    torch.manual_seed(seed)

    input_q = (
        torch.randn((tokens, dim), dtype=torch.float32, device="cuda")
        .clamp(-6.0, 6.0)
        .to(torch.float8_e4m3fnuz)
    )
    w13_q = (
        torch.randn((1, inter_dim * 2, dim), dtype=torch.float32, device="cuda")
        .clamp(-6.0, 6.0)
        .to(torch.float8_e4m3fnuz)
    )
    w2_q = (
        torch.randn((1, dim, inter_dim), dtype=torch.float32, device="cuda")
        .clamp(-6.0, 6.0)
        .to(torch.float8_e4m3fnuz)
    )

    w13_q = shuffle_weight(w13_q, layout=(16, 16))
    w2_q = shuffle_weight(w2_q, layout=(16, 16))

    sorted_token_ids = torch.arange(tokens, dtype=torch.int32, device="cuda")
    sorted_weights = torch.sigmoid(
        torch.randn(tokens, dtype=torch.float32, device="cuda")
    )
    sorted_expert_ids = torch.zeros(
        (tokens // ROUTE_BLOCK_SIZE,), dtype=torch.int32, device="cuda"
    )
    num_valid_ids = torch.tensor([tokens, tokens], dtype=torch.int32, device="cuda")

    input_scale = (
        torch.rand(
            (tokens, dim // SCALE_BLOCK_SIZE), dtype=torch.float32, device="cuda"
        )
        * 0.25
        + 0.01
    )
    fc1_scale = (
        torch.rand(
            (1, 2 * (inter_dim // SCALE_BLOCK_SIZE) * (dim // SCALE_BLOCK_SIZE)),
            dtype=torch.float32,
            device="cuda",
        )
        * 0.25
        + 0.01
    )
    fc2_scale = (
        torch.rand(
            (1, (dim // SCALE_BLOCK_SIZE) * (inter_dim // SCALE_BLOCK_SIZE)),
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
        1,
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
    warmup: int,
    iters: int,
    seed: int,
    validate: bool,
) -> None:
    _ensure_environment()
    _validate_shape(tokens, dim, inter_dim)
    args = _make_case(tokens, dim, inter_dim, seed)

    petit_ms = _benchmark(
        petit_kernel.fused_moe_fp8_blockscale_g1u1, args, warmup, iters
    )
    avelang_ms = _benchmark(
        fused_moe.fused_moe_fp8_blockscale_g1u1, args, warmup, iters
    )

    print(
        f"tokens={tokens} dim={dim} inter_dim={inter_dim} warmup={warmup} "
        f"iters={iters} seed={seed}"
    )
    print(
        f"petit_ms={petit_ms:.4f} avelang_ms={avelang_ms:.4f} "
        f"ratio={avelang_ms / petit_ms:.3f}"
    )

    if not validate:
        return

    expected = petit_kernel.fused_moe_fp8_blockscale_g1u1(*args)
    actual = fused_moe.fused_moe_fp8_blockscale_g1u1(*args)
    split_k = (inter_dim + GROUP_DIM - 1) // GROUP_DIM
    if split_k == 1:
        torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=1e-3, atol=1e-3)
        print(f"validation=max_abs_diff:{_error_metrics(actual, expected)['max_abs']:.6f}")
        return

    expected_2 = petit_kernel.fused_moe_fp8_blockscale_g1u1(*args)
    actual_2 = fused_moe.fused_moe_fp8_blockscale_g1u1(*args)
    print(_format_metrics("petit_vs_petit", _error_metrics(expected_2, expected)))
    print(_format_metrics("avelang_vs_avelang", _error_metrics(actual_2, actual)))
    print(_format_metrics("avelang_vs_petit", _error_metrics(actual, expected)))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark AveLang vs petit-kernel fused_moe_fp8_blockscale_g1u1. "
            "This measures public Python wrapper calls, including output allocation."
        )
    )
    parser.add_argument("--tokens", type=int, required=True)
    parser.add_argument("--dim", type=int, required=True)
    parser.add_argument("--inter-dim", type=int, required=True)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--validate", action="store_true", default=False)
    args = parser.parse_args()

    run_benchmark(
        tokens=args.tokens,
        dim=args.dim,
        inter_dim=args.inter_dim,
        warmup=args.warmup,
        iters=args.iters,
        seed=args.seed,
        validate=args.validate,
    )


if __name__ == "__main__":
    main()
