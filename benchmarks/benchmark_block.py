from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Callable

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from local_conv_attention import LocalTransformerBlock2d, ReferenceLocalTransformerBlock2d


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def measure(
    fn: Callable[[], None],
    *,
    device: torch.device,
    warmup: int,
    repeat: int,
) -> float:
    for _ in range(warmup):
        fn()
    synchronize(device)

    start = time.perf_counter()
    for _ in range(repeat):
        fn()
    synchronize(device)
    return (time.perf_counter() - start) / repeat


def measure_backward(
    build_loss: Callable[[], torch.Tensor],
    *,
    device: torch.device,
    warmup: int,
    repeat: int,
) -> float:
    for _ in range(warmup):
        loss = build_loss()
        loss.backward()
    synchronize(device)

    start = time.perf_counter()
    for _ in range(repeat):
        loss = build_loss()
        loss.backward()
    synchronize(device)
    return (time.perf_counter() - start) / repeat


def maybe_peak_memory(device: torch.device, fn: Callable[[], None]) -> int | None:
    if device.type != "cuda":
        return None
    torch.cuda.reset_peak_memory_stats(device)
    fn()
    synchronize(device)
    return int(torch.cuda.max_memory_allocated(device))


def maybe_compile_module(
    module: torch.nn.Module,
    *,
    enabled: bool,
    mode: str | None,
) -> torch.nn.Module:
    if not enabled:
        return module
    if not hasattr(torch, "compile"):
        raise RuntimeError("torch.compile is not available in this PyTorch build.")
    return torch.compile(module, mode=mode)


def maybe_channels_last(x: torch.Tensor, *, enabled: bool) -> torch.Tensor:
    if enabled and x.dim() == 4:
        return x.contiguous(memory_format=torch.channels_last)
    return x


def gradient_metrics(forward_seconds: float, backward_seconds: float, forward_backward_seconds: float) -> tuple[float, float]:
    backward_over_forward = backward_seconds / forward_seconds if forward_seconds > 0 else float("inf")
    forward_share = forward_seconds / forward_backward_seconds if forward_backward_seconds > 0 else 0.0
    return backward_over_forward, forward_share


def shorten_error(exc: BaseException) -> str:
    return " ".join(str(exc).split())[:160]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Wrap benchmarked modules with torch.compile.",
    )
    parser.add_argument(
        "--compile-mode",
        default="default",
        choices=["default", "reduce-overhead", "max-autotune"],
        help="torch.compile mode when --compile is enabled.",
    )
    parser.add_argument(
        "--channels-last",
        action="store_true",
        help="Feed 4D benchmark tensors in channels_last memory format.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks") / "block_results.csv",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = torch.float32
    args.output.parent.mkdir(parents=True, exist_ok=True)

    shapes = [
        (1, 32, 16, 1, 3),
        (2, 64, 16, 4, 5),
        (1, 64, 32, 4, 3),
        (1, 128, 32, 8, 5),
    ]

    implementations = {
        "optimized_block": lambda **kwargs: LocalTransformerBlock2d(
            **kwargs,
            implementation="optimized",
        ),
        "shift_block_reference": lambda **kwargs: LocalTransformerBlock2d(
            **kwargs,
            implementation="shift",
        ),
        "flattened_reference_block": ReferenceLocalTransformerBlock2d,
    }

    rows: list[dict[str, object]] = []
    print(
        "implementation        shape                         forward      backward     fwd+bwd     bwd/fwd   peak_mem"
    )
    print("-" * 116)

    for batch, dim, size, heads, window in shapes:
        for name, factory in implementations.items():
            if name == "flattened_reference_block" and size * size > 1024:
                continue

            shape_label = f"B={batch},C={dim},H=W={size},heads={heads},M={window}"
            try:
                block = factory(dim=dim, num_heads=heads, window_size=window).to(device=device, dtype=dtype)
                block = maybe_compile_module(
                    block,
                    enabled=args.compile,
                    mode=args.compile_mode,
                )
                x = maybe_channels_last(
                    torch.randn(batch, dim, size, size, device=device, dtype=dtype),
                    enabled=args.channels_last,
                )

                def forward_only() -> None:
                    with torch.no_grad():
                        block(x)

                def forward_backward() -> None:
                    x_req = maybe_channels_last(x.detach().clone().requires_grad_(True), enabled=args.channels_last)
                    out = block(x_req)
                    out.square().mean().backward()

                def build_loss() -> torch.Tensor:
                    x_req = maybe_channels_last(x.detach().clone().requires_grad_(True), enabled=args.channels_last)
                    out = block(x_req)
                    return out.square().mean()

                fwd = measure(forward_only, device=device, warmup=args.warmup, repeat=args.repeat)
                backward_seconds = measure_backward(
                    build_loss,
                    device=device,
                    warmup=max(1, args.warmup // 2),
                    repeat=max(3, args.repeat // 2),
                )
                fwd_bwd = measure(
                    forward_backward,
                    device=device,
                    warmup=max(1, args.warmup // 2),
                    repeat=max(3, args.repeat // 2),
                )
                backward_over_forward, forward_share = gradient_metrics(fwd, backward_seconds, fwd_bwd)
                peak = maybe_peak_memory(device, forward_backward)

                print(
                    f"{name:22s} {shape_label:28s} {fwd * 1e3:8.3f} ms {backward_seconds * 1e3:8.3f} ms "
                    f"{fwd_bwd * 1e3:8.3f} ms {backward_over_forward:7.2f}x "
                    f"{('-' if peak is None else f'{peak / (1024 ** 2):7.1f} MB')}"
                )
                rows.append(
                    {
                        "implementation": name,
                        "batch": batch,
                        "dim": dim,
                        "height": size,
                        "width": size,
                        "heads": heads,
                        "window_size": window,
                        "forward_seconds": fwd,
                        "backward_seconds_estimate": backward_seconds,
                        "forward_backward_seconds": fwd_bwd,
                        "backward_over_forward_ratio": backward_over_forward,
                        "forward_share_of_train_step": forward_share,
                        "peak_memory_bytes": peak if peak is not None else "",
                        "device": str(device),
                        "dtype": str(dtype),
                        "compiled": args.compile,
                        "compile_mode": args.compile_mode if args.compile else "",
                        "channels_last": args.channels_last,
                        "error": "",
                    }
                )
            except Exception as exc:  # pragma: no cover - benchmark resilience path.
                error_text = shorten_error(exc)
                print(f"{name:22s} {shape_label:28s} FAILED       FAILED       FAILED       FAILED   {error_text}")
                rows.append(
                    {
                        "implementation": name,
                        "batch": batch,
                        "dim": dim,
                        "height": size,
                        "width": size,
                        "heads": heads,
                        "window_size": window,
                        "forward_seconds": "",
                        "backward_seconds_estimate": "",
                        "forward_backward_seconds": "",
                        "backward_over_forward_ratio": "",
                        "forward_share_of_train_step": "",
                        "peak_memory_bytes": "",
                        "device": str(device),
                        "dtype": str(dtype),
                        "compiled": args.compile,
                        "compile_mode": args.compile_mode if args.compile else "",
                        "channels_last": args.channels_last,
                        "error": error_text,
                    }
                )

    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved CSV to {args.output}")


if __name__ == "__main__":
    main()
