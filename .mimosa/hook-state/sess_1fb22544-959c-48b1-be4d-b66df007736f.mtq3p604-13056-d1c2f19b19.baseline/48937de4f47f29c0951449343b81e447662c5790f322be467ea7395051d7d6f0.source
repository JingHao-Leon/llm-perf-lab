"""Benchmark Triton kernels vs eager PyTorch (CUDA only; skipped elsewhere)."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from perf_lab.bench import BenchReport, bench, save  # noqa: E402


def _load_kernels():
    """Import Triton kernels only when the platform actually supports them."""
    from triton_kernels.fused_rmsnorm import triton_available

    if not triton_available():
        return None
    from triton_kernels.fused_rmsnorm import rmsnorm_triton, softmax_triton

    return rmsnorm_triton, softmax_triton


RESULTS = Path(__file__).parent / "results"


def main() -> None:
    kernels = _load_kernels()
    if kernels is None:
        print("triton/CUDA not available on this machine — skipped. "
              "Run on a CUDA GPU: uv run python benchmarks/bench_triton.py")
        return
    rmsnorm_triton, softmax_triton = kernels
    device = torch.device("cuda")
    torch.manual_seed(0)
    M, N = 4096, 4096
    x = torch.randn(M, N, device=device, dtype=torch.float16)
    w = torch.randn(N, device=device, dtype=torch.float16)

    report = BenchReport(title=f"Triton fused kernels ({M}x{N} fp16)", device=str(device))

    def eager_rms():
        return torch.nn.functional.rms_norm(x, (N,), w)

    t_eager = bench(eager_rms)
    t_triton = bench(lambda: rmsnorm_triton(x, w))
    report.add("rmsnorm eager", t_eager * 1e3, "ms")
    report.add("rmsnorm triton", t_triton * 1e3, "ms")
    report.add("rmsnorm speedup", t_eager / t_triton, "x")

    t_eager = bench(lambda: torch.softmax(x, dim=-1))
    t_triton = bench(lambda: softmax_triton(x))
    report.add("softmax eager", t_eager * 1e3, "ms")
    report.add("softmax triton", t_triton * 1e3, "ms")
    report.add("softmax speedup", t_eager / t_triton, "x")

    # correctness
    torch.testing.assert_close(rmsnorm_triton(x, w), eager_rms(), rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(softmax_triton(x), torch.softmax(x, dim=-1), rtol=1e-2, atol=1e-2)
    report.add("correctness", 1, "pass", "triton vs eager, rtol/atol 1e-2")

    out = save(report, RESULTS, "triton")
    print(f"written: {out}\n{report.to_markdown()}")


if __name__ == "__main__":
    main()
