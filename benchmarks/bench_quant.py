"""Benchmark quantized matmul: accuracy cost vs weight footprint / speed.

Usage: uv run python benchmarks/bench_quant.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from perf_lab.bench import BenchReport, bench, pick_device, save  # noqa: E402
from perf_lab.quant import (QuantizedLinear, dequantize_nf4, quantize_int8,  # noqa: E402
                            quantize_nf4)

RESULTS = Path(__file__).parent / "results"


def main() -> None:
    device = pick_device()
    dtype = torch.float16 if device.type in ("cuda", "mps") else torch.float32
    torch.manual_seed(0)
    OUT, IN = 2048, 2048
    w = (torch.randn(OUT, IN) * 0.05).to(device, dtype)
    x = torch.randn(64, IN, device=device, dtype=dtype)

    report = BenchReport(title=f"Quantization: fp{16 if dtype==torch.float16 else 32} vs int8 vs nf4 "
                               f"({OUT}x{IN} linear, batch 64)", device=str(device))

    linear = torch.nn.Linear(IN, OUT).to(device, dtype)
    with torch.no_grad():
        linear.weight.copy_(w)
    ref = linear(x)

    fp32_bytes = w.numel() * 4

    t_fp = bench(lambda: linear(x), warmup=3, iters=10)
    report.add("fp matmul", t_fp * 1e3, "ms", f"weight {fp32_bytes/1024:.0f} KiB")

    for kind in ("int8", "nf4"):
        ql = QuantizedLinear(linear, kind=kind).to(device)
        with torch.no_grad():
            out = ql(x)
        rel = ((out - ref).norm() / ref.norm()).item()
        t = bench(lambda: ql(x), warmup=3, iters=10)
        report.add(f"{kind} matmul", t * 1e3, "ms",
                   f"weight {ql.weight_footprint_bytes()/1024:.0f} KiB, rel err {rel*100:.2f}%")
        report.add(f"{kind} compression", fp32_bytes / ql.weight_footprint_bytes(), "x", "vs fp32 weight")

    # raw quantize/dequantize cost of nf4 (per-block scales dominate)
    t_q = bench(lambda: quantize_nf4(w), warmup=1, iters=3)
    report.add("nf4 quantize", t_q * 1e3, "ms", "one-time, per weight tensor")

    out = save(report, RESULTS, "quant")
    print(f"written: {out}\n{report.to_markdown()}")


if __name__ == "__main__":
    main()
