"""Benchmark attention backends on the local device.

Usage: uv run python benchmarks/bench_attention.py [--seq 1024 2048 4096]
Writes JSON + markdown into benchmarks/results/.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from perf_lab.attention import BACKENDS  # noqa: E402
from perf_lab.bench import BenchReport, bench, pick_device, save  # noqa: E402

RESULTS = Path(__file__).parent / "results"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, nargs="+", default=[512, 1024, 2048])
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--head-dim", type=int, default=64)
    args = ap.parse_args()

    device = pick_device()
    dtype = torch.float16 if device.type in ("cuda", "mps") else torch.float32
    report = BenchReport(title=f"Attention backends (B={args.batch}, H={args.heads}, hd={args.head_dim}, dtype={dtype})",
                         device=str(device))

    for T in args.seq:
        g = torch.Generator(device="cpu").manual_seed(0)
        q = torch.randn(args.batch, args.heads, T, args.head_dim, generator=g).to(device, dtype)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        for name, fn in BACKENDS.items():
            if name == "naive" and T > 4096:
                continue  # O(T^2) memory; skip very long seqs on small devices
            try:
                t = bench(lambda: fn(q, k, v), warmup=2, iters=5)
                speed = t * 1e3
                report.add(f"T={T} {name}", speed, "ms", "median of 5")
            except torch.cuda.OutOfMemoryError:
                report.add(f"T={T} {name}", -1, "ms", "OOM")

    out = save(report, RESULTS, "attention")
    print(f"written: {out}\n{report.to_markdown()}")


if __name__ == "__main__":
    main()
