"""Small benchmark harness: warmup + timed repetitions, JSON + markdown export."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def bench(fn, *args, warmup: int = 3, iters: int = 10, synchronize: bool = True, **kwargs) -> float:
    """Return median wall-clock seconds per call."""
    for _ in range(warmup):
        fn(*args, **kwargs)
    if synchronize:
        _sync()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn(*args, **kwargs)
        _sync()
        times.append(time.perf_counter() - t0)
    times.sort()
    return times[len(times) // 2]


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        torch.mps.synchronize()


@dataclass
class BenchReport:
    title: str
    device: str
    rows: list[dict] = field(default_factory=list)  # {name, value, unit, note}

    def add(self, name: str, value: float, unit: str, note: str = "") -> None:
        self.rows.append({"name": name, "value": round(value, 4), "unit": unit, "note": note})

    def to_json(self) -> str:
        return json.dumps({"title": self.title, "device": self.device, "rows": self.rows},
                          ensure_ascii=False, indent=2)

    def to_markdown(self) -> str:
        lines = [f"### {self.title}", f"device: `{self.device}`", "",
                 "| 指标 | 值 | 单位 | 备注 |", "|---|---|---|---|"]
        for r in self.rows:
            lines.append(f"| {r['name']} | {r['value']} | {r['unit']} | {r['note']} |")
        return "\n".join(lines)


def save(report: BenchReport, out_dir: str | Path, slug: str) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{slug}.json").write_text(report.to_json())
    with (out / f"{slug}.md").open("a") as f:
        f.write(report.to_markdown() + "\n\n")
    return out / f"{slug}.json"
