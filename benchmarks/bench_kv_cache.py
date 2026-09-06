"""Measure decode throughput with vs without a KV cache.

Usage: uv run python benchmarks/bench_kv_cache.py [--prompt 128] [--new 64]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from perf_lab.bench import BenchReport, pick_device, save  # noqa: E402
from perf_lab.kv_cache import KVCache  # noqa: E402
from perf_lab.model import GPTConfig, TinyGPT, generate  # noqa: E402

RESULTS = Path(__file__).parent / "results"


@torch.no_grad()
def timed_generate(model, idx, n, use_cache):
    t0 = time.perf_counter()
    out = generate(model, idx, max_new_tokens=n, use_cache=use_cache)
    return time.perf_counter() - t0, out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", type=int, nargs="+", default=[128, 512, 1024],
                    help="prompt lengths to sweep (new tokens fixed)")
    ap.add_argument("--new", type=int, default=128)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=8)
    args = ap.parse_args()

    device = pick_device()
    torch.manual_seed(0)
    max_prompt = max(args.prompts)
    cfg = GPTConfig(d_model=args.d_model, n_heads=args.heads, n_layers=args.layers,
                    block_size=max_prompt + args.new + 8)
    model = TinyGPT(cfg).to(device).eval()

    report = BenchReport(title=f"Decode with/without KV cache (new={args.new}, "
                               f"{cfg.n_layers}L×{cfg.d_model}d, {model.num_params()/1e6:.1f}M params)",
                         device=str(device))

    for prompt in args.prompts:
        idx = torch.randint(0, cfg.vocab_size, (1, prompt), device=device)
        t_nc, out_nc = timed_generate(model, idx, args.new, use_cache=False)
        t_c, out_c = timed_generate(model, idx, args.new, use_cache=True)
        assert torch.equal(out_nc, out_c), "cache changed greedy outputs"
        report.add(f"prompt={prompt} no cache", t_nc, "s", f"{args.new / t_nc:.1f} tok/s")
        report.add(f"prompt={prompt} KV cache", t_c, "s", f"{args.new / t_c:.1f} tok/s")
        report.add(f"prompt={prompt} speedup", t_nc / t_c, "x", "no-cache / cache")

    cache = KVCache.init_for(model, max_len=cfg.block_size)
    report.add("cache memory @block_size", cache.memory_bytes() / 1024, "KiB",
               "2*n_layers*B*H*len*hd*bytes")

    out = save(report, RESULTS, "kv_cache")
    print(f"written: {out}\n{report.to_markdown()}")


if __name__ == "__main__":
    main()
