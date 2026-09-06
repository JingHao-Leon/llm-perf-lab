"""Benchmark speculative decoding: draft model + target model vs target alone.

Usage: uv run python benchmarks/bench_speculative.py [--gammas 2 4 8]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from perf_lab.bench import BenchReport, pick_device, save  # noqa: E402
from perf_lab.model import GPTConfig, TinyGPT, generate  # noqa: E402
from perf_lab.speculative import speculative_generate  # noqa: E402

RESULTS = Path(__file__).parent / "results"


def train_to_count(model: TinyGPT, vocab: int, seq_len: int, steps: int = 200,
                   lr: float = 3e-3) -> None:
    """Fit next-token prediction on a counting sequence (arange % vocab).

    An untrained network's argmax is decided by initialization noise, so the
    draft (even a truncated copy) agrees ~never. After a minute of training
    the target's greedy output is a deterministic pattern a shallow draft can
    follow — mirroring how real draft/target pairs come from the same family.
    """
    import torch.nn.functional as F

    stream = (torch.arange(steps * seq_len + 1, dtype=torch.long,
                            device=next(model.parameters()).device) % vocab)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    for i in range(steps):
        x = stream[i * seq_len:(i + 1) * seq_len].unsqueeze(0)
        y = stream[i * seq_len + 1:(i + 1) * seq_len + 1].unsqueeze(0)
        loss = F.cross_entropy(model(x).reshape(-1, vocab), y.reshape(-1))
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
    model.eval()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gammas", type=int, nargs="+", default=[2, 4, 8])
    ap.add_argument("--prompt", type=int, default=256)
    ap.add_argument("--new", type=int, default=256)
    args = ap.parse_args()

    device = pick_device()
    torch.manual_seed(0)
    shared = dict(vocab_size=200, d_model=256, n_heads=8,
                  block_size=args.prompt + args.new + 16)
    target = TinyGPT(GPTConfig(n_layers=6, **shared)).to(device).eval()
    print("pre-training target to count (makes greedy output deterministic)...")
    train_to_count(target, vocab=shared["vocab_size"], seq_len=256, steps=800, lr=1e-3)
    # draft = truncation of the target (first 2 layers): a legitimately close
    # distribution — random init would give ~1/vocab acceptance and no speedup
    draft = TinyGPT(GPTConfig(n_layers=2, **shared)).to(device).eval()
    with torch.no_grad():
        draft.embed.weight.copy_(target.embed.weight)
        draft.pos_embed.weight.copy_(target.pos_embed.weight)
        for i in range(2):
            draft.blocks[i].load_state_dict(target.blocks[i].state_dict())
        draft.ln_f.load_state_dict(target.ln_f.state_dict())
    report = BenchReport(
        title=f"Speculative decoding (draft=target truncated 2L×256d, "
              f"target 6L×256d {target.num_params()/1e6:.1f}M, prompt={args.prompt}, new={args.new})",
        device=str(device))
    idx = torch.randint(0, shared["vocab_size"], (1, args.prompt), device=device)

    with torch.no_grad():
        t0 = time.perf_counter()
        ref = generate(target, idx, max_new_tokens=args.new, use_cache=True)
        t_plain = time.perf_counter() - t0
    report.add("target greedy (KV cache)", t_plain, "s", f"{args.new/t_plain:.0f} tok/s")

    for gamma in args.gammas:
        with torch.no_grad():
            t0 = time.perf_counter()
            out, stats = speculative_generate(target, draft, idx,
                                              max_new_tokens=args.new, gamma=gamma)
            t_spec = time.perf_counter() - t0
        assert torch.equal(out, ref[:, idx.shape[1]:]), "output mismatch!"
        report.add(f"speculative γ={gamma}", t_spec, "s",
                   f"{args.new/t_spec:.0f} tok/s, accept {stats.acceptance_rate:.0%}, "
                   f"{stats.tokens_per_target_forward:.2f} tok/forward")
        report.add(f"γ={gamma} speedup", t_plain / t_spec, "x")

    out = save(report, RESULTS, "speculative")
    print(f"written: {out}\n{report.to_markdown()}")


if __name__ == "__main__":
    main()
