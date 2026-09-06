"""Speculative decoding (Leviathan et al. 2023 / Chen et al. 2023).

A small *draft* model autoregressively proposes ``gamma`` tokens; the large
*target* model scores all of them in ONE forward pass; the longest matching
prefix is accepted plus one bonus token from the target. Greedy verification
makes the output **token-for-token identical** to the target model's own
greedy decoding — the speedup is free correctness-wise, and comes from
amortizing target forwards when the draft model agrees often.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .kv_cache import KVCache
from .model import TinyGPT


@dataclass
class SpecStats:
    steps: int = 0          # target forward passes
    accepted: int = 0       # draft tokens accepted
    proposed: int = 0       # draft tokens proposed
    emitted: int = 0        # tokens in the final output

    @property
    def acceptance_rate(self) -> float:
        return self.accepted / self.proposed if self.proposed else 0.0

    @property
    def tokens_per_target_forward(self) -> float:
        return self.emitted / self.steps if self.steps else 0.0


@torch.no_grad()
def speculative_generate(
    target: TinyGPT,
    draft: TinyGPT,
    idx: torch.Tensor,
    max_new_tokens: int,
    gamma: int = 4,
) -> tuple[torch.Tensor, SpecStats]:
    """Greedy speculative decode. Output is identical to target-only greedy."""
    assert target.cfg.vocab_size == draft.cfg.vocab_size, "models must share a vocab"
    assert target.cfg.block_size == draft.cfg.block_size
    stats = SpecStats()

    t_cache = KVCache.init_for(target)
    d_cache = KVCache.init_for(draft)

    logits = target(idx, t_cache)  # prefill
    draft(idx, d_cache)
    cur = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # first token, straight from target
    seq = torch.cat([idx, cur], dim=1)  # running full sequence (source of truth)
    stats.emitted += 1

    while stats.emitted < max_new_tokens and t_cache.seq_len + gamma + 1 < target.cfg.block_size:
        # 0) heal any draft-cache drift: target's prefix is the source of truth
        if d_cache.seq_len < t_cache.seq_len:
            draft(seq[:, d_cache.seq_len:t_cache.seq_len], d_cache)
        d_len = d_cache.seq_len

        # 1) draft proposes gamma tokens autoregressively
        proposals = []
        nxt = cur
        for _ in range(gamma):
            d_logits = draft(nxt, d_cache)
            nxt = d_logits[:, -1, :].argmax(dim=-1, keepdim=True)
            proposals.append(nxt)

        # 2) target scores [cur, p1..p_gamma] in ONE forward (positions L..L+gamma)
        verify_in = torch.cat([cur] + proposals, dim=1)
        t_logits = target(verify_in, t_cache, defer_commit=True)
        choices = t_logits.argmax(dim=-1)  # target's greedy token for each next position

        # 3) accept the longest prefix where the proposal matches target's choice
        n_accept = 0
        for i in range(gamma):
            if bool(torch.all(proposals[i] == choices[:, i : i + 1])):
                n_accept += 1
            else:
                break
        # emit accepted proposals + the target's next token (bonus or correction)
        emitted = choices[:, : n_accept + 1]
        if stats.emitted + emitted.shape[1] > max_new_tokens:
            emitted = emitted[:, : max_new_tokens - stats.emitted]
        stats.emitted += emitted.shape[1]
        stats.accepted += n_accept
        stats.proposed += gamma
        stats.steps += 1

        # 4) commit/rollback both caches to the accepted prefix
        t_cache.commit_upto(n_accept + 1)          # inputs cur + p1..p_n_accept
        d_cache.rollback(min(d_len + n_accept + 1, d_cache.seq_len))
        seq = torch.cat([seq, emitted], dim=1)
        cur = emitted[:, -1:]

    return seq[:, idx.shape[1]:], stats
