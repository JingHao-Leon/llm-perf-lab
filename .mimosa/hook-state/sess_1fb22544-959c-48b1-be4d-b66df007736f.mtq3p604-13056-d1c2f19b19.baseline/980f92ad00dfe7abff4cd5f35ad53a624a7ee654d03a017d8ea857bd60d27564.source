"""Tiny decoder-only transformer used as the benchmark subject.

Everything is intentionally written in plain PyTorch so the same forward pass
can run with or without a KV cache, on CPU / MPS / CUDA alike. The model is
small (default ~5M params) which keeps CPU benchmarks quick while preserving
the *relative* cost structure of real LLM decoding.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .kv_cache import KVCache


@dataclass
class GPTConfig:
    vocab_size: int = 4096
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 4
    block_size: int = 512
    mlp_ratio: int = 4

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)

    def forward(self, x: torch.Tensor, cache: KVCache | None, layer_idx: int) -> torch.Tensor:
        B, T, _ = x.shape
        qkv = self.qkv(x)  # (B, T, 3*d)
        q, k, v = qkv.split(self.cfg.d_model, dim=-1)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # (B, H, T, hd)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        if cache is not None:
            k, v = cache.append(layer_idx, k, v)  # (B, H, T_past + T, hd)

        T_total = k.shape[2]
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # Causal mask over the full history; query i may attend to keys <= i.
        mask = torch.ones(T, T_total, dtype=torch.bool, device=x.device).tril(diagonal=T_total - T)
        att = att.masked_fill(~mask, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = att @ v  # (B, H, T, hd)
        y = y.transpose(1, 2).contiguous().view(B, T, self.cfg.d_model)
        return self.proj(y)


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.RMSNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.RMSNorm(cfg.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.mlp_ratio * cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.mlp_ratio * cfg.d_model, cfg.d_model),
        )

    def forward(self, x: torch.Tensor, cache: KVCache | None, layer_idx: int) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), cache, layer_idx)
        x = x + self.mlp(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(self, cfg: GPTConfig | None = None):
        super().__init__()
        self.cfg = cfg or GPTConfig()
        self.embed = nn.Embedding(self.cfg.vocab_size, self.cfg.d_model)
        self.pos_embed = nn.Embedding(self.cfg.block_size, self.cfg.d_model)
        self.blocks = nn.ModuleList(Block(self.cfg) for _ in range(self.cfg.n_layers))
        self.ln_f = nn.RMSNorm(self.cfg.d_model)
        self.head = nn.Linear(self.cfg.d_model, self.cfg.vocab_size, bias=False)
        # GPT-2 style weight tying
        self.head.weight = self.embed.weight
        self.apply(self._init)

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        cache: KVCache | None = None,
    ) -> torch.Tensor:
        """idx: (B, T) token ids. With a cache, T may be 1 for incremental decoding."""
        B, T = idx.shape
        total_len = T if cache is None else cache.seq_len + T
        assert total_len <= self.cfg.block_size, "sequence exceeds block_size"
        pos = torch.arange(total_len - T, total_len, device=idx.device)
        x = self.embed(idx) + self.pos_embed(pos)[None, :, :]
        for i, block in enumerate(self.blocks):
            x = block(x, cache, i)
        if cache is not None:
            cache.commit(T)
        return self.head(self.ln_f(x))

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


@torch.no_grad()
def generate(
    model: TinyGPT,
    idx: torch.Tensor,
    max_new_tokens: int,
    use_cache: bool = True,
    temperature: float = 0.0,
) -> torch.Tensor:
    """Greedy (temperature=0) or sampled generation, with/without KV cache."""
    cache = KVCache.init_for(model) if use_cache else None
    out = idx
    prefix, rest = idx, torch.empty(idx.shape[0], 0, dtype=idx.dtype, device=idx.device)
    for _ in range(max_new_tokens):
        logits = model(prefix if cache is not None else out, cache)
        logits = logits[:, -1, :]
        if temperature > 0:
            probs = F.softmax(logits / temperature, dim=-1)
            nxt = torch.multinomial(probs, 1)
        else:
            nxt = logits.argmax(dim=-1, keepdim=True)
        out = torch.cat([out, nxt], dim=1)
        prefix = nxt
    if cache is not None:
        cache.clear()
    return out
