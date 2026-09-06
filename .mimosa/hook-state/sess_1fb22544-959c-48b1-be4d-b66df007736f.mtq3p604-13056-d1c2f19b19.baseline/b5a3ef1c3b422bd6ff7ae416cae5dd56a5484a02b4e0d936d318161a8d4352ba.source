"""Attention implementations with different memory/time trade-offs.

- ``naive_attention``: materializes the full (T, T) score matrix. The
  textbook reference — O(T^2) memory, the reason long contexts OOM.
- ``chunked_attention``: query-blocked attention that never materializes
  more than (chunk, T) scores — the bridge to streaming/flash attention.
- ``sdpa_attention``: torch.nn.functional.scaled_dot_product_attention,
  which dispatches to FlashAttention / memory-efficient kernels on CUDA
  and to a fused path on MPS.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def naive_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                    causal: bool = True) -> torch.Tensor:
    """q/k/v: (B, H, T, hd). Returns (B, H, T, hd)."""
    T = q.shape[2]
    att = (q @ k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
    if causal:
        mask = torch.ones(T, T, dtype=torch.bool, device=q.device).tril()
        att = att.masked_fill(~mask, float("-inf"))
    return F.softmax(att, dim=-1) @ v


def chunked_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                      causal: bool = True, chunk: int = 128) -> torch.Tensor:
    """Compute attention one query-chunk at a time to bound peak memory."""
    B, H, T, hd = q.shape
    out = torch.empty_like(q)
    for start in range(0, T, chunk):
        end = min(start + chunk, T)
        qc = q[:, :, start:end]  # (B, H, c, hd)
        att = (qc @ k.transpose(-2, -1)) / math.sqrt(hd)  # (B, H, c, T)
        if causal:
            # row i of the chunk is global query start+i; it may attend keys <= start+i
            mask = torch.ones(end - start, T, dtype=torch.bool, device=q.device).tril(diagonal=start)
            att = att.masked_fill(~mask, float("-inf"))
        out[:, :, start:end] = F.softmax(att, dim=-1) @ v
    return out


def sdpa_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                   causal: bool = True) -> torch.Tensor:
    return F.scaled_dot_product_attention(q, k, v, is_causal=causal)


BACKENDS = {"naive": naive_attention, "chunked": chunked_attention, "sdpa": sdpa_attention}
