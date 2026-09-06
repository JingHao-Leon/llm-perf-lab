"""Pre-allocated KV cache, the single most effective decode-time optimization.

The cache stores keys/values per layer in pre-allocated tensors and hands the
model a *view* of only the valid prefix, so appending is O(T_new) instead of
recomputing the whole prefix every step (O(T^2) total).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:  # avoid a circular import at runtime
    from .model import TinyGPT


class KVCache:
    """Per-layer key/value cache with append + commit semantics.

    ``append`` makes the new tokens visible to the current forward pass;
    ``commit`` advances the cached length afterwards. Splitting the two lets
    attention see the freshly appended k/v in the same step.
    """

    def __init__(self, n_layers: int, n_heads: int, head_dim: int, max_len: int,
                 dtype: torch.dtype, device: torch.device, batch_size: int = 1):
        self.max_len = max_len
        shape = (n_layers, batch_size, n_heads, max_len, head_dim)
        self.k = torch.zeros(*shape, dtype=dtype, device=device)
        self.v = torch.zeros(*shape, dtype=dtype, device=device)
        self._len = 0  # committed length
        self._pending = 0

    @classmethod
    def init_for(cls, model: "TinyGPT", max_len: int | None = None,
                 dtype: torch.dtype | None = None, device: torch.device | None = None,
                 batch_size: int = 1) -> "KVCache":
        cfg = model.cfg
        p = next(model.parameters())
        return cls(
            n_layers=cfg.n_layers,
            n_heads=cfg.n_heads,
            head_dim=cfg.head_dim,
            max_len=max_len or cfg.block_size,
            dtype=dtype or p.dtype,
            device=device or p.device,
            batch_size=batch_size,
        )

    def append(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Write the new k/v and return the full history (valid prefix + new)."""
        T = k.shape[2]
        self.k[layer_idx, :, :, self._len:self._len + T, :] = k.to(self.k.dtype)
        self.v[layer_idx, :, :, self._len:self._len + T, :] = v.to(self.v.dtype)
        self._pending = T
        end = self._len + T
        return self.k[layer_idx, :, :, :end], self.v[layer_idx, :, :, :end]

    def commit(self, n_tokens: int) -> None:
        assert n_tokens == self._pending, "commit must match the appended token count"
        self._len += n_tokens
        self._pending = 0

    @property
    def seq_len(self) -> int:
        return self._len

    def clear(self) -> None:
        self._len = 0
        self._pending = 0

    def memory_bytes(self) -> int:
        return self.k.numel() * self.k.element_size() * 2  # k + v
