"""Weight-only quantization: INT8 dynamic per-channel and NF4 (QLoRA-style).

Both are implemented in plain PyTorch so they run on any backend and can be
*verified* against float reference — the same numerics used by production
inference stacks (llama.cpp / bitsandbytes), at textbook scale.
"""

from __future__ import annotations

import torch

# ---------------------------------------------------------------- INT8 ------

def quantize_int8(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-output-channel int8 quantization.

    Returns (int8_weight, scale) with weight ≈ int8_weight * scale[:, None].
    """
    scale = weight.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / 127.0
    q = torch.round(weight / scale).clamp(-127, 127).to(torch.int8)
    return q, scale.squeeze(1)


def int8_linear(x: torch.Tensor, q_weight: torch.Tensor, scale: torch.Tensor,
                bias: torch.Tensor | None = None) -> torch.Tensor:
    """Matmul with an int8 weight: dequant-on-the-fly to x.dtype, then mm.

    Real engines fuse this into an int8 GEMM with int32 accumulate; the point
    here is the numerics contract and the 4x smaller weight footprint.
    """
    w = q_weight.to(x.dtype) * scale.to(x.dtype)[:, None]
    return torch.nn.functional.linear(x, w, bias)


def quant_error_int8(weight: torch.Tensor) -> float:
    """Max abs element-wise error of the round-trip."""
    q, s = quantize_int8(weight)
    return (q.to(weight.dtype) * s[:, None] - weight).abs().max().item()


# ----------------------------------------------------------------- NF4 ------

# NF4 lookup table (normalized float 4-bit), transcribed from the QLoRA paper
# (Dettmers et al., 2023) — 16 quantization levels, information-theoretically
# optimal for weights distributed ~N(0, 1) after per-block scaling.
NF4_LEVELS = [
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
    0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0,
]


def _nf4_lut(dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return torch.tensor(NF4_LEVELS, dtype=dtype, device=device)


def quantize_nf4(weight: torch.Tensor, block_size: int = 64) -> dict[str, torch.Tensor]:
    """Block-wise NF4 quantization, mirroring QLoRA's storage format.

    Weights are reshaped into blocks of ``block_size``; each block gets an
    fp32 absmax scale. Every weight is then snapped to the nearest NF4 level
    and stored as a uint4 *index* into the 16-entry table.
    """
    flat = weight.reshape(-1)
    pad = (-flat.numel()) % block_size
    padded = torch.nn.functional.pad(flat, (0, pad))
    blocks = padded.view(-1, block_size)
    absmax = blocks.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
    normalized = blocks / absmax  # in [-1, 1]
    lut = _nf4_lut(weight.dtype, weight.device)
    # nearest level by |distance|
    idx = (normalized.unsqueeze(-1) - lut).abs().argmin(dim=-1).to(torch.uint8)  # (n_blocks, bs)
    return {"idx": idx, "absmax": absmax.squeeze(1), "block_size": block_size,
            "orig_shape": torch.tensor(weight.shape), "pad": pad}


def dequantize_nf4(q: dict[str, torch.Tensor], dtype: torch.dtype | None = None) -> torch.Tensor:
    lut = _nf4_lut(torch.float32, q["idx"].device).to(dtype or torch.float32)
    blocks = lut[q["idx"].long()] * q["absmax"].unsqueeze(1)
    flat = blocks.reshape(-1)
    if q["pad"]:
        flat = flat[: flat.numel() - int(q["pad"])]
    return flat.reshape(tuple(int(s) for s in q["orig_shape"]))


def nf4_compression_ratio() -> float:
    """fp32 -> uint4 + fp32-absmax-per-64-blocks."""
    return 32.0 / (4.0 + 32.0 / 64.0)


class QuantizedLinear(torch.nn.Module):
    """Drop-in nn.Linear backed by either int8 or nf4 storage."""

    def __init__(self, linear: torch.nn.Linear, kind: str = "int8"):
        super().__init__()
        assert kind in ("int8", "nf4")
        self.kind = kind
        self.in_features, self.out_features = linear.in_features, linear.out_features
        self.bias = None if linear.bias is None else torch.nn.Parameter(linear.bias.data.clone())
        if kind == "int8":
            qw, s = quantize_int8(linear.weight.data)
            self.register_buffer("q_weight", qw)
            self.register_buffer("scale", s)
        else:
            q = quantize_nf4(linear.weight.data)
            self.register_buffer("idx", q["idx"])
            self.register_buffer("absmax", q["absmax"])
            self._block_size = q["block_size"]
            self._orig_shape = tuple(int(s) for s in q["orig_shape"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.kind == "int8":
            return int8_linear(x, self.q_weight, self.scale,
                               None if self.bias is None else self.bias)
        w = dequantize_nf4(
            {"idx": self.idx, "absmax": self.absmax, "block_size": self._block_size,
             "orig_shape": torch.tensor(self._orig_shape), "pad": 0},
            dtype=x.dtype,
        )
        return torch.nn.functional.linear(x, w, None if self.bias is None else self.bias)

    def weight_footprint_bytes(self) -> int:
        if self.kind == "int8":
            return self.q_weight.numel() * self.q_weight.element_size()
        return self.idx.numel() + self.absmax.numel() * 4
