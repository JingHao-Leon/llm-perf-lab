"""GPU-only Triton kernels.

These import Triton lazily and are skipped by the test suite / benchmarks on
machines without CUDA. Keep them small but *useful*: fused RMSNorm and fused
softmax are the canonical first two kernels of any inference stack.

Run on a CUDA box:
    python benchmarks/bench_triton.py
"""

from __future__ import annotations

import torch


def triton_available() -> bool:
    try:
        import triton  # noqa: F401
        return torch.cuda.is_available()
    except ImportError:
        return False


if triton_available():
    import triton
    import triton.language as tl

    @triton.jit
    def _rmsnorm_kernel(X, Y, W, stride_row, N, EPS: tl.constexpr, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        x_ptr = X + row * stride_row
        y_ptr = Y + row * stride_row
        _sum = tl.zeros([BLOCK], dtype=tl.float32)
        for off in range(0, N, BLOCK):
            cols = off + tl.arange(0, BLOCK)
            mask = cols < N
            x = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)
            _sum += x * x
        ms = tl.sum(_sum, axis=0) / N
        rs = 1.0 / tl.sqrt(ms + EPS)
        for off in range(0, N, BLOCK):
            cols = off + tl.arange(0, BLOCK)
            mask = cols < N
            x = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)
            w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
            tl.store(y_ptr + cols, (x * rs * w).to(Y.dtype.element_ty), mask=mask)

    def rmsnorm_triton(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
        """Fused RMSNorm: one kernel, single read/write of x."""
        y = torch.empty_like(x)
        *_, N = x.shape
        x2 = x.contiguous().view(-1, N)
        y2 = y.view(-1, N)
        BLOCK = triton.next_power_of_2(N)
        _rmsnorm_kernel[(x2.shape[0],)](
            x2, y2, weight, x2.stride(0), N, EPS=eps, BLOCK=BLOCK, num_warps=8,
        )
        return y

    @triton.jit
    def _softmax_kernel(X, Y, stride_row, N, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        x_ptr = X + row * stride_row
        y_ptr = Y + row * stride_row
        m = -float("inf")
        for off in range(0, N, BLOCK):
            cols = off + tl.arange(0, BLOCK)
            x = tl.load(x_ptr + cols, mask=cols < N, other=float("-inf"))
            m = tl.maximum(m, tl.max(x, axis=0))
        s = tl.zeros([BLOCK], dtype=tl.float32)
        for off in range(0, N, BLOCK):
            cols = off + tl.arange(0, BLOCK)
            x = tl.load(x_ptr + cols, mask=cols < N, other=float("-inf")).to(tl.float32)
            e = tl.exp(x - m)
            s += e
        denom = tl.sum(s, axis=0)
        for off in range(0, N, BLOCK):
            cols = off + tl.arange(0, BLOCK)
            x = tl.load(x_ptr + cols, mask=cols < N, other=float("-inf")).to(tl.float32)
            tl.store(y_ptr + cols, (tl.exp(x - m) / denom).to(Y.dtype.element_ty), mask=cols < N)

    def softmax_triton(x: torch.Tensor) -> torch.Tensor:
        """Numerically-stable row-wise softmax, fused into one kernel."""
        y = torch.empty_like(x)
        N = x.shape[-1]
        x2 = x.contiguous().view(-1, N)
        y2 = y.view(-1, N)
        BLOCK = triton.next_power_of_2(N)
        _softmax_kernel[(x2.shape[0],)](x2, y2, x2.stride(0), N, BLOCK=BLOCK)
        return y
