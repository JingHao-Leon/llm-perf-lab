import math

import torch

from perf_lab.quant import (QuantizedLinear, dequantize_nf4, int8_linear,
                            nf4_compression_ratio, quant_error_int8,
                            quantize_int8, quantize_nf4)


def test_int8_roundtrip_error_bounded():
    w = torch.randn(256, 128) * 0.05
    err = quant_error_int8(w)
    # per-channel symmetric int8: max err <= scale/2 = absmax/254
    bound = w.abs().amax(dim=1).max().item() / 254
    assert err <= bound + 1e-6
    assert err / w.abs().max().item() < 0.01  # <1% relative error


def test_int8_linear_close_to_fp():
    g = torch.Generator().manual_seed(0)
    w = torch.randn(128, 64, generator=g) * 0.05
    x = torch.randn(8, 64, generator=g)
    q, s = quantize_int8(w)
    out_q = int8_linear(x, q, s)
    out_ref = torch.nn.functional.linear(x, w)
    rel = (out_q - out_ref).norm() / out_ref.norm()
    assert rel < 0.02


def test_int8_storage_is_one_byte():
    w = torch.randn(64, 32)
    q, _ = quantize_int8(w)
    assert q.dtype == torch.int8
    assert q.element_size() == 1
    assert w.element_size() == 4


def test_nf4_roundtrip_error():
    g = torch.Generator().manual_seed(0)
    w = torch.randn(128, 96, generator=g) * 0.1  # block_size=64 divides 96*128
    q = quantize_nf4(w, block_size=64)
    w2 = dequantize_nf4(q, dtype=torch.float32)
    rel = (w2 - w).norm() / w.norm()
    assert rel < 0.15  # 4-bit: typical NF4 relative error ~5-10%
    assert q["idx"].dtype == torch.uint8
    assert int(q["idx"].max()) <= 15


def test_nf4_handles_non_divisible_shapes():
    w = torch.randn(37, 53)
    q = quantize_nf4(w, block_size=64)
    w2 = dequantize_nf4(q, dtype=torch.float32)
    assert w2.shape == w.shape
    rel = (w2 - w).norm() / w.norm()
    assert rel < 0.2


def test_nf4_compression_ratio():
    r = nf4_compression_ratio()
    assert math.isclose(r, 32 / 4.5, rel_tol=1e-3)  # ≈ 7.1x smaller than fp32


def test_quantized_linear_dropin():
    g = torch.Generator().manual_seed(0)
    linear = torch.nn.Linear(64, 32)
    with torch.no_grad():
        linear.weight.copy_(torch.randn(32, 64, generator=g) * 0.05)
    x = torch.randn(4, 64, generator=g)
    ref = linear(x)
    for kind, tol in (("int8", 0.02), ("nf4", 0.25)):
        ql = QuantizedLinear(linear, kind=kind)
        out = ql(x)
        rel = (out - ref).norm() / ref.norm()
        assert rel < tol, kind
        assert ql.weight_footprint_bytes() < 32 * 64 * 4  # smaller than fp32 weight
