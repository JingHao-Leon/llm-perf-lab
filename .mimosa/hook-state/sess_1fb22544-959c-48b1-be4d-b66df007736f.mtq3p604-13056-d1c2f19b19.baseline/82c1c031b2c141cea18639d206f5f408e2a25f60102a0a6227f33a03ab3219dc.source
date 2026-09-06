import pytest
import torch

from perf_lab.attention import BACKENDS, chunked_attention, naive_attention, sdpa_attention


@pytest.fixture(scope="module")
def qkv():
    g = torch.Generator().manual_seed(0)
    B, H, T, hd = 2, 4, 300, 32
    q = torch.randn(B, H, T, hd, generator=g)
    k = torch.randn(B, H, T, hd, generator=g)
    v = torch.randn(B, H, T, hd, generator=g)
    return q, k, v


def test_all_attention_backends_agree(qkv):
    q, k, v = qkv
    ref = naive_attention(q, k, v)
    chunk_sizes = [64, 128, 256]
    for cs in chunk_sizes:
        out = chunked_attention(q, k, v, chunk=cs)
        torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(sdpa_attention(q, k, v), ref, rtol=1e-4, atol=1e-5)


def test_non_causal_matches_sdpa(qkv):
    q, k, v = qkv
    ref = torch.nn.functional.softmax(q @ k.transpose(-2, -1) / q.shape[-1] ** 0.5, dim=-1) @ v
    torch.testing.assert_close(naive_attention(q, k, v, causal=False), ref,
                               rtol=1e-4, atol=1e-5)


def test_backend_registry():
    assert set(BACKENDS) == {"naive", "chunked", "sdpa"}
    for fn in BACKENDS.values():
        assert callable(fn)
