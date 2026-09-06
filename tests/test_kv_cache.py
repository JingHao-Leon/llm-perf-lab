import torch

from perf_lab.kv_cache import KVCache
from perf_lab.model import GPTConfig, TinyGPT, generate


def _tiny_cfg():
    return GPTConfig(vocab_size=128, d_model=64, n_heads=4, n_layers=2, block_size=256,
                     mlp_ratio=2)


def test_cached_and_uncached_logits_identical():
    torch.manual_seed(0)
    model = TinyGPT(_tiny_cfg()).eval()
    idx = torch.randint(0, 128, (1, 40))

    logits_full = model(idx)  # no cache

    cache = KVCache.init_for(model)
    logits_cached = model(idx, cache)
    torch.testing.assert_close(logits_full, logits_cached, rtol=1e-4, atol=1e-5)

    # incremental: feed tokens one by one from a fresh cache, last logits must match
    cache2 = KVCache.init_for(model)
    last = None
    for t in range(idx.shape[1]):
        last = model(idx[:, t:t + 1], cache2)
    torch.testing.assert_close(logits_full[:, -1], last[:, -1], rtol=1e-4, atol=1e-5)


def test_cache_append_commit_semantics():
    c = KVCache(n_layers=1, n_heads=2, head_dim=8, max_len=16,
                dtype=torch.float32, device=torch.device("cpu"))
    k = torch.randn(1, 2, 3, 8)
    v = torch.randn(1, 2, 3, 8)
    kk, vv = c.append(0, k, v)
    assert kk.shape[2] == 3 and c.seq_len == 0  # visible but not committed
    c.commit(3)
    assert c.seq_len == 3
    kk, vv = c.append(0, k[:, :, :1], v[:, :, :1])
    assert kk.shape[2] == 4


def test_generate_cache_and_no_cache_agree_greedy():
    torch.manual_seed(1)
    model = TinyGPT(_tiny_cfg()).eval()
    prompt = torch.randint(0, 128, (1, 16))
    with torch.no_grad():
        a = generate(model, prompt, max_new_tokens=20, use_cache=True)
        b = generate(model, prompt, max_new_tokens=20, use_cache=False)
    assert torch.equal(a, b)


def test_cache_memory_scales_linearly():
    c = KVCache(n_layers=2, n_heads=2, head_dim=8, max_len=1024,
                dtype=torch.float32, device=torch.device("cpu"))
    assert c.memory_bytes() == 2 * 2 * 1024 * 8 * 4 * 2  # layers*heads*len*hd*bytes*k+v
