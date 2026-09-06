import pytest
import torch

from perf_lab.kv_cache import KVCache
from perf_lab.model import GPTConfig, TinyGPT, generate
from perf_lab.speculative import SpecStats, speculative_generate


def _models(seed=0):
    torch.manual_seed(seed)
    shared = dict(vocab_size=128, block_size=256)
    target = TinyGPT(GPTConfig(d_model=96, n_heads=4, n_layers=3, **shared)).eval()
    draft = TinyGPT(GPTConfig(d_model=48, n_heads=4, n_layers=1, **shared)).eval()
    return target, draft


@pytest.mark.parametrize("gamma", [1, 2, 4, 8])
def test_output_identical_to_target_greedy(gamma):
    torch.manual_seed(0)
    target, draft = _models(seed=3)
    prompt = torch.randint(0, 128, (1, 32))
    reference = generate(target, prompt, max_new_tokens=48, use_cache=True)
    spec_out, stats = speculative_generate(target, draft, prompt,
                                           max_new_tokens=48, gamma=gamma)
    assert spec_out.shape == reference[:, prompt.shape[1]:].shape
    assert torch.equal(spec_out, reference[:, prompt.shape[1]:]), (
        "speculative decoding must be token-for-token identical to target greedy"
    )
    assert stats.emitted == spec_out.shape[1]
    assert stats.accepted <= stats.proposed
    assert 0.0 <= stats.acceptance_rate <= 1.0
    assert stats.tokens_per_target_forward > 0


def test_identical_draft_model_accepts_everything():
    target, _ = _models(seed=5)
    draft = target  # same model: greedy proposals always match target choices
    prompt = torch.randint(0, 128, (1, 24))
    _, stats = speculative_generate(target, draft, prompt, max_new_tokens=40, gamma=4)
    assert stats.acceptance_rate == 1.0
    # every target forward emits the full gamma + 1 bonus
    assert stats.tokens_per_target_forward == pytest.approx(5.0)


def test_random_draft_still_correct():
    torch.manual_seed(7)
    target, draft = _models(seed=11)
    prompt = torch.randint(0, 128, (2, 20))  # batch of 2
    reference = generate(target, prompt, max_new_tokens=30, use_cache=True)
    spec_out, stats = speculative_generate(target, draft, prompt, max_new_tokens=30)
    assert torch.equal(spec_out, reference[:, prompt.shape[1]:])


def test_cache_commit_upto_and_rollback():
    c = KVCache(n_layers=1, n_heads=2, head_dim=8, max_len=32,
                dtype=torch.float32, device=torch.device("cpu"), batch_size=1)
    k = torch.randn(1, 2, 3, 8)
    c.append(0, k, k)
    c.commit_upto(2)  # accept only 2 of 3
    assert c.seq_len == 2
    c.rollback(0)
    assert c.seq_len == 0
    c.append(0, k[:, :, :1], k[:, :, :1])
    assert c.seq_len == 0  # pending until committed
    c.commit(1)
    assert c.seq_len == 1


def test_defer_commit_keeps_pending():
    torch.manual_seed(0)
    model = TinyGPT(GPTConfig(vocab_size=32, d_model=32, n_heads=2, n_layers=1,
                               block_size=64)).eval()
    cache = KVCache.init_for(model)
    idx = torch.randint(0, 32, (1, 8))
    model(idx, cache, defer_commit=True)
    assert cache.seq_len == 0  # nothing committed
    cache.commit_upto(5)       # accept a prefix
    assert cache.seq_len == 5
