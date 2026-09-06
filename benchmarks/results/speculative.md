### Speculative decoding (draft=target truncated 2L×256d, target 6L×256d 4.9M, prompt=256, new=256)
device: `mps`

| 指标 | 值 | 单位 | 备注 |
|---|---|---|---|
| target greedy (KV cache) | 1.6394 | s | 156 tok/s |
| speculative γ=2 | 3.0621 | s | 84 tok/s, accept 100%, 3.01 tok/forward |
| γ=2 speedup | 0.5354 | x |  |
| speculative γ=4 | 2.2535 | s | 114 tok/s, accept 100%, 5.02 tok/forward |
| γ=4 speedup | 0.7275 | x |  |
| speculative γ=8 | 1.6695 | s | 153 tok/s, accept 100%, 8.83 tok/forward |
| γ=8 speedup | 0.9819 | x |  |

