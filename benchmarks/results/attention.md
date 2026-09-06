### Attention backends (B=1, H=8, hd=64, dtype=torch.float16)
device: `mps`

| 指标 | 值 | 单位 | 备注 |
|---|---|---|---|
| T=512 naive | 2.4432 | ms | median of 5 |
| T=512 chunked | 1.1607 | ms | median of 5 |
| T=512 sdpa | 0.4405 | ms | median of 5 |
| T=1024 naive | 4.9137 | ms | median of 5 |
| T=1024 chunked | 4.8039 | ms | median of 5 |
| T=1024 sdpa | 1.2318 | ms | median of 5 |
| T=2048 naive | 11.3396 | ms | median of 5 |
| T=2048 chunked | 13.096 | ms | median of 5 |
| T=2048 sdpa | 2.436 | ms | median of 5 |

