### Attention backends (B=1, H=8, hd=64, dtype=torch.float16)
device: `mps`

| ÊåáÊ†á | ÂÄº | Âçï‰Ωç | Â§áÊ≥® |
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

### Attention backends (B=1, H=8, hd=64, dtype=torch.float16)
device: `cuda`

| ÷∏±Í | ÷µ | µ•Œª | ±∏◊¢ |
|---|---|---|---|
| T=512 naive | 0.2074 | ms | median of 5 |
| T=512 chunked | 0.7578 | ms | median of 5 |
| T=512 sdpa | 0.0574 | ms | median of 5 |
| T=1024 naive | 0.3214 | ms | median of 5 |
| T=1024 chunked | 1.9539 | ms | median of 5 |
| T=1024 sdpa | 0.1011 | ms | median of 5 |
| T=2048 naive | 1.1122 | ms | median of 5 |
| T=2048 chunked | 3.4699 | ms | median of 5 |
| T=2048 sdpa | 0.2049 | ms | median of 5 |
| T=4096 naive | 4.4519 | ms | median of 5 |
| T=4096 chunked | 6.5265 | ms | median of 5 |
| T=4096 sdpa | 0.5137 | ms | median of 5 |
| T=8192 chunked | 22.7692 | ms | median of 5 |
| T=8192 sdpa | 1.5526 | ms | median of 5 |

