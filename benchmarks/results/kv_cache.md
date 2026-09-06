### Decode with/without KV cache (new=128, 4L×256d, 4.5M params)
device: `mps`

| 指标 | 值 | 单位 | 备注 |
|---|---|---|---|
| prompt=128 no cache | 0.8904 | s | 143.8 tok/s |
| prompt=128 KV cache | 0.8611 | s | 148.6 tok/s |
| prompt=128 speedup | 1.034 | x | no-cache / cache |
| prompt=512 no cache | 1.3107 | s | 97.7 tok/s |
| prompt=512 KV cache | 0.738 | s | 173.5 tok/s |
| prompt=512 speedup | 1.7762 | x | no-cache / cache |
| prompt=1024 no cache | 2.5869 | s | 49.5 tok/s |
| prompt=1024 KV cache | 0.7946 | s | 161.1 tok/s |
| prompt=1024 speedup | 3.2555 | x | no-cache / cache |
| cache memory @block_size | 9280.0 | KiB | 2*n_layers*B*H*len*hd*bytes |

