### Decode with/without KV cache (new=128, 4L√ó256d, 4.5M params)
device: `mps`

| ÊåáÊ†á | ÂÄº | Âçï‰Ωç | Â§áÊ≥® |
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

### Decode with/without KV cache (new=128, 4L°¡256d, 4.8M params)
device: `cuda`

| ÷∏±Í | ÷µ | µ•Œª | ±∏◊¢ |
|---|---|---|---|
| prompt=128 no cache | 0.5137 | s | 249.2 tok/s |
| prompt=128 KV cache | 0.3597 | s | 355.9 tok/s |
| prompt=128 speedup | 1.4283 | x | no-cache / cache |
| prompt=512 no cache | 0.304 | s | 421.1 tok/s |
| prompt=512 KV cache | 0.3534 | s | 362.2 tok/s |
| prompt=512 speedup | 0.8602 | x | no-cache / cache |
| prompt=1024 no cache | 0.6131 | s | 208.8 tok/s |
| prompt=1024 KV cache | 0.7242 | s | 176.7 tok/s |
| prompt=1024 speedup | 0.8465 | x | no-cache / cache |
| prompt=2048 no cache | 1.3349 | s | 95.9 tok/s |
| prompt=2048 KV cache | 0.8048 | s | 159.1 tok/s |
| prompt=2048 speedup | 1.6587 | x | no-cache / cache |
| cache memory @block_size | 17472.0 | KiB | 2*n_layers*B*H*len*hd*bytes |

### Decode with/without KV cache (new=128, 8L°¡512d, 29.5M params)
device: `cuda`

| ÷∏±Í | ÷µ | µ•Œª | ±∏◊¢ |
|---|---|---|---|
| prompt=512 no cache | 0.8331 | s | 153.6 tok/s |
| prompt=512 KV cache | 0.9226 | s | 138.7 tok/s |
| prompt=512 speedup | 0.903 | x | no-cache / cache |
| prompt=1024 no cache | 1.7145 | s | 74.7 tok/s |
| prompt=1024 KV cache | 1.4427 | s | 88.7 tok/s |
| prompt=1024 speedup | 1.1884 | x | no-cache / cache |
| prompt=2048 no cache | 5.671 | s | 22.6 tok/s |
| prompt=2048 KV cache | 1.6627 | s | 77.0 tok/s |
| prompt=2048 speedup | 3.4107 | x | no-cache / cache |
| prompt=4096 no cache | 557.6724 | s | 0.2 tok/s |
| prompt=4096 KV cache | 3.1087 | s | 41.2 tok/s |
| prompt=4096 speedup | 179.3886 | x | no-cache / cache |
| cache memory @block_size | 135424.0 | KiB | 2*n_layers*B*H*len*hd*bytes |

