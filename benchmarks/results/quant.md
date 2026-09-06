### Quantization: fp16 vs int8 vs nf4 (2048x2048 linear, batch 64)
device: `mps`

| 指标 | 值 | 单位 | 备注 |
|---|---|---|---|
| fp matmul | 0.9943 | ms | weight 16384 KiB |
| int8 matmul | 2.0806 | ms | weight 4096 KiB, rel err 0.83% |
| int8 compression | 4.0 | x | vs fp32 weight |
| nf4 matmul | 3.0243 | ms | weight 4352 KiB, rel err 9.20% |
| nf4 compression | 3.7647 | x | vs fp32 weight |
| nf4 quantize | 30.3267 | ms | one-time, per weight tensor |

