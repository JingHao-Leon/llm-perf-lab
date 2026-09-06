### Triton fused kernels (4096x4096 fp16)
device: `cuda`

| 指标 | 值 | 单位 | 备注 |
|---|---|---|---|
| rmsnorm eager | 0.1357 | ms |  |
| rmsnorm triton | 0.1801 | ms |  |
| rmsnorm speedup | 0.7535 | x |  |
| softmax eager | 0.1861 | ms |  |
| softmax triton | 0.2192 | ms |  |
| softmax speedup | 0.849 | x |  |
| correctness | 1 | pass | triton vs eager, rtol/atol 1e-2 |

