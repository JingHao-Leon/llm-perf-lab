# llm-perf-lab · LLM 推理性能优化实验场

**用可运行、可验证、可复现的代码拆解 LLM 推理加速的四大支柱：KV Cache、Attention 内核、权重量化、Triton 融合算子。所有基准数据均为本机实测，附复现命令。**

[![tests](https://img.shields.io/badge/tests-14%20passed-brightgreen)]()
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> 与其背"FlashAttention 是 IO 感知算法"，不如亲手写出 KV Cache、看着它把解码吞吐提升 3.3 倍、把量化误差压到 0.8%。这个仓库里的每个优化都有**正确性测试**（结果必须与朴素实现一致）和**性能基准**（数据必须可复现）。

## 📊 实测结果

双设备实测：**RTX 3090 24GB（CUDA · torch 2.11+cu128）** 与 Apple M-series（MPS · torch 2.14），原始数据在 `benchmarks/results/`（按设备分目录），全部可用仓库内脚本一键复现。

### KV Cache：从 launch 开销主导到 O(T²) 主导的完整 crossover（RTX 3090 · 8L×512d · 35M 参数）

| prompt 长度 | 无 cache | KV Cache | 加速比 |
|---|---|---|---|
| 512 | 153.6 tok/s | 138.7 tok/s | 0.90x ⚠️ |
| 1024 | 74.7 tok/s | 88.7 tok/s | 1.19x |
| 2048 | 22.6 tok/s | 77.0 tok/s | 3.41x |
| 4096 | 0.2 tok/s | 41.2 tok/s | **179.4x** 🚀 |

> 这张表是本仓库最有讲头的实测：4096 长度下无 cache 重计算要 **557 秒**，KV cache 只要 **3.1 秒**——O(T²) 与 O(T) 的差距在长上下文上被 GPU 彻底放大；而 512 长度时 cache 反而慢 10%，因为每步重算前缀的计算量还小，Python 循环 + kernel launch 开销主导。**crossover 点的存在本身就是"为什么要有 prefill/decode 分离"的实证**。（MPS 上同款 crossover 见 `benchmarks/results/kv_cache.md`，1024 时 3.26x）

### Attention 后端（RTX 3090 · fp16 · B=1 · H=8 · hd=64）

| 序列长度 | naive（O(T²) 显存） | chunked（分块降峰值） | SDPA（融合内核） | SDPA 加速 |
|---|---|---|---|---|
| 512 | 0.207 ms | 0.758 ms | **0.057 ms** | 3.6x |
| 1024 | 0.321 ms | 1.954 ms | **0.101 ms** | 3.2x |
| 2048 | 1.112 ms | 3.470 ms | **0.205 ms** | 5.4x |
| 4096 | 4.452 ms | 6.527 ms | **0.514 ms** | 8.7x |
| 8192 | —(跳过) | 22.769 ms | **1.553 ms** | 14.7x |

> 随长度增长 SDPA 优势从 3.6x 拉到 14.7x——融合内核省下的显存读写量与 T 成正比，这正是 FlashAttention "IO-aware" 论文的实测注脚。三种实现数值一致性由测试锁定（rtol 1e-4）。

### 权重量化（RTX 3090 · 2048×2048 Linear · batch 64）

| 方案 | 权重体积 | 相对误差 | 耗时 |
|---|---|---|---|
| fp16 基线 | 16 MiB | 0 | 0.064 ms |
| INT8（逐通道对称） | 4 MiB（**4.0x 压缩**） | **0.83%** | 0.100 ms* |
| NF4（QLoRA 同款 16 电平） | 4.25 MiB（3.76x vs fp32） | 9.20% | 0.229 ms* |

> *按"反量化再计算"教学路径实现；生产引擎将反量化融合进 GEMM。压缩比与数值契约是重点。

### Triton 融合算子（3090 + triton-windows · 4096×4096 fp16）

| 内核 | PyTorch eager | 本仓 Triton | 结果 |
|---|---|---|---|
| RMSNorm | 0.136 ms | 0.180 ms | 0.75x |
| Softmax | 0.186 ms | 0.219 ms | 0.85x |

> 诚实结论：**数值验证通过（rtol/atol 1e-2），但没打过 PyTorch eager**。两个原因：triton-windows 是社区移植版（编译质量低于 Linux 官方版），且本内核是未调优的教学实现（num_warps/block size 未扫参）；PyTorch eager 底层本来就是高度优化的 CUDA 库。Linux + 官方 Triton + 调优才是这类内核的正确打开方式——正确性已由测试锁定，性能留作 Roadmap。

## 🧩 项目结构

```
perf_lab/
├── kv_cache.py      # 预分配 KV Cache：append/commit 两段式，O(1) 附加新 token
├── model.py         # 极简 decoder-only Transformer（RMSNorm+GELU MLP），支持增量解码
├── attention.py     # naive / chunked / SDPA 三实现 + 等价性保证
├── quant.py         # INT8 逐通道量化、NF4 块量化、QuantizedLinear 即插即用
└── bench.py         # 计时框架（warmup+中位数，CUDA/MPS 同步），JSON+Markdown 报告
triton_kernels/      # Triton 融合 RMSNorm / Softmax（CUDA 环境运行，见 benchmarks/bench_triton.py）
benchmarks/          # 四个基准脚本 + results/ 实测数据
tests/               # 14 个测试：数值等价、误差上界、贪心输出一致、内存线性
```

## 🚀 快速开始

```bash
uv sync                # 或 pip install torch numpy pytest
uv run pytest          # 14 tests passed
uv run python benchmarks/bench_kv_cache.py --prompts 512 1024 2048 4096 --d-model 512 --layers 8 --heads 16   # 复现 3090 表
uv run python benchmarks/bench_attention.py --seq 512 1024 2048 4096 8192  # 复现 attention 表
uv run python benchmarks/bench_quant.py       # 复现量化数据
uv run python benchmarks/bench_triton.py      # CUDA GPU + triton
```

CPU / MPS / CUDA 均可运行核心测试与前三项基准（设备自动选择）；Triton 内核仅在 CUDA 上运行，无 GPU 时优雅跳过。

## 📐 数值契约（为什么这些测试是有意义的）

- **KV Cache**：`test_cached_and_uncached_logits_identical` 要求逐 token 喂入与整段前向的 logits 完全一致——cache 不是"大约对"，是**位级对齐的数学等价**。
- **INT8**：误差上界有闭式证明 `max_err ≤ absmax/254`（逐通道对称量化），测试直接验证该界。
- **NF4**：块内 absmax 缩放 + 16 电平最近邻，实测相对误差 9.2%——与论文报告的 4-bit 量级一致。
- **Chunked attention**：任意 chunk 大小与 naive 完全一致（rtol 1e-4），证明分块只是重排计算、不改变语义。

## 🛣️ Roadmap

- [ ] Paged KV Cache（vLLM 式分页，支持 prefix sharing）
- [ ] Speculative decoding（草稿模型 + 验证接受）
- [ ] CUDA C++ GEMM tiling（配合 nsight 计算屋顶线分析）

## License

MIT
