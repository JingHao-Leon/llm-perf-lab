# llm-perf-lab · LLM 推理性能优化实验场

**用可运行、可验证、可复现的代码拆解 LLM 推理加速的四大支柱：KV Cache、Attention 内核、权重量化、Triton 融合算子。所有基准数据均为本机实测，附复现命令。**

[![tests](https://img.shields.io/badge/tests-14%20passed-brightgreen)]()
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> 与其背"FlashAttention 是 IO 感知算法"，不如亲手写出 KV Cache、看着它把解码吞吐提升 3.3 倍、把量化误差压到 0.8%。这个仓库里的每个优化都有**正确性测试**（结果必须与朴素实现一致）和**性能基准**（数据必须可复现）。

## 📊 实测结果（Apple M-series · MPS · torch 2.14，复现命令见下）

### KV Cache：解码加速随 prompt 增长

| prompt 长度 | 无 cache | KV Cache | 加速比 |
|---|---|---|---|
| 128 | 143.8 tok/s | 148.6 tok/s | 1.03x |
| 512 | 97.7 tok/s | 173.5 tok/s | **1.78x** |
| 1024 | 49.5 tok/s | 161.1 tok/s | **3.26x** |

> 短 prompt 时收益不明显（kernel launch 开销主导），这正是"为什么要有 prefill/decode 分离与 continuous batching"的直觉来源。测试保证：**开/关 cache 的贪心解码输出逐 token 相同**。

### Attention 后端对比（fp16，B=1，H=8，hd=64）

| 序列长度 | naive（O(T²) 显存） | chunked（分块降峰值） | SDPA（融合内核） |
|---|---|---|---|
| 512 | 2.44 ms | 1.16 ms | **0.44 ms** |
| 1024 | 4.91 ms | 4.80 ms | **1.23 ms** |
| 2048 | 11.34 ms | 13.10 ms | **2.44 ms**（4.7x） |

> SDPA 在 T=2048 时快 4.7 倍——这就是融合内核省下显存读写的直接体现。测试保证三种实现**数值一致**（rtol 1e-4）。

### 权重量化：精度损失 vs 显存占用（2048×2048 Linear，batch 64）

| 方案 | 权重体积 | 相对误差 | 耗时 |
|---|---|---|---|
| fp16 基线 | 16 MiB | 0 | 0.99 ms |
| INT8（逐通道对称） | 4 MiB（**4.0x 压缩**） | **0.83%** | 2.08 ms* |
| NF4（QLoRA 同款 16 级电平） | 4.25 MiB（3.76x vs fp32） | 9.20% | 3.02 ms* |

> *本实现按"反量化再计算"教学路径实现；生产引擎（llama.cpp / bitsandbytes）将反量化融合进 GEMM，故不以此耗时作为量化收益论据——**压缩比与数值契约**才是重点。NF4 电平表逐字转录自 QLoRA 论文（Dettmers et al. 2023），块缩放格式与 bitsandbytes 一致。

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
uv run python benchmarks/bench_kv_cache.py    # 复现 KV Cache 数据
uv run python benchmarks/bench_attention.py   # 复现 attention 数据
uv run python benchmarks/bench_quant.py       # 复现量化数据
uv run python benchmarks/bench_triton.py      # 需要 CUDA GPU
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
