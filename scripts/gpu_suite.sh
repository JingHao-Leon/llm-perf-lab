#!/usr/bin/env bash
# One-shot GPU suite: run every benchmark + a real QLoRA fine-tune, then
# bundle all results for pasting back into the READMEs.
#
#   bash gpu_suite.sh [workdir]        # default ~/ai-gpu-suite
#
# Requirements: Linux/CUDA GPU (tested target: RTX 3090 24GB), git, uv.
# uv install: https://docs.astral.sh/uv/getting-started/installation/
set -euo pipefail

WORK="${1:-$HOME/ai-gpu-suite}"
mkdir -p "$WORK" && cd "$WORK"

if ! command -v uv >/dev/null 2>&1; then
    echo "ERROR: uv is not installed. Install it first (see https://docs.astral.sh/uv/)" >&2
    exit 1
fi
export PATH="$HOME/.local/bin:$PATH"

if [ ! -d llm-perf-lab ]; then git clone https://github.com/JingHao-Leon/llm-perf-lab; fi
if [ ! -d lora-lab ]; then git clone https://github.com/JingHao-Leon/lora-lab; fi

echo "=========== [1/2] llm-perf-lab: tests + benchmarks ==========="
cd llm-perf-lab
git pull -q || true
uv sync --extra gpu
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || true
uv run pytest -q
uv run python benchmarks/bench_kv_cache.py --prompts 128 512 1024 2048
uv run python benchmarks/bench_attention.py --seq 512 1024 2048 4096 8192
uv run python benchmarks/bench_quant.py
uv run python benchmarks/bench_triton.py
cd ..

echo "=========== [2/2] lora-lab: Qwen QLoRA fine-tune ==========="
cd lora-lab
git pull -q || true
uv sync --extra gpu
uv run pytest -q
uv run python examples/train_qwen_gpu.py \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --dataset belle --samples 3000 --epochs 2
cd ..

STAMP=$(date +%m%d_%H%M)
tar czf "gpu_suite_results_${STAMP}.tar.gz" \
    llm-perf-lab/benchmarks/results lora-lab/runs 2>/dev/null || true
echo
echo "DONE — results bundle: $WORK/gpu_suite_results_${STAMP}.tar.gz"
echo "Send this file back (or paste the console output) to update the READMEs with real 3090 numbers."
