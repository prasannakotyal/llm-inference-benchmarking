#!/usr/bin/env bash
set -euo pipefail

export HF_HOME="${HF_HOME:-/workspace/hf-cache}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

uv run python -m llm_benchmark.benchmark \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --dtype fp16 \
  --device cuda:0 \
  --prompt-lengths 64,256,512 \
  --batch-sizes 1,4,8 \
  --concurrencies 4,8,16 \
  --output-pattern 16,32 \
  --warmup \
  --output results/runpod/benchmark_results.jsonl

uv run python -m llm_benchmark.plot_results \
  results/runpod/benchmark_results.jsonl \
  --output-dir assets/runpod

uv run python -m llm_benchmark.summarize \
  results/runpod/benchmark_results.jsonl > results/runpod/summary.md
