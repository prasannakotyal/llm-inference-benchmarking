#!/usr/bin/env bash
set -euo pipefail

export HF_HOME="${HF_HOME:-/workspace/hf-cache}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

uv run python -m llm_benchmark.benchmark \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --dtype fp16 \
  --device cuda:0 \
  --prompt-lengths 64,256,512 \
  --batch-sizes 1,4,8 \
  --concurrencies 4,8,16 \
  --output-pattern 16,32 \
  --warmup \
  --output results/runpod_qwen_1_5b/benchmark_results.jsonl \
  --trace-output results/runpod_qwen_1_5b/request_traces.jsonl

uv run python -m llm_benchmark.plot_results \
  results/runpod_qwen_1_5b/benchmark_results.jsonl \
  --output-dir assets/runpod_qwen_1_5b

uv run python -m llm_benchmark.summarize \
  results/runpod_qwen_1_5b/benchmark_results.jsonl > results/runpod_qwen_1_5b/summary.md

uv run python -m llm_benchmark.export_prometheus \
  results/runpod_qwen_1_5b/benchmark_results.jsonl \
  --output results/runpod_qwen_1_5b/prometheus_metrics.prom
