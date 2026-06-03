#!/usr/bin/env bash
set -euo pipefail

export HF_HOME="${HF_HOME:-/workspace/hf-cache}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

uv run python -m llm_benchmark.benchmark \
  --model sshleifer/tiny-gpt2 \
  --dtype fp16 \
  --device cuda:0 \
  --prompt-lengths 16 \
  --batch-sizes 1 \
  --concurrencies 1 \
  --output-pattern 2 \
  --warmup \
  --output results/smoke/smoke.jsonl
