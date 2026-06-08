#!/usr/bin/env bash
#
# Start vLLM with the Phase 1 chosen configuration.
# Flag rationale documented in REPORT.md.
# Reference: https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html

set -euo pipefail

MODEL="Qwen/Qwen3-30B-A3B-Instruct-2507"

exec uv run python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --host 0.0.0.0 \
    --port 8000 \
    --quantization fp8 \
    --kv-cache-dtype fp8 \
    --max-model-len 8192 \
    --max-num-seqs 64 \
    --max-num-batched-tokens 2048 \
    --gpu-memory-utilization 0.92 \
    --enable-prefix-caching \
    --enable-chunked-prefill
