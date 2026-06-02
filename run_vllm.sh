#!/bin/bash
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_DISABLE_FLASHINFER=1
export PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH=$(find /mnt/c/Users/Swapnil/Projects/DART/.venv/lib/python3.14/site-packages/nvidia/ -type d -name "lib" | paste -sd : -):$LD_LIBRARY_PATH
/mnt/c/Users/Swapnil/Projects/DART/.venv/bin/python -u -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-1.5B-Instruct --gpu-memory-utilization 0.7 --port 8000 "$@"
