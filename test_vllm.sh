#!/bin/bash
export LD_LIBRARY_PATH=$(find /mnt/c/Users/Swapnil/Projects/DART/.venv/lib/python3.14/site-packages/nvidia/ -type d -name "lib" | paste -sd : -):$LD_LIBRARY_PATH
$HOME/.local/bin/uv run python -c "import torch; print('CUDA:', torch.cuda.is_available()); from vllm import LLM; print('vLLM OK:', LLM)"


