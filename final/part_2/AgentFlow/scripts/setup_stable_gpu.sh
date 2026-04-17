#!/bin/bash

set -ex

python -m uv pip install --upgrade uv pip

uv pip install --no-cache-dir packaging ninja numpy pandas ipython ipykernel gdown wheel setuptools
uv pip install --no-cache-dir torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install --no-cache-dir transformers==4.53.3

# For local/dev machines without matching CUDA + wheel, allow skipping flash-attn.
if [ "${INSTALL_FLASH_ATTN:-1}" = "1" ]; then
  uv pip install "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.1/flash_attn-2.8.1+cu12torch2.7cxx11abiTRUE-cp311-cp311-linux_x86_64.whl" --no-build-isolation || \
  echo "Warning: flash-attn install failed. Continue without it (set INSTALL_FLASH_ATTN=0 to skip)."
fi

uv pip install --no-cache-dir vllm==0.9.2
uv pip install --no-cache-dir verl==0.5.0

uv pip install --no-cache-dir -e .[dev,agent]
