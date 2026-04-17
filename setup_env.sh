#!/usr/bin/env bash
set -e

pip install uv

# Install JAX before the rest of the project so uv can resolve against the
# correct jax/jaxlib versions when it processes pyproject.toml.
if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
    echo "GPU detected — loading CUDA 12 modules and installing jax[cuda12]"
    module load cuda/12.9.1 cudnn/9.8.0.87-cuda12
    uv pip install "jax[cuda12]==0.9.2" "jaxlib==0.9.2" \
        --find-links https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
else
    echo "No GPU detected — installing jax[cpu]"
    uv pip install "jax[cpu]==0.9.2" "jaxlib==0.9.2"
fi

# Install the project and all remaining dependencies from pyproject.toml.
uv pip install -e .
