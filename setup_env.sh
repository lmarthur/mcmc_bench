#!/usr/bin/env bash
set -e

BASE_DEPS="blackjax numpyro exojax numpy matplotlib scipy optax arviz ipython pytest pytest-cov"

if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
    echo "GPU detected — installing jax[cuda12]"
    uv pip install $BASE_DEPS "jax[cuda12]" jaxlib \
        --find-links https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
else
    echo "No GPU detected — installing jax[cpu]"
    uv pip install $BASE_DEPS "jax[cpu]" jaxlib
fi
