# mcmc-bench

Benchmarking MCMC samplers on multimodal posterior inference problems in astrophysics and exoplanet science. All inference algorithms are implemented in JAX via [BlackJAX](https://github.com/blackjax-devs/blackjax), [NumPyro](https://github.com/pyro-ppl/numpyro), and related packages.

## Setup

### Prerequisites
- [conda](https://docs.conda.io/en/latest/) (or miniconda)

### Installation

```bash
conda create -n mcmc_bench python=3.11 -y
conda activate mcmc_bench
bash setup_env.sh
```

`setup_env.sh` detects whether a CUDA-capable GPU is available and installs `jax[cuda12]` or `jax[cpu]` accordingly. All other dependencies are defined in `pyproject.toml`.

## Running tests

```bash
pytest --rootdir=. -v
```

## Benchmark Problems

Two problems are included, each under its own directory with a common structure (`src/model.py` for the model, `scripts/<algo>.py` for each sampler):

- **`gaussian_mix/`** — 2D mixture of 8 Gaussians on a regular octagon; fast to evaluate, designed to expose mode-missing failures.
- **`sajax/`** — 13-parameter exoplanet transit + stellar spot crossing using [sajax](https://github.com/SamMerc/sajax) and [jaxoplanet](https://github.com/exoplanet-dev/jaxoplanet).

Samplers available for each problem: RWMH, NUTS, DEO/SEO parallel tempering, SMC, nested sampling (JAXNS), and affine-invariant ensemble (emcee_jax).

To run a sampler:
```bash
python gaussian_mix/scripts/rwmh.py
python sajax/scripts/deo.py
```

To run the full benchmark comparison:
```bash
python gaussian_mix/scripts/benchmark.py
python gaussian_mix/scripts/scaling_study_compute.py
```

Results (JSON + plots) are saved under `gaussian_mix/output/` and `sajax/output/`.
