# Class 6.7830 - Bayesian Modeling and Inference - Final Project

Project Pre-proposal Google Docs [link](https://docs.google.com/document/d/10dxllF_9HKoAEWeWYtg3HkMVfwvhc1fx0QWjYhUD3fQ/edit?usp=sharing)


# Setup

## Prerequisites
- [conda](https://docs.conda.io/en/latest/) (or miniconda)
- [uv](https://github.com/astral-sh/uv) — installed automatically in step 2

## Installation

```bash
conda create -n mcmc_bench python=3.11 -y
conda activate mcmc_bench
pip install uv
bash setup_env.sh
```

`setup_env.sh` detects whether a CUDA-capable GPU is available and installs `jax[cuda12]` or `jax[cpu]` accordingly. All other dependencies are defined in `pyproject.toml`.

# Proposal Outline

## What do you want to do? What questions are you answering?
The primary goal is to benchmark a set of MCMC samplers on multimodal posterior inference problems. We want to evaluate the samplers on a set of metrics covering speed and ability to recover all of the modes. In particular, we want to test the samplers on problems specific to astrophysics and exoplanet science. 

## How does your question and your approach relate to this class? 
In the readings and the lectures we have reviewed a number of MCMC methods, from RWMH to HMC and NUTS. However, these methods are not very popular in the astrophysics and exoplanet communities. The primary reason that is cited is that astrophysical posteriors are often multimodal. So, we want to test how the commonly used samplers in astrophysics compare to the HMC, NUTS, and similar algorithms common in the statistics literature. We also want to perform these tests in an implementation-agnostic way by implementing all of our inference algorithms in a JAX-compatible way. 

## What data will you use? 
I think we should use at least one toy model, perhaps a Gaussian mixture, that can be constructed to have several modes. We can also fit an exoplanet transit with a spot, for which we can easily create synthetic data and already have much of the existing code. Another possible test is atmospheric retrievals, for which we have an implementation that currently relies on the emcee package, which runs affine-invariant MCMC. 

## What is some relevant work?
- Non-reversible parallel tempering: [link](https://ui.adsabs.harvard.edu/abs/2019arXiv190502939S/abstract)
- Dynamic temperature selection for parallel tempering in MCMC simulations [link](https://ui.adsabs.harvard.edu/abs/2016MNRAS.455.1919V/abstract)
- Modern Bayesian Sampling Methods for Cosmological Inference [link](https://arxiv.org/pdf/2501.06022)

## What is your project plan?
- Solidify the example problems and datasets, and get at least one method working for each problem (by proposal deadline)
- Solidify the metrics of comparison and scoring of each method (by proposal deadline)
- Outline each of the figures we want to create and include in our paper (by end of month)
- Implement Non-reversible PT MCMC in JAX/BlackJAX
- Run trials on each of our algorithms (by mid-April)
- Review results and write up report (by end of April)

## What are the risks? 
Implementation of PT MCMC may prove more difficult than expected. Some sampling algorithms may simply fail to converge in reasonable time for some of our cases. The number of test cases and number of algorithms that we plan to test may result in excessive compute times that significantly slow our efforts. 
