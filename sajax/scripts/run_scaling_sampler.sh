#!/bin/bash
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem-per-cpu=4G
#SBATCH -p mit_normal_gpu
#SBATCH -G l40s:1
#SBATCH -o sajax/output/scaling_compute/slurm_%x_%j.out
#SBATCH -e sajax/output/scaling_compute/slurm_%x_%j.err

# Usage:
#   sbatch --job-name=rwmh  run_scaling_sampler.sh rwmh
#   sbatch --job-name=smc   run_scaling_sampler.sh smc
#   sbatch --job-name=ns    run_scaling_sampler.sh ns
#
# To rerun a completed job:
#   sbatch --job-name=rwmh  run_scaling_sampler.sh rwmh --force

ALGO=${1:?Usage: sbatch run_scaling_sampler.sh ALGO [--force]}
shift

module load miniforge
module load nvhpc
module load cuda/13.0.1

source activate base
conda activate mcmc_bench

mkdir -p sajax/output/scaling_compute

python -u sajax/scripts/scaling_run_sampler.py --algorithm "$ALGO" "$@"
