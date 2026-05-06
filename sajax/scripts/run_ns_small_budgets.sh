#!/bin/bash
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem-per-cpu=4G
#SBATCH -p mit_preemptable
#SBATCH -G h100:1
#SBATCH --job-name=ns_small
#SBATCH -o sajax/output/scaling_compute/slurm_%x_%j.out
#SBATCH -e sajax/output/scaling_compute/slurm_%x_%j.err

module load miniforge
module load nvhpc
module load cuda/13.0.1

source activate base
conda activate mcmc_bench

mkdir -p sajax/output/scaling_compute

python -u sajax/scripts/run_ns_small_budgets.py
