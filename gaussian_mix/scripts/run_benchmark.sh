#!/bin/bash
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem-per-cpu=4G
#SBATCH -p mit_preemptable
#SBATCH -G l40s:1
#SBATCH -o gaussian_mix/output/benchmark/slurm_%j.out
#SBATCH -e gaussian_mix/output/benchmark/slurm_%j.err

module load miniforge
module load nvhpc
module load cuda/13.0.1

source activate base
conda activate mcmc_bench

python -u gaussian_mix/scripts/benchmark.py
