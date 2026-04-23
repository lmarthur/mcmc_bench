#!/bin/bash
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem-per-cpu=4G
#SBATCH -p mit_normal_gpu
#SBATCH -G l40s:1
#SBATCH -o sajax/output/ns/slurm_%j.out
#SBATCH -e sajax/output/ns/slurm_%j.err

module load miniforge
module load nvhpc
module load cuda/13.0.1

source activate base
conda activate mcmc_bench

python -u sajax/scripts/ns.py
