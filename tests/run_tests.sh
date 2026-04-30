#!/bin/bash
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem-per-cpu=4G
#SBATCH -p mit_preemptable
#SBATCH -G l40s:1
#SBATCH -o tests/output/test.out
#SBATCH -e tests/output/test.err

mkdir -p tests/output

module load miniforge
module load nvhpc
module load cuda/13.0.1

source activate base
conda activate mcmc_bench

pytest -v
