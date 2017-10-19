#!/bin/bash
#SBATCH -n 1
#SBATCH -t 10:00:00
#SBATCH --mem-per-cpu=15G


srun weight_optimization.py 2 10 0.1 1 -c 0.12




