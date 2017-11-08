#!/bin/bash
#SBATCH -n 1
#SBATCH -t 10:00:00
#SBATCH --mem-per-cpu=15G

#SBATCH --array=1-44


srun automated-runs.py 0.05 --input $SLURM_ARRAY_TASK_ID
