#!/bin/bash
#SBATCH -n 1
#SBATCH -t 10:00:00
#SBATCH --mem-per-cpu=15G

#SBATCH --array=0-27


srun automated-runs.py --input $SLURM_ARRAY_TASK_ID
