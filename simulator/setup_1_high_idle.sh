#!/bin/bash
#SBATCH -n 1
#SBATCH -t 10:00:00
#SBATCH --mem-per-cpu=15G

#SBATCH --array=32-63


srun automated-runs_high_idle_power.py --input $SLURM_ARRAY_TASK_ID
