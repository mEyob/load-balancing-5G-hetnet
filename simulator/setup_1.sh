#!/bin/bash
#SBATCH -n 1
#SBATCH -t 10:00:00
#SBATCH --mem-per-cpu=15G

#SBATCH --array=0-31


srun automated-runs.py --input inputs_2/input_$SLURM_ARRAY_TASK_ID
