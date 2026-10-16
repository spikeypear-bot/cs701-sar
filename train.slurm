#!/bin/bash
# One training run on the SMU SCIS GPU cluster. Submit from the repo folder; options go to train.py:
#   mkdir -p logs   (once)
#   sbatch train.slurm --backbone terramind --init pretrained --adapt lora --name tm-lora
# Limits of cs425qos: 1 running job, 2 submitted, 1 GPU, 30 CPUs, 100 GB RAM, 1 day.

#SBATCH --nodes=1
#SBATCH --cpus-per-task=8           # train.py uses 8 data-loading workers
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00             # one run is about 1 h on a fast GPU; the partition allows at most 1 day
#SBATCH --output=logs/%x_%j.out     # %x = job name, %j = job id; logs/ must exist
#SBATCH --partition=project
#SBATCH --account=cs425
#SBATCH --qos=cs425qos
#SBATCH --job-name=sar-train
# For emails, remove one '#' from each of these two lines and put in your email:
##SBATCH --mail-type=END,FAIL
##SBATCH --mail-user=YOUR_EMAIL@scis.smu.edu.sg

set -e
module purge
module load Anaconda3/2024.06-1
eval "$(conda shell.bash hook)"
conda activate cs701bench

cd "$SLURM_SUBMIT_DIR"              # the repo folder sbatch was run from
nvidia-smi                          # which GPU this job got, at the top of the log

# The env's python by full path: when sbatch runs from a shell with cs701bench already active, `module load` puts
# Anaconda's base python first on PATH and `conda activate` (seeing the env already active) does not fix it.
srun "$CONDA_PREFIX/bin/python" train.py --data data "$@"
