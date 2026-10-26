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
#SBATCH --open-mode=append          # a requeued job (see the end) adds to its log instead of overwriting it
#SBATCH --requeue                   # allows the requeue at the end
#SBATCH --partition=project
#SBATCH --account=cs425
#SBATCH --qos=cs425qos
#SBATCH --job-name=sar-train
# For emails, remove one '#' from each of these two lines and put in your email:
##SBATCH --mail-type=END,FAIL
##SBATCH --mail-user=YOUR_EMAIL@scis.smu.edu.sg

set -e
# A first argument ending in .py picks another script: sbatch train.slurm pretrain_ssl.py --backbone swin_b ...
script=train.py
if [[ "$1" == *.py ]]; then script=$1; shift; fi

module purge
module load Anaconda3/2024.06-1
eval "$(conda shell.bash hook)"
if [[ $script == yolo26.py ]]; then conda activate cs701y26; else conda activate cs701bench; fi  # newer Ultralytics

cd "$SLURM_SUBMIT_DIR"              # the repo folder sbatch was run from
nvidia-smi                          # which GPU this job got, at the top of the log

# Runs longer than the job's time limit (train.py, yolo26.py): --stop-hours = the limit minus 15 minutes, so the run
# stops after an epoch with its checkpoint (exit code 75) and writes its folder to logs/resume_<job id>; the job then
# requeues itself (same job id and sbatch options, not a new submission) and the requeued job resumes that run.
note="logs/resume_$SLURM_JOB_ID"
extra=()
if [[ $script == train.py || $script == yolo26.py ]]; then
    if [[ -f $note ]]; then set -- --resume "$(cat "$note")"; fi
    limit=$(squeue -h -j "$SLURM_JOB_ID" -o %l)  # [days-]hours:minutes:seconds, minutes:seconds or UNLIMITED
    if [[ $limit != UNLIMITED ]]; then
        days=0
        if [[ $limit == *-* ]]; then days=${limit%%-*}; limit=${limit#*-}; fi
        seconds=0
        IFS=: read -ra parts <<< "$limit"
        for part in "${parts[@]}"; do seconds=$((seconds * 60 + 10#$part)); done
        extra=(--stop-hours "$(awk -v s=$((days * 86400 + seconds)) 'BEGIN { printf "%.3f", (s - 900) / 3600 }')")
    fi
fi
# The env's python by full path: when sbatch runs from a shell with cs701bench already active, `module load` puts
# Anaconda's base python first on PATH and `conda activate` (seeing the env already active) does not fix it.
status=0
RESUME_NOTE=$note srun "$CONDA_PREFIX/bin/python" "$script" --data data "$@" "${extra[@]}" || status=$?
if [[ $status == 75 ]]; then
    echo "time limit near: requeueing job $SLURM_JOB_ID to resume $(cat "$note")"
    if scontrol requeue "$SLURM_JOB_ID"; then sleep 300; fi  # the requeue ends this job; the note stays for it
    echo "requeue failed; continue by hand: sbatch train.slurm $script --resume $(cat "$note")"
else
    rm -f "$note"  # finished (or failed): a later requeue must not resume
fi
exit $status
