---
name: model-evaluation
description: Use when a training run of this SAR detection + classification repo has finished, failed or been stopped, when the user pastes a log, results.csv row, plots or a run folder name, or asks how a run did, whether it over- or underfits, which classes or box sizes it gets wrong, or how it compares with another run.
---

# Model evaluation

## Overview

Runs and analysis scripts execute only on the cluster (CLAUDE.md). So evaluation is two steps: give the user the
commands, then read what they bring back (text output, and the PNGs copied to the laptop) and write the review in
the fixed shape below. Report facts and likely causes; do not propose next experiments (the user picks them, or asks
for the model-planning skill).

## Step 1: commands for the user

Fill in `<run>` (the folder under `runs/`) and `<base>` (the equal-epoch baseline). Login node, repo folder; none
needs a GPU.

```bash
module purge; module load Anaconda3/2024.06-1; eval "$(conda shell.bash hook)"; conda activate cs701bench
PY="$CONDA_PREFIX/bin/python"; echo $CONDA_PREFIX   # must not be empty
column -s, -t runs/results.csv | grep -E "^run|<run name>|<base name>"
grep -E "parameters:|val:|test:|training " runs/<run>/log.txt     # train.py runs; yolo26.py: logs/sar-train_<jobid>.out
$PY analyze_runs.py runs/<base> runs/<run>      # AP by box size and class, F1 per class, top confusions
$PY class_errors.py runs/<run>                  # per class: localisation (AP@0.1..0.75), recall, other-class detections
$PY plot_run.py runs/<base> runs/<run>          # runs/<x>/plots/curves.png, confusion.png; runs/plots/class_ap.png
```

Then on the laptop (repo folder), so the plots can be read here:

```bash
mkdir -p runs/<run> runs/plots
scp -r "$CLUSTER:/common/home/projectgrps/CS425/CS425G9/cs701-sar/runs/<run>/plots" runs/<run>/
scp "$CLUSTER:/common/home/projectgrps/CS425/CS425G9/cs701-sar/runs/plots/class_ap.png" runs/plots/
```

Read every PNG the user brings back with the Read tool before writing the review.

## Step 2: the review, in this order

```markdown
## Evaluation: <run> vs <base>

**Verdict:** <one sentence: better / same within noise / worse, by how much, and the main reason>

| | <base> | <run> | Δ |
|---|---:|---:|---:|
| mAP · AP50 · macro-F1 · dm_rough | | | |
| AP tiny <16 · small · medium · large | | | |
| trainable M · train h (GPU) | | | |

**Per class (AP, F1):** <only rows with |Δ| ≥ 1 point, with n boxes / images from analyze_runs>

**Fit:** <underfitting / overfitting / converged, with the numbers from plot_run.py: best epoch vs last,
last-quarter change, val loss trend (YOLO)>

**Classification:** <top confusions (truth → predicted, counts) and whether they changed>

**Errors (class_errors.py):** <localisation (AP@0.1 high but AP low) vs missed (low R@0.1) per weak class>

**Hypothesis:** <the run's hypothesis from its model summary: supported / not supported / inconclusive, and why>

**Codabench estimate:** holdout dm_rough − ~6 (gap was −5.6 at e48, −5.9 at e100); only an upload confirms it.
```

Then record it with the results-recorder skill.

## Reading the numbers

| signal | reading |
|---|---|
| val mAP last quarter > +0.5 and best epoch = last | underfitting: still learning when the schedule ended |
| best val mAP epoch < last and last ≥ 0.5 below best | overfitting (train.py keeps the last epoch, so the drop is the score) |
| YOLO val loss minimum well before the end, rising while train loss falls | overfitting |
| train loss flat high, val flat low | underfitting from capacity / lr, not epochs |
| \|Δ mAP\| < 1 with mixed class signs | noise (seed noise ≈ ±1 mAP): say "flat" |
| Δ concentrated in one size bucket or class | real but narrow: name it, with its n |

- train.py has no val loss and val is the holdout (= test images). Its curves are train loss vs val mAP/F1;
  yolo26.py's Ultralytics val loss is on the same holdout.
- Compare only equal-epoch runs; otherwise say the comparison is confounded by epochs.
- Bridge and playground sit at 1–3 AP in every run (finding 5): a change there is noteworthy, their absence is not.
- A run without `metrics.json` (stopped, timed out) still has curves; say what is missing and why.
