---
name: model-summary
description: Use when a new model, architecture change, loss, augmentation, training flag or recipe change in this SAR detection + classification repo is ready to be sent to the GPU cluster for training, or when the user asks what a proposed or queued run changes, how many parameters it trains, or how to launch it.
---

# Model summary

## Overview

Every run that goes to the cluster gets one summary, in the fixed shape below, before the user submits it. The user
reads it to decide whether to queue the run and later copies parts into the report, so it states facts with their
sources (file:line, paper, earlier run) and nothing else.

Never run code on this machine (CLAUDE.md): parameter counts come from reading the code, and the summary gives the
command that confirms them on the cluster.

## The summary, in this order

````markdown
## Model summary: <run name>

**Baseline:** <run it is compared with, its holdout mAP / macro-F1 / dm_rough, same epochs as this run>

### What changes
- <one bullet per change, with the flag(s): `--x 0.5`; "everything else as <baseline>">

### Why
- Hypothesis: <what should improve, on which classes / box sizes, and by roughly how much>
- Evidence: <earlier run, finding number in docs/claude.md, or paper (link)>
- Risk: <what could go down, e.g. tiny AP, F1, speed>

### Parameters and size
| | this run | baseline | limit |
|---|---:|---:|---:|
| total parameters | <M> | <M> | 130 M |
| trainable (frozen excluded, heads included) | <M> | <M> | 40 M |
| model.pt (fp32, ≈ 4 B × total) | <MB> | <MB> | |
| compute | <h on which GPU, from a similar earlier run> | <h> | 1 day/job |

How the numbers were derived: <the arithmetic, e.g. "v8m 25.9 M + new head 0.3 M (sarbench/yolo.py:120)">.
Confirm on the cluster: `grep "parameters:" runs/<run>/log.txt` (train.py) or
`grep "parameters:" logs/sar-train_<jobid>.out` (yolo26.py), a few minutes after the job starts; both scripts abort
if over the limits.

### Files changed
| file | change |
|---|---|
| `path/file.py:line` | <what and why> |

### Rules check
One model · input ≤ 512 px, once · no TTA / ensembles · no SAR-pretrained weights · ≤ 130 M / ≤ 40 M:
<pass, or which item is unclear and the question for the TA>

### Commands (cluster, repo folder)
```bash
git pull
# tests of the changed code first, in a GPU shell (srun ... --pty bash; see docs/claude.md "Commands")
$PY -m pytest tests/<test files touching the change> -v
# the run (also added to docs/RUNS.md)
sbatch <options> train.slurm <args> --name <run name>
# 5 minutes in: it started, loss is finite, parameter count as above
tail -n 20 logs/sar-train_<jobid>.out
```
````

## Rules for filling it in

- **Baseline at equal epochs.** Compare a 48-epoch run with a 48-epoch baseline; epochs alone moved mAP by ~3
  points (e48 41.9 → e100 44.8). If no equal-epoch baseline exists, say so and name the closest.
- **Run name** follows the existing pattern (`yolov8m-<change>-e48`) and is unique in `runs/results.csv` and
  `docs/RUNS.md`.
- **Expected effect vs noise:** seed noise is about ±1 mAP. If the hypothesis predicts less than that, say the run
  cannot confirm it alone (seeds needed).
- **Uncommitted code:** if the change is not pushed yet, the first command line is the push (after the user's OK,
  per the review-before-push rule), then `git pull` on the cluster.
- **Write the `sbatch` line into `docs/RUNS.md`** (one comment line + one command, nothing else), and record the
  run as "queued" with the results-recorder skill.
- Unknown numbers (e.g. compute for a new architecture) are written as an estimate with how it was estimated, never
  left out.
