# Instructions for Claude

- **Never run tests, training, prediction or any other model code on this machine** (no `pytest`, no `python
  train.py`, no smoke runs, not even "small" CPU-only tests): the CPU cannot handle it. Review code by reading it.
  Tests and runs happen only on the cluster, by the user. When a change needs testing, give the exact command for the
  user to run there.
- Research notes, results and cluster commands: `docs/claude.md` (private, gitignored).
- Add the sbatch command for each new experiment to `docs/RUNS.md` (one commented line per run, nothing else).
- Runs, results and decisions (all provisional, each with a "revisit if"): `docs/experiments.md` (private,
  gitignored). Approved plans: `docs/plans/` (private, gitignored).
- **Keep the docs in step with every change:** a new or changed flag, script or file updates `docs/claude.md`
  (Implemented / Commands) and, where it documents that part, `docs/DETAILS.md` or `README.md`; a run, result or
  decision updates `docs/experiments.md`; a change to how Claude works here updates this file.

## Skills (`.claude/skills/`)

| skill | when |
|---|---|
| `model-planning` | a new model idea (the user's, or Claude's when asked): research, criticism, final proposal per file |
| `model-summary` | a change is ready for the cluster: what, why, parameters, files, commands |
| `model-evaluation` | a run finished: commands for `analyze_runs.py`, `class_errors.py`, `plot_run.py`, then the review |
| `results-recorder` | a run is queued / done / submitted, or a decision is made or reopened |

Order for one experiment: model-planning → (implement) → model-summary → run → model-evaluation → results-recorder.
