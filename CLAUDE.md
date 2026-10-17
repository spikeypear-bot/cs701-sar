# Instructions for Claude

- **Never run tests, training, prediction or any other model code on this machine** (no `pytest`, no `python
  train.py`, no smoke runs, not even "small" CPU-only tests): the CPU cannot handle it. Review code by reading it.
  Tests and runs happen only on the cluster, by the user. When a change needs testing, give the exact command for the
  user to run there.
- Research notes, results and cluster commands: `docs/claude.md` (private, gitignored).
- Add the sbatch command for each new experiment to `docs/RUNS.md` (one commented line per run, nothing else).
