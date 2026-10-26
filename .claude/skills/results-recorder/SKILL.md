---
name: results-recorder
description: Use when a run of this SAR repo is queued, finished, evaluated or submitted to Codabench, when a plan is approved, or when the user decides to adopt, drop, park or revisit an idea, so the result, analysis and decision are kept for the report and the ablation tables.
---

# Results recorder

## Overview

One ledger, `docs/experiments.md` (private, gitignored), holds every run and every decision so the report and its
ablation tables can be written from it. **Every decision is provisional:** it records the evidence it rests on and
the condition that would reopen it. Nothing in the ledger is deleted; a reversed decision gets a new row that points
to the old one.

## What to write, by event

| event | ledger | also update |
|---|---|---|
| plan approved | Run entry, status `proposed` | |
| run queued | Run entry status `queued`, job id, the `sbatch` line | `docs/RUNS.md` (one comment + command) |
| run finished and evaluated | Results + Analysis of the entry, status `done` | `docs/claude.md` Results table row; a Finding if report-worthy |
| run failed / killed | status `failed`, why, what was salvaged | |
| Codabench upload | Codabench line in the entry | `docs/claude.md` Results table |
| decision made or reopened | Decisions register row | memory, if it changes what future sessions should know |
| code / flag added or changed | | `docs/claude.md` "Implemented"; `docs/DETAILS.md` if it documents that part; `CLAUDE.md` if it changes how Claude works here |

## Run entry (under `## Runs`, newest first)

```markdown
### <run name> — <status> (<YYYY-MM-DD>)
- Hypothesis: <from the model summary>
- Change vs <baseline>: <flags>
- Command: `<sbatch line>` · job <id> · commit <hash>(dirty?)
- Parameters: <total> M total, <trainable> M trainable · compute <h> on <GPU>
- Holdout: mAP <x> · AP50 <x> · macro-F1 <x> · dm_rough <x> (baseline: …)
- Codabench: Δm <x> (<date>) | not submitted
- Breakdown: <size and class deltas ≥ 1 point, with n>
- Fit: <under / over / converged, with the numbers>
- Analysis: <what it shows, the likely cause, what it does not show>
- Decision: <D-number>
```

## Decisions register row

`| D<n> | <date> | <decision> | <evidence: run names, numbers> | <status> | <revisit if> |`

Status is one of: `adopted (provisional)`, `open`, `parked`, `rejected (provisional)`, `superseded by D<m>`.
"Revisit if" is an observable condition: a run result, a TA answer, a confound removed (e.g. "a 100-epoch run
exists", "TA allows 800 px"). A decision without one is not finished.

## Rules

- Numbers are copied from the outputs (results.csv, analyze_runs, Codabench), never retyped from memory; state the
  set (holdout vs Codabench val).
- Name confounds next to the decision (unequal epochs, one seed, different GPU) and put their removal in
  "revisit if".
- When a new result touches an existing decision, add a line to that decision's evidence and say whether its
  "revisit if" is now met; tell the user if it is.
- Update `Last updated` at the top of `docs/claude.md` and `docs/experiments.md` when they change.
