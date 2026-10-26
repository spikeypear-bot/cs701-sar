---
name: model-planning
description: Use when the user brings an idea for a new model, loss, augmentation, architecture or training change for this SAR detection + classification repo, or explicitly asks Claude for ideas, research or a plan to raise mAP or macro-F1, before any code for it is written.
---

# Model planning

## Overview

Turn an idea into one researched, criticised and costed proposal. The user's ideas and Claude's are treated the same
way: argued for and against on evidence. Claude offers its own ideas only inside this skill, when the user asked for
them.

**REQUIRED:** run superpowers:brainstorming for the dialogue (approaches laid out in their own message, the user picks
one, design doc approved before any code). This skill supplies what the research must cover and what the design doc
contains. Write the design doc to `docs/plans/YYYY-MM-DD-<topic>.md` (private, gitignored) instead of brainstorming's
default path.

## What to look at first

1. `docs/claude.md`: Results so far, Findings, Ideas, **Ruled out**. Do not re-propose a ruled-out idea without new
   evidence that answers why it was ruled out.
2. `docs/experiments.md` Decisions register: an idea touching a decision with status parked / rejected is a
   re-review of that decision; say which one and whether its "revisit if" condition is met.
3. The code the idea touches (`train.py`, `sarbench/*.py`, `yolo26.py`), so the cost estimate is real.

## Research

- Search for papers on the same problem: tiny objects (half the boxes < 16 px), SAR speckle, one class per image,
  detection + classification heads sharing a backbone, YOLO / DETR training recipes. Prefer papers with code and
  numbers on SAR or aerial datasets (SSDD, HRSID, SARDet-100K, DOTA, AI-TOD).
- For each relevant paper: what it did, the gain it reported, on what data, and why it may or may not transfer here.
  Link it. Mark numbers taken from a paper as "reported", not checked.
- Find 2–3 alternatives that target the same weakness differently.

## Criticise, then argue for it

- **Against:** rules (one model, ≤ 512 px once, no TTA / ensembles, no SAR-pretrained weights, ≤ 130 M total / ≤ 40 M
  trainable), compute (hours vs the 1-day job limit and the 2 Nov test release), earlier runs that tested something
  close, whether the expected gain is above seed noise (±1 mAP), which classes / sizes could get worse.
- **For:** the strongest honest case: what mechanism makes it work here, the best evidence, the cheapest test that
  would show it.
- If the case against wins, say so plainly and recommend not building it. That is a valid outcome.

## The design doc and final message, in this order

```markdown
## Plan: <topic>

### Final improvement proposed
<one paragraph: the change, the run(s) that test it, and the baseline at equal epochs>

### Why
- Weakness it targets: <with numbers from our runs>
- Evidence: <papers (links, "reported" gains), our earlier runs>
- Expected effect: <metric, classes / sizes, size of effect vs ±1 mAP noise>
- Against / risks: <the strongest objections and how the run would reveal them>
- Alternatives considered: <each in one line, and why not chosen>

### Changes, per file
| file | change |
|---|---|
| `sarbench/x.py` | <new function / class, what it does> |
| `train.py` | <new flag `--x`, default off, wired at …> |
| `tests/test_x.py` | <what the test checks> |
| `docs/RUNS.md`, `docs/claude.md`, `docs/DETAILS.md` | <lines added> |

### Cost
Parameters (total / trainable), compute hours, GPU type: <estimates, with how they were estimated>
```

After the user approves: record the plan as "proposed" with the results-recorder skill, then
superpowers:writing-plans for the implementation. When the code is ready, the model-summary skill produces the
run summary.
