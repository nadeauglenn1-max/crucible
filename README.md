# Crucible

> **Provisional codename.** A *crucible* is both the vessel that contains and the
> trial that transforms — which is exactly what an environment does to an agent.
> The name may change.

**Crucible turns any real software into a trainable, gradable world for AI agents.**

Wrap a CLI, a database, a codebase, or an API in a few lines and it becomes an
*environment*: an agent can be run through it, scored on real outcomes, and — the
part nobody else has — its whole episode **replayed deterministically**, so every
run is reproducible and every reward is auditable.

## Why this exists

The frontier of AI has moved off pre-training. Models now improve by **doing** —
reinforcement learning in environments — and the whole field agrees on the
bottleneck: *"RL environments are the key bottleneck to the next wave of AI
progress, but big labs are locking them down."* Environments are the **training
data of 2026**, and there is no open, easy way to *make* them. Writing one takes
days; reward functions are brittle; most are too narrow to matter.

Crucible is the missing authoring layer:

- **Environments as code.** Wrap real software; don't hand-build a simulator.
- **Verifiable rewards.** Reward comes from programmatic checks on real state
  (tests pass, query returns the right rows) — not a learned reward model — so it
  is compatible with RLVR / GRPO-style training out of the box.
- **Deterministic replay.** Every episode records a trajectory that re-runs
  byte-for-byte against a fresh environment. Reproducible training, auditable
  rewards, regression-testable environments.

## What it is *not*

Crucible is **training infrastructure** — the Gymnasium / Prime-Intellect lineage:
a place to *make* and *run* environments. It is not a runtime agent-accountability
or governance system. That boundary is deliberate and kept in the code.

## Status

**V1 complete** — author → run → grade → replay → persist, as a real tool. Core
(`Environment`, `Trajectory`, `rollout`, `replay`), trajectory persistence
(versioned on-disk format), a `crucible` CLI, and three example environments: a
deterministic guessing game, a **real SQLite** SQL task, and a **code task graded by
its own tests**. Python 3.11+, zero-dependency core, **100% coverage**, CI gate.
See [`docs/VISION.md`](docs/VISION.md). Next up is the fun (a TRL/verifiers export, a
Space, grader sandbox, the trajectory commons).

```bash
python -m examples.demo             # forge agents through all three worlds, replay each
crucible show episode.trajectory.json   # summarize + integrity-check a saved episode
```
