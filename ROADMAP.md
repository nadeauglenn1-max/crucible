# Crucible — Roadmap

Where this is going. The *why* is in [`docs/VISION.md`](docs/VISION.md); the granular
"what's built / how to build the next item" is in [`docs/BACKLOG.md`](docs/BACKLOG.md).
This file is the middle layer: the direction, in three phases, and the honest reasoning
behind the order.

Crucible is built the way good infrastructure is built — smallest complete thing first,
one code path, tests and docs moving with the code. So the roadmap is deliberately
sequenced, not a wish list. Each phase earns the next.

---

## Where we are

**Phase 0 — the tool works, end to end.** ✅

Author → run → grade → **replay** → export, as a real library: a zero-dependency core,
six environment types (SQL, code/pytest, shell command, stateful terminal, recorded
HTTP, guessing game), sandboxed grading, a registry, rubrics, and adapters that turn any
Crucible environment into a reward function for TRL and Prime Intellect's `verifiers`.
100% test coverage, CI on Python 3.11–3.13.

And it doesn't just *grade* — it **trains**. Using a Crucible environment directly as the
reward (no labels, no hand-written reward code), GRPO took a small model from 5% → 100% on
a real SQL task, generalized across four distinct SQL skills, and — the point that matters —
did the same across **three different environment types**: a database agent, a shell agent,
and a coding agent graded by *actually running its code*. The proof, charts, and one-command
reproduction are in [`examples/results/`](examples/results/README.md).

---

## Phase 1 — Adoption: the best open way to *make* an environment

The near-term goal is singular: make Crucible the path of least resistance for anyone
already doing RL who needs an environment. The ecosystem has a hub (Prime Intellect) and
training stacks (TRL, `verifiers`, `prime-rl`); Crucible is the **authoring layer** that
feeds them. We complement that ecosystem, we don't fight it.

- **`prime-rl` adapter** — the last leg of the training-stack triangle (TRL ✅,
  `verifiers` ✅). Then "wrap software → train with whatever stack you already use" is
  true with no glue code.
- **Ship on PyPI** — `pip install crucible-rl` as the frictionless front door.
- **More environment wrappers where the pain is real** — a browser/Playwright
  environment and a full SWE-bench-style repo environment are the two most-requested
  shapes; both fall out of the existing seams.
- **Multi-step / agentic episodes** — today's example tasks are mostly single-step for
  clarity of the training demo, but the trajectory format is already multi-turn. Real
  agents are multi-turn, and that's where the field is heading.
- **A cited reference environment** — one hard, real, *reproducible* task that becomes
  the example people copy.

**How you'll know it's working:** environments authored by people who aren't the
maintainer. Stars are vanity; authored environments are the signal.

---

## Phase 2 — The auditable commons

An open library alone doesn't compound. What compounds is a **shared, growing library of
environments and trajectories that are replay-verifiable** — the one property a plain
dataset dump can't offer. Because a Crucible reward can be *re-checked* by replaying the
episode, a commons of Crucible trajectories is auditable by construction: you can prove a
reward was real, not fabricated or leaked.

- **`crucible push` / `pull`** — read/write the versioned trajectory format to and from a
  Hugging Face dataset repo, with a dataset card that records the environment, the seed,
  and the replay fingerprint.
- **Environment sharing** — publish and pull *environments* (not just episodes) by name,
  building on the existing registry.
- **Replay-verified badges** — a trajectory dataset that anyone can re-run to confirm its
  rewards, turning "trust me" into "check it yourself."

This is the network-effect phase. It's the bet that Crucible is more than a library.

---

## Phase 3 — A hosted option, if the commons earns it

Only once the open tool and commons have traction: a managed path — *point Crucible at
your repo or service, get a trained adapter back* — for people who want the outcome
without running the loop themselves. The open core stays the product; a hosted option
exists to sustain the work, not to gate it. Deliberately last, and deliberately optional.

---

## What Crucible is *not* (the boundary that keeps the roadmap honest)

Crucible is **training and evaluation infrastructure** — the Gymnasium / Prime-Intellect
lineage. Its "verification" is programmatic task-success checking, used for *reward*. It
is deliberately **not** a runtime agent-accountability, trust, or governance system —
"prove what a deployed agent did in production" is a different field, and Crucible will
not drift into it. See [`docs/VISION.md`](docs/VISION.md) §5.

---

## Contributing

The backlog items above are real and mostly independent — a good first contribution is a
new environment wrapper or the `prime-rl` adapter. Every change ships with tests and moves
the docs in the same commit; merges need green CI (3.11–3.13) and a maintainer +1. See
[`CONTRIBUTING.md`](CONTRIBUTING.md) and [`docs/BACKLOG.md`](docs/BACKLOG.md) for the
"how to build it" detail on each item.
