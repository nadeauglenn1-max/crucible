# Crucible — Vision & Design

The design of record. Code follows it, not the other way around.

## 1. The bet

AI's center of gravity has moved from **capability** (pre-training a smarter model)
to **diffusion and experience** (making models improve by *doing*). The dominant
mechanism is reinforcement learning in **environments**, and the field's own
consensus is that environments — not compute, not model size — are the binding
bottleneck of the next wave. Big labs are hoarding proprietary environments.

The open ecosystem has a **hub** (Prime Intellect's Environments Hub — distribution)
and a **training stack** (verifiers, prime-rl, TRL). It does **not** have a great
open **authoring layer**: the thing that makes a rich, verifiable, reproducible
environment cheap to create. Writing environments and reward functions today is
slow, manual, and brittle. That is the seam Crucible fills.

## 2. The one idea

> **An environment is real software plus a way to score outcomes and replay them.**

Instead of hand-building a bespoke simulator, you *wrap software you already have*
— a CLI, a database, a repo, an API — and Crucible gives it the three properties a
trainable environment needs:

1. **A rollout protocol** (`reset` / `step`) an agent can be driven through.
2. **Verifiable reward** from programmatic checks on real state (tests, assertions,
   query results) — not a learned reward model — so it plugs into RLVR/GRPO.
3. **Deterministic replay** — every episode is a trajectory that re-runs
   byte-for-byte against a fresh environment.

## 3. Why deterministic replay is the wedge (not a nice-to-have)

Reproducibility is the unglamorous property that makes the rest trustworthy, and
almost no environment framework has it end-to-end:

- **Reproducible training data.** A trajectory is a self-contained, re-runnable
  record — shareable, cacheable, diffable.
- **Auditable reward.** Anyone can replay an episode and confirm the reward was
  real, not fabricated or leaked. (Programmatic reward + replay = a reward you can
  *check*, which matters as environments become an economy.)
- **Regression-testable environments.** An environment change that alters old
  rewards is caught by replaying a frozen trajectory — the same pin-the-golden
  discipline good infra always has.

## 4. Architecture in one breath

A tiny, dependency-free **core** — `Environment` (the `reset`/`step` contract),
`Trajectory` (the replayable record), `rollout` (drive an agent, record), `replay`
(re-run and verify) — and everything else is an **environment** or an **agent**
that plugs in. Environments are ordinary Python wrapping ordinary software; agents
are anything with an `act(observation) -> action`.

**Determinism contract:** `reset(seed)` fully determines an episode. Same seed +
same actions ⇒ same observations, rewards, and state digests. An environment that
can't honor this isn't replayable, and Crucible says so loudly (replay reports the
mismatch) rather than hiding it.

## 5. The boundary (provenance — read this)

Crucible is **training/eval infrastructure**: making and running environments to
*generate learning signal*. It is explicitly **not** a runtime
agent-accountability, trust, or governance system — that is a different field (and
the author's separate IP lives there). The "verification" in Crucible is
**programmatic task-success checking for reward** (did the tests pass, did the query
return the right rows), the decades-old RL-reward lineage — not tamper-evident
accountability of a deployed agent. Keep v1 rewards programmatic/verifiable; do not
drift into learned-trust or runtime-governance territory.

## 6. The arc (why open source)

The MySQL pattern: ship something good and free that a new platform wave needs, let
developers adopt it bottom-up, let a company condense out of the adoption.

- **Adoption:** Python-first, zero-dep core, wrap-what-you-have ergonomics → RL
  researchers, indie fine-tuners, and eval builders pick it up.
- **Community:** environments are contributions — a library of open, verifiable,
  replayable worlds grows like a package registry.
- **Company:** the hosted layer — run rollouts at scale, store/serve/replay
  trajectories, curate and rank environments, connect to training. Monetize the
  cloud, not the core.
- **Licensing decision (open):** Apache-2.0 for max adoption vs. source-available
  (BSL, HashiCorp/Cockroach-style) to protect the hosted business vs. dual-license
  (the MySQL move). Deferred until publication.

## 7. Roadmap (first bricks)

- [x] Core: `Environment`, `Trajectory`, `rollout`, `replay`.
- [x] Example envs: deterministic `GuessEnv`; real-SQLite `SQLTaskEnv`
      (wrap-real-software → verifiable reward).
- [ ] Trajectory on-disk format + `crucible replay <file>` CLI.
- [ ] A `verifiers`/prime-rl-compatible export so trajectories feed real training.
- [ ] Environment "wrappers" for the big three: a subprocess/CLI env, an HTTP/API
      env, a git-repo-with-tests env (the SWE-agent shape).
- [ ] Reward composition (rubrics) + the honest hard part: signal for
      non-verifiable tasks — *stay programmatic until this is designed carefully.*
