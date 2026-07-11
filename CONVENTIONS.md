# Crucible — Conventions

Standing conventions apply (reuse over rebuild, fix-don't-flag, docs-never-lag-code,
no-MVP production bar, clean-room provenance, tests ship with the change).
Project-specific notes:

## Core stays tiny and dependency-free

The `crucible` core (`env`, `trajectory`, `rollout`, `replay`) imports only the
Python standard library. Environments and agents may pull dependencies; the core
may not. Adoption is a feature, and a zero-dependency core is how you get it.

## The determinism contract is law

`reset(seed)` fully determines an episode. Same seed + same actions ⇒ same
observations, rewards, and digests. `replay` verifies this and **reports mismatches
loudly** — a non-reproducible environment is a bug we surface, never one we hide.

## Provenance boundary (read `docs/VISION.md` §5)

Crucible is training/eval infrastructure. Reward = **programmatic task-success
checking**. Do **not** drift into runtime agent-accountability, tamper-evident
trust, or governance — that is a different field and the author's separate IP lives
there. If a design starts to look like "prove what a deployed agent did," stop.

## Quality gates

- **Coverage ≥ 90%** on the core + envs, enforced in CI; tests ship with the code.
- Deterministic tests only — no flaky sleeps; drive time/seeds explicitly.
- Docs move with the code (README, `docs/VISION.md`) in the same change.

## Provenance

Clean-room: personal account and environment only, its own history, no code from
the author's other projects. Licensed **MIT**; public. Contributions merge only with
green CI plus a maintainer +1 (CODEOWNERS + branch protection) — see
[`CONTRIBUTING.md`](CONTRIBUTING.md).
