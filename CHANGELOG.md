# Changelog

Notable changes to `crucible-rl`. Dates are release dates.

The rule this file follows: **a change to what the library GUARANTEES is a
headline, not a bullet under "fixes".** A user upgrading needs to know whether
something that passed yesterday fails today, and a version number is the only
place most of them will look.

## 0.2.0 — 2026-07-22

Every claim a trajectory makes is now bound by `replay`. Previously several were
recorded and verified by nothing, which meant a tampered file could pass.

Found by [@sterngold](https://github.com/sterngold) in
[#14](https://github.com/nadeauglenn1-max/crucible/issues/14) — an external user
who cloned the repo, built a 13-class tamper matrix against it, and reported
exactly which fields survived tampering. That is the discovery mechanism worth
recording: the first person outside the project found this, not the project's own
review pass.

### Breaking

- **`replay` now binds step `info`.** An environment that puts nondeterministic
  data in `info` — a timestamp, a wall-clock duration, a random id — now FAILS
  replay where it previously passed. This was always required: the determinism
  contract says everything recorded must be a pure function of seed and actions,
  and `info` is where a `Rubric`'s breakdown lives, which is the evidence a
  trajectory carries. Every environment shipped in-tree already satisfied it.

  If you have an environment that fails on this, the data belongs outside the
  graded record, not inside an unverified field.

- **`replay` now binds environment identity and config.** `env.name()` must equal
  `traj.env_id` and `env.config()` must equal `traj.env_config`. Replaying a
  trajectory against a differently-configured world used to return a green tick.

### Added

- **Trajectory format v3** records `final_observation`, the last thing the agent
  saw. v1 and v2 dropped it entirely. Old files still load; the missing field
  reads back as the `UNRECORDED` sentinel rather than `None`, so `replay` does not
  invent a verdict about a value the record never carried.
- `Trajectory.integrity_mismatches()` — the single implementation of the
  self-checks, called by both `crucible show` and `replay`.
- `UNRECORDED` is exported from the package root.

### Fixed

- **`crucible show` and `crucible replay` could disagree about the same file.**
  `show` re-derived `total_reward` and reported MISMATCH; `replay` did not check
  it and printed "reproduced OK" with exit 0. One guarantee had two
  implementations, and the weaker one was in `replay` — the command the whole
  library rests on. There is now one implementation and `replay` is strictly
  stronger.
- `docs/ARCHITECTURE.md` claimed byte-for-byte replay was "a claim about the
  entire episode, not just the numbers." It was not, in precisely the gap
  reported. The docs were overclaiming and are corrected.

## 0.1.0 — 2026-07-11

First public release.
