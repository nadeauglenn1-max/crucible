# Crucible — Backlog

The single source of truth for what's built and what's next. Built the way we build:
V1 is the **MVP** — the smallest *complete*, production-quality whole on one code
path (not a cut-corner prototype), every default maintained (100% coverage, CI gate,
docs move with code). Then the fun.

Each backlog item below has **What / Why / How** — the *How* is written to
junior-dev detail: the files to touch, the approach, and the gotchas (especially the
determinism contract and security).

---

## V1 — the MVP ✅ complete

- [x] **Core** (`env.py`, `trajectory.py`, `rollout.py`): the `Environment` contract,
      the replayable `Trajectory`, `rollout` (record), `replay` (verify).
- [x] **Example environments** (`envs/`): `GuessEnv` (deterministic replay proof),
      `SQLTaskEnv` (wrap real SQLite → verifiable reward), `CodeTaskEnv` (the test
      suite *is* the reward function).
- [x] **Persistence** (`trajectory.py`): `save`/`load` with a versioned JSON
      envelope.
- [x] **CLI** (`cli.py`): `crucible show <file>` — summarize + integrity-check.
- [x] **CI + coverage gate** (`.github/workflows/ci.yml`): pytest ≥90% on 3.11–3.13.

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for how all of the above works.

---

## Then the fun (post-V1)

Ordered by leverage. Each rides on the solid V1.

### 1. Sandbox the code grader ✅ done  ·  *the public-flip gate*

Shipped in `crucible/sandbox.py`: a `Sandbox` seam (`run(files, command) ->
GradeResult`), a `SubprocessSandbox` adapter (materialize to a temp dir, run the
command in a subprocess with a minimal environment, `PYTHONHASHSEED=0`, and a hard
timeout — fail closed on timeout / crash / launch error), and `command_grader`, which
turns a sandboxed command into a `CodeTaskEnv` grader in one line. The agent's code
now runs in a *child process*, not ours. Proven: pass/fail, a looping submission
contained by timeout, a bad binary → grade-not-crash, and a full episode replays
byte-for-byte through the sandbox. 100% covered. *Second adapter (container /
seccomp / nsjail) slots behind the same seam when it's needed.*

### 2. TRL / verifiers / prime-rl export  ·  *the adoption keystone*

- **What.** Turn a `Trajectory` (or a batch) into the data shape an RL trainer
  consumes, so Crucible episodes feed a *real* training run.
- **Why.** The difference between "starred the repo" and "uses the tool." HF's TRL is
  the socket most researchers already have.
- **How.**
  1. New module `crucible/export.py`. Study the input format of the target
     (`verifiers` environment spec / TRL `GRPOTrainer` dataset / prime-rl rollout).
     They want, per step: prompt/observation, the completion/action, and the
     scalar/binary reward — which is exactly a `Transition`.
  2. Write `to_verifiers(traj) -> dict` and/or `to_trl_dataset(trajs) -> list[dict]`
     that flatten transitions into `{prompt, completion, reward}` records.
  3. Keep it an **optional extra**, not a core dependency: `pip install
     crucible-rl[trl]`. The core stays zero-dep (CONVENTIONS).
  4. **Gotchas:** observations/actions are rich (dicts) but trainers usually want
     text — provide a `render(observation) -> str` hook per environment (default:
     `json.dumps`) so the export is faithful without hard-coding a text shape.
- **Done when:** a Crucible trajectory round-trips into a minimal TRL/verifiers run in
  an example, with a test asserting the exported record shape.

### 3. Environment registry + `crucible replay <file>` ✅ done

Shipped: `crucible/registry.py` (`register(name)` decorator, `make(name, config)`,
`registered()`); `Environment.config()` returns a serializable reconstruction dict;
`rollout` records `env_config` on the trajectory; the on-disk format is now **v2**
(v1 still loads, `env_config` defaults empty); `crucible replay <file>` rebuilds the
env with `make` and re-runs it, printing an OK or the itemized mismatches. `GuessEnv`
and `SQLTaskEnv` are registered; `CodeTaskEnv` (live grader) is library-replayable but
not CLI-replayable, and the CLI says so. 100% covered.

### 4. Crucible Space  ·  *the 1am-star demo*

- **What.** A Hugging Face Space where you drop an agent into an environment in the
  browser and watch it get scored and replayed.
- **Why.** Visibility. Environments aren't yet a first-class HF artifact — this stakes
  the claim.
- **How.** A small Gradio app (`spaces/` dir, separate from the packaged core):
  pick an env, step an agent, render the trajectory, hit "replay" and show the
  `ReplayReport` going green. Reuses `rollout`/`replay` unchanged.

### 5. More environment wrappers

- **What.** Cover the three shapes people actually have: a **subprocess/CLI** env
  (wrap any command-line tool), an **HTTP/API** env (wrap a service), and a real
  **git-repo-with-pytest** env (builds on `CodeTaskEnv` + the §1 sandbox).
- **Why.** Reach. Each wrapper turns a whole class of existing software into
  environments with near-zero code — the "wrap what you have" promise at scale.
- **How.** One module per wrapper in `envs/`. The pattern is always the same:
  `reset` establishes fixed initial state, `step` applies the action to the real
  system, reward comes from a programmatic check. **Gotcha:** the HTTP env is the
  hardest to make deterministic (external state) — start with a recorded/mock backend
  and document the determinism caveat honestly (`replay` will expose any drift).

### 6. Reward composition (rubrics)

- **What.** Build a reward from multiple weighted checks instead of one boolean.
- **Why.** Real tasks are partial-credit; a rubric (tests pass **and** style **and**
  no secrets leaked) is closer to how work is graded.
- **How.** A `Rubric` helper (`crucible/reward.py`): a list of `(name, weight, check)`;
  `score(state) -> (float, breakdown)`. Environments call it in `step`; put the
  `breakdown` in `info`, keep the scalar as `reward`. **Gotcha:** each check must be
  deterministic, or the composed reward isn't replayable. Do **not** wander into
  *learned* reward for non-verifiable tasks yet — that's a careful research design,
  flagged separately, and stays out until it's earned.

### 7. The trajectory commons

- **What.** Trajectories as shareable, versioned datasets ("GitHub for agent
  experience").
- **Why.** The network-effect endgame: a growing library of open, *auditable*
  episodes (auditable because replay makes the reward checkable — the thing no
  trace-dataset today offers).
- **How.** A `crucible push/pull` that reads/writes the versioned trajectory format
  to/from an HF dataset repo (via `huggingface_hub`, an optional extra). Ship a
  dataset card template that records the env, the seed, and the fingerprint.

---

## Not yet (parked deliberately)

- **Learned reward for non-verifiable tasks.** The genuinely hard frontier (support,
  RAG, open-ended work). Stay programmatic until it's designed carefully — a bad
  learned reward is worse than no environment. Also watch the provenance boundary:
  "grade the quality of an agent's judgment" drifts toward the excluded field.
- **Publishing / licensing.** Repo is local. When it opens: Apache-2.0 (adoption) vs.
  BSL (protect the hosted business) vs. dual — decided at publication.
