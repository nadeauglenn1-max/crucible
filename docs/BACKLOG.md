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
byte-for-byte through the sandbox. 100% covered.

**Second adapter shipped — `DockerSandbox`** (container isolation: no network, memory
+ pids caps, ephemeral container). Fails closed when Docker is unavailable. Command
construction and the docker-unavailable path are unit-tested; the container execution
needs a running Docker daemon and is validated against a real daemon (marked
`pragma: no cover`, not run in unit CI). Two adapters behind one seam — the plugin law
holds.

### 2. Training export  ·  *the adoption keystone*  ·  *core done; trainer adapter open*

- [x] **Format-agnostic core** — `crucible/export.py`: `to_records(traj)` flattens a
      trajectory into `{env_id, seed, step, prompt, completion, reward}` records (the
      per-step triple every RL trainer wants); `write_jsonl(trajs, path)` writes a
      training dataset. A `render_observation` / `render_action` hook turns rich
      dict observations into the text a trainer expects (default: strings as-is,
      else canonical JSON). Plain dicts/JSONL ⇒ zero dependency. 100% covered.
- [x] **TRL adapter** — `crucible/integrations/trl.py`, built against TRL's *current*
      GRPO API (which is **online**: it generates completions, so it needs a prompt
      dataset + a reward function). `to_prompt_dataset(trajs)` builds the
      `[{"prompt", "env_id"}]` rows; `env_reward_func(env_factory, parse_completion)`
      turns a Crucible **environment into a TRL reward function** — each completion is
      parsed to an action and the env scores it verifiably. Zero-dep and tested
      without `trl` installed (it only produces TRL-shaped values). 100% covered.
- [x] **The learning loop, proven on CPU** — `examples/learn.py` +
      `tests/test_learning.py`: a policy learns a task purely from a Crucible
      environment's reward (random baseline 0.26 → learned 0.94) — the same
      generate → grade → reinforce loop GRPO runs at scale, small enough to run on a
      laptop and assert in CI.
- [x] **The LLM-scale GRPO run** — the credibility demo, done and verified on a GPU:
      [`examples/train_grpo.py`](../examples/train_grpo.py) fine-tunes
      `Qwen2.5-0.5B-Instruct` (LoRA, bf16) with real `trl.GRPOTrainer`, the reward
      being a `SQLTaskEnv` and nothing else. It took a second-highest-salary SQL task
      from **5% → 100%** in 80 steps on an RTX 5070 (8GB) — the model discovered
      `... ORDER BY salary DESC LIMIT 1, 1` on its own. Result, chart, and one-command
      reproduction in [`examples/results/`](../examples/results/README.md). Needs `trl`
      + a GPU, so it's a documented example, not a CI test.
- [x] **Generalization battery** — [`examples/train_grpo_suite.py`](../examples/train_grpo_suite.py)
      answers "is it a one-off?": a *fresh* 0.5B model trained on **four distinct SQL
      skills** (subquery/OFFSET, `GROUP BY`+`SUM`, `HAVING`, `AVG`) improved on every
      one (15→90, 25→55, 65→70, 40→85 %). The method trains, not one lucky task. The
      task ground truth is CPU-tested in `tests/test_train_grpo_suite.py`; the run
      needs a GPU.
- [x] **Cross-modality battery** — [`examples/train_xmodal.py`](../examples/train_xmodal.py)
      answers "is it just SQL?": the identical loop over **three different environment
      types** on `Qwen2.5-1.5B` — `CommandEnv` (shell) 70→100, `CodeTaskEnv` (code,
      graded by real execution) 55→85, `SQLTaskEnv` (a hard correlated subquery) 25→35.
      The two non-SQL agents climbed +30; only the environment changes. Ground truth
      CPU-tested in `tests/test_train_xmodal.py`; the run needs a GPU.
- [x] **verifiers adapter** — `crucible/integrations/verifiers.py`, built against
      Prime Intellect's verifiers API (reward funcs return a float, combined by a
      `Rubric`). `env_reward_fn(env_factory)` turns a Crucible environment into a
      verifiers reward function: extract the completion (chat-style or plain),
      parse to an action, step a fresh env, return the reward. Zero-dep, tested
      without `verifiers`. 100% covered.
- [ ] **prime-rl adapter** — prime-rl consumes verifiers environments, so this is
      largely downstream of the verifiers adapter; add the thin glue when a user runs
      it.

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

### 5. More environment wrappers  ·  *CLI done; HTTP + git-repo open*

- [x] **Subprocess/CLI env** — `envs/command.py` `CommandEnv`: the agent emits a
      command (argv), reward is exit-0 **and** stdout matches expected. Reuses the
      sandbox, is registerable + CLI-replayable (whole config is data), replays
      byte-for-byte. 100% covered.
- [x] **Stateful terminal env** — `envs/terminal.py` `TerminalEnv`: a persistent
      shell session where commands accumulate state across steps (mkdir here, write
      there); reward is a goal over the workdir files. The terminal-agent shape;
      replays byte-for-byte for deterministic command sequences. 100% covered.
- [x] **HTTP/API env** — `envs/http.py` `HttpTaskEnv`: wraps a **recording**
      (request → response, the VCR/cassette pattern), so it's deterministic and
      registerable. The agent's action is a request `{"method","path"}`; reward when
      the response body matches expected; unknown paths return a recorded 404, not a
      crash. Wrapping a live WSGI app behind the same shape is a later option. 100%
      covered.
- [x] **git-repo-with-pytest** — done by composition (no new code): `CodeTaskEnv`
      over a repo (`calc.py` + `test_calc.py`) with `command_grader(["python","-m",
      "pytest","-q"])`. The agent edits the source, **real pytest** runs in the
      sandbox, and green is the reward — proven end-to-end and replayable in
      `tests/test_pytest_repo.py`. Reuse over rebuild: the shape everyone wants falls
      out of the pieces already built.
- **Why.** Reach — each wrapper turns a whole class of existing software into
  environments with near-zero code. Pattern is always the same: `reset` fixes initial
  state, `step` applies the action to the real system, reward is a programmatic check.

### 6. Reward composition (rubrics) ✅ done

Shipped: `crucible/reward.py` — `rubric(("name", weight, check), ...)` builds a
`Rubric`; `score(state) -> (float, breakdown)` returns the weighted fraction passed
in [0, 1] plus a per-criterion `name → passed` map for `info`. Validated (non-empty,
non-negative weights, positive total). An environment scores partial credit in `step`
and still replays byte-for-byte (proven). Stays strictly programmatic — learned
reward for non-verifiable tasks remains parked (see below). 100% covered.

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
- **A hosted training service.** The core is the open authoring layer; a managed
  "point it at your software, get a trained adapter back" service is the eventual
  business. Deliberately parked until the open tool has traction.

_Resolved: the repo is **public under MIT** and the packaging is PyPI-ready — the
earlier "publish / license" question is settled._
