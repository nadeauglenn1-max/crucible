# Crucible — How it works

A junior-developer's guide to the built system: the mental model, how each piece
works, and how to add your own environment or agent. If you want the *why*, read
[`VISION.md`](VISION.md); this is the *how*.

---

## 1. The mental model

Three nouns and two verbs. That's the whole core.

| Piece | What it is | Analogy |
| --- | --- | --- |
| **Environment** | Real software, wrapped so an agent can act in it and be scored | the game / the exam |
| **Agent** | Anything that chooses an action given an observation | the player / the student |
| **Trajectory** | The recorded episode: seed, observations, actions, rewards | the exam booklet |
| `rollout` | Drives an agent through an environment, records a Trajectory | sitting the exam |
| `replay` | Re-runs a Trajectory against a fresh environment and verifies it | re-grading the booklet |

The core (`crucible/env.py`, `trajectory.py`, `rollout.py`) is **~180 lines and
imports only the Python standard library.** Everything else is an environment or an
agent that plugs into it.

## 2. The Environment contract (`crucible/env.py`)

An environment implements two methods:

```python
class MyEnv(Environment):
    def reset(self, seed: int) -> Observation:
        # start a fresh episode, fully determined by `seed`; return what the agent sees
        ...
    def step(self, action: Action) -> StepResult:
        # apply one action, return StepResult(observation, reward, done, info)
        ...
```

- **`Observation` and `Action` are `Any`** — deliberately rich. A guess is an int, a
  SQL task's observation is `{"task": ..., "schema": ...}`, a code task's action is
  `{"path": "new file contents"}`. The only constraint: they must be
  **JSON-serializable**, because the trajectory records them (see §5).
- **`StepResult`** is a small dataclass: `observation` (what the agent sees next),
  `reward` (a float), `done` (episode over?), and `info` (a dict for anything that
  must *not* influence reward — debug data, attempt counts).

### The determinism contract (this is the important part)

> `reset(seed)` fully determines the episode. The same seed followed by the same
> actions must reproduce the same observations, rewards, and digests.

This is what makes an episode *replayable*. Practically, it means: put every source
of randomness behind the seed (`random.Random(seed)`, not the global `random`), and
never let wall-clock time, network nondeterminism, or dict-ordering leak into
rewards. An environment that can't honor this isn't a bug we hide — `replay` will
report the mismatch loudly.

### Digests (`Environment.digest`)

`digest()` returns a short, stable hash of the environment's *hidden* state. It's
optional — the base class returns `""` — but it makes replay **strict**: replay
compares the digest at every step, so a divergence in internal state is caught even
if the observation and reward happen to match. Rule of thumb: **hash whatever the
observation doesn't already reveal.** `GuessEnv` hashes the secret + counters;
`SQLTaskEnv` hashes the task + solved-flag + attempt count.

## 3. The rollout loop (`crucible/rollout.py`)

`rollout(env, agent, *, seed, max_steps)` is the reset/act/step cycle, and its job is
to *record*:

```
agent.reset()
observation = env.reset(seed)
loop up to max_steps:
    action  = agent.act(observation)          # the agent decides
    result  = env.step(action)                # the world responds
    record Transition(observation, action,    # <-- the observation the agent ACTED ON
                      result.reward, result.done, result.info,
                      env.digest())            # <-- env state AFTER the step
    observation = result.observation
    if result.done: break
return Trajectory(...)
```

Two details worth internalizing:
- The recorded `observation` is the one the agent **acted on**, not the one that came
  back — so a transition is a complete "here's what I saw, here's what I did, here's
  what happened" tuple.
- The `digest` recorded is taken **after** the step, so it captures the state the
  action produced.

## 4. Replay: the verifier (`crucible/rollout.py`)

`replay(env, traj)` is the wedge that makes Crucible more than a logger. It takes a
*fresh* environment and re-drives it with the trajectory's recorded actions:

```
check traj.total_reward == sum of step rewards   # the record, checked against itself
check env.name()   == traj.env_id                # ... in the world it claims
check env.config() == traj.env_config
observation = env.reset(traj.seed)            # same seed
check observation == traj.initial_observation
for each recorded transition t:
    check observation == t.observation        # the obs the agent acted on reproduces
    result = env.step(t.action)               # same action
    check result.reward == t.reward
    check result.done   == t.done
    check result.info   == t.info             # the evidence, not just the score
    check env.digest()  == t.digest
    observation = result.observation
    if result.done: check we're at the last transition (else "ended early")
check observation == traj.final_observation      # how the episode actually ended
return ReplayReport(ok, steps, mismatches)
```

Every check that fails appends a human-readable string to `mismatches`; `ok` is
`True` only when the list is empty. This is what "reproducible training data and
auditable reward" means concretely: **anyone with the trajectory and the environment
can re-derive the reward and confirm it wasn't fabricated.**

### The governing rule: every recorded claim is bound

A trajectory field that replay does not verify is not a harmless omission — it is
precisely the field worth tampering with, because it travels with the artifact
wearing the artifact's credibility. So the rule is that **replay binds everything the
record claims**: the world it says it ran in, the numbers it reports, the `info`
evidence behind those numbers, and the observations at both ends of the episode.

This was not true until [issue #14](https://github.com/nadeauglenn1-max/crucible/issues/14):
`info`, `total_reward`, `env_id`, and `env_config` were recorded but unchecked, and
the final observation was never recorded at all. A trajectory with rewritten evidence
replayed green. It doesn't now — but the general lesson is the one to keep: when you
add a field to the trajectory, you have added something replay must verify.

## 5. The Trajectory format (`crucible/trajectory.py`)

A `Trajectory` is a dataclass: `env_id`, `seed`, `env_config`, `initial_observation`,
a list of `Transition`, `total_reward` (kept in sync by `add`), and
`final_observation` (the last thing the agent saw). It knows how to:

- **Serialize** — `to_json()` uses `sort_keys=True`, so the encoding is *canonical*
  (stable byte-for-byte regardless of insertion order). That's what makes the
  fingerprint meaningful.
- **Fingerprint** — `fingerprint()` is `sha256(to_json())`: a stable id for this
  exact episode. Same fingerprint ⇒ same episode.
- **Persist** — `save(path)` writes a **versioned envelope**:
  `{"version": FORMAT_VERSION, "trajectory": {...}}`. `load(path)` refuses an
  unrecognized `version` rather than silently misreading an old file. This is why the
  artifact can leave memory and still be trusted later. v1 → v2 added `env_config`;
  v2 → v3 added `final_observation`. Old files still load: a field the record never
  carried reads back as `UNRECORDED` (a distinct marker, *not* `None`, which is a
  perfectly good observation) and replay checks nothing there rather than inventing a
  value to have an opinion about.
- **Self-check** — `integrity_mismatches()` is everything the record can verify about
  itself with no environment in hand (today: `total_reward` equals the sum of the
  step rewards). It lives on the trajectory, once, and *both* `crucible show` and
  `replay` call it — so the two can never return opposite verdicts on one file.

## 6. The CLI (`crucible/cli.py`)

- **`crucible show <file>`** loads a saved trajectory, prints a summary (env, seed,
  steps, total reward, fingerprint), and runs `integrity_mismatches()` — the checks
  that need no environment. A self-contained sniff test.
- **`crucible replay <file>`** is strictly stronger: it runs *the same* integrity
  check and then re-derives the whole episode. `show` never fails a file `replay`
  passes. It rebuilds the environment from
  the registry (`make(traj.env_id, traj.env_config)`) and calls `replay`, printing
  "reproduced OK" or the itemized mismatches. Works for **registered** environments
  whose full state is a serializable `config` (§6c); an environment carrying a live
  callable is library-replayable (`crucible.replay`, env in hand) but not
  CLI-replayable, and the CLI says so.

## 6c. The environment registry (`crucible/registry.py`)

For `crucible replay` to rebuild an environment from a file, the environment must be
reconstructable from data:

- **`@register("name")`** on the class registers a factory and stamps `env_id =
  "name"`, so trajectories carry the registered name.
- **`Environment.config()`** returns a JSON-serializable dict of the constructor
  arguments. `rollout` records it (trajectory format v2); `make(name, config)`
  rebuilds an identical environment.

An environment whose behavior lives in a callable (like `CodeTaskEnv`'s grader) can't
express itself as data, so it leaves `config()` empty and stays out of the registry —
honestly, rather than pretending. The trajectory format is versioned precisely so this
addition (v1 → v2) doesn't break old files.

---

## 6b. Sandboxed grading (`crucible/sandbox.py`)

`CodeTaskEnv`'s default grader runs in-process (imports the file, calls it) — fine
for trusted, scripted graders, reckless against a real model. The safe path runs the
check in a **subprocess**:

```python
from crucible import SubprocessSandbox, command_grader
from crucible.envs import CodeTaskEnv

check = ["python", "-m", "pytest", "-q"]          # or any check command
grader = command_grader(check, sandbox=SubprocessSandbox(timeout=30))
env = CodeTaskEnv(files, "make the tests pass", grader)  # untrusted code now runs in a child process
```

- **`Sandbox`** is a one-method seam (`run(files, command) -> GradeResult`).
  `SubprocessSandbox` is the first adapter; a container/seccomp adapter slots in
  behind the same interface without touching any environment.
- **Fail closed:** a timeout, a crash, or a bad command is a *non-passing grade*
  (`GradeResult.passed == False`), never a hang or an exception that escapes.
- **Determinism preserved:** the environment is stripped to a minimal allowlist and
  `PYTHONHASHSEED` is pinned, so a deterministic test is still a pure function of the
  files — the episode replays byte-for-byte through the sandbox.

## 6d. Reward composition (`crucible/reward.py`)

Real tasks are partial-credit. A `Rubric` turns several weighted checks into one
scalar reward:

```python
from crucible import rubric

r = rubric(
    ("tests_pass",  3.0, lambda state: run_tests(state)),
    ("no_secrets",  1.0, lambda state: not leaks_secret(state)),
)
score, breakdown = r.score(state)   # score in [0,1]; breakdown = {name: passed}
```

An environment calls `score` in `step`, returns `score` as the reward, and stashes
`breakdown` in `info` — where **replay verifies it** (§4), so the *reason* for a
reward is auditable and not merely the scalar. **Every check must be a deterministic
pure function of the state**, or the composed reward won't replay: an LLM judge
inside a criterion records a number nobody can re-derive, which degrades replay's
guarantee to "the judge said 0.7 that day." Rubrics stay strictly programmatic;
learned reward for non-verifiable tasks is parked (see [`BACKLOG.md`](BACKLOG.md)).

## 7. Recipe: write your own environment

The whole point of Crucible is that this is easy. Wrap software you already have:

```python
import hashlib
from crucible.env import Environment, StepResult

class ReverseStringEnv(Environment):
    """Task: reply with the reverse of a target string. Verifiable reward."""

    def __init__(self, target: str):
        self.target = target

    def reset(self, seed: int):
        self._done = False
        return {"task": f"reverse this string: {self.target!r}"}

    def step(self, action):
        correct = str(action) == self.target[::-1]
        self._done = correct
        return StepResult(
            observation={"you_said": str(action)},
            reward=1.0 if correct else -0.1,
            done=correct,
            info={"correct": correct},
        )

    def digest(self):
        # hash the hidden state the observation doesn't reveal
        return hashlib.sha256(f"{self.target}:{self._done}".encode()).hexdigest()[:16]
```

Checklist for a *good* (replayable) environment:
1. **Determinism** — everything random goes behind `seed`; no wall-clock, no network
   nondeterminism in the reward path.
2. **JSON-serializable** observations, actions, and `info` (so the trajectory records
   them). Note `info` is **verified by replay**, not just logged — so it must be
   deterministic like everything else. Put a wall-clock timestamp in there and replay
   will correctly call your environment non-reproducible.
3. **Verifiable reward** — score against real state (a check, a test, a query), not a
   vibe. That's what plugs into RLVR/GRPO training.
4. **A `digest`** covering hidden state, if you have any — it makes replay strict.

Then drive it:

```python
from crucible import rollout, replay
traj = rollout(ReverseStringEnv("crucible"), my_agent, seed=0, max_steps=5)
assert replay(ReverseStringEnv("crucible"), traj).ok   # reproduces byte-for-byte
traj.save("episode.trajectory.json")
```

## 8. Recipe: write your own agent

An agent is anything with `reset()` and `act(observation) -> action` — the
`Agent` protocol is `runtime_checkable`, so `isinstance(x, Agent)` works. A scripted
policy, a search algorithm, or an LLM wrapper all satisfy the same shape:

```python
class MyLLMAgent:
    def reset(self): self.history = []
    def act(self, observation):
        self.history.append(observation)
        return call_your_model(observation)   # returns an action
```

## 9. The provenance boundary (don't cross it)

Crucible is **training/eval infrastructure** — making and running environments to
*generate learning signal*. The "verification" here is programmatic task-success
checking for reward. It is deliberately **not** a runtime agent-accountability,
trust, or governance system (a different field, and the author's separate IP lives
there). If a feature starts to look like "prove what a deployed agent did in
production," stop — see [`VISION.md`](VISION.md) §5 and [`CONVENTIONS.md`](../CONVENTIONS.md).
