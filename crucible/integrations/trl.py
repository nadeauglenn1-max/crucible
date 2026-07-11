"""TRL adapters: feed Crucible into Hugging Face TRL's GRPO trainer.

`GRPOTrainer` is *online* — it generates completions itself during training — so it
needs two things: a **dataset of prompts** and a **reward function** that scores the
generated completions and returns a list of floats. Crucible supplies both:

- `to_prompt_dataset` turns trajectory observations into the `[{"prompt": ...}]` rows
  GRPO trains on.
- `env_reward_func` turns a Crucible **environment** into a TRL reward function: each
  completion is parsed into an action, the environment scores it (verifiably), and
  the step reward is returned. This is the real bridge — the environment *is* the
  reward.

Both produce TRL-shaped values using only Crucible + the stdlib, so they need no
dependency on `trl` and are testable without it. Pass the results straight to
`trl.GRPOTrainer(model=..., reward_funcs=[reward], train_dataset=dataset, ...)`.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable

from ..env import Environment
from ..export import _default_render
from ..trajectory import Trajectory


def to_prompt_dataset(
    trajectories: Iterable[Trajectory],
    *,
    render_observation: Callable[[Any], str] | None = None,
) -> list[dict]:
    """Flatten trajectory observations into GRPO prompt rows: `{"prompt", "env_id"}`.
    GRPO generates its own completions, so only the prompt (and any extra columns a
    reward function needs) belongs in the dataset."""
    render = render_observation or _default_render
    rows: list[dict] = []
    for traj in trajectories:
        for t in traj.transitions:
            rows.append({"prompt": render(t.observation), "env_id": traj.env_id})
    return rows


def env_reward_func(
    env_factory: Callable[[], Environment],
    *,
    parse_completion: Callable[[Any], Any] = lambda completion: completion,
    seed: int = 0,
) -> Callable[..., list[float]]:
    """Wrap a Crucible environment as a TRL reward function.

    Returns `reward_func(prompts=None, completions=None, **kwargs) -> list[float]`
    matching TRL's convention: for each completion, a fresh environment is reset and
    stepped with `parse_completion(completion)`, and the step's reward is returned.

    Best suited to single-step, verifiable-reward environments (SQL, command, ...).
    `parse_completion` maps the model's text completion to an environment action — the
    identity default fits environments whose action *is* the text (e.g. a SQL query);
    supply a parser (e.g. `shlex.split`) for others.
    """

    def reward_func(prompts=None, completions=None, **kwargs) -> list[float]:
        rewards: list[float] = []
        for completion in completions or []:
            env = env_factory()
            env.reset(seed)
            result = env.step(parse_completion(completion))
            rewards.append(float(result.reward))
        return rewards

    return reward_func
