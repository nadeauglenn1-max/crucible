"""Verifiers (Prime Intellect) adapter: a Crucible environment as a verifiers reward.

Verifiers scores a rollout with reward functions that return a float (typically in
[0, 1]); an environment's `Rubric` combines them by weight. Crucible supplies one:
extract the model's completion, parse it into an action, step a fresh Crucible
environment, and return the step reward. Same "the environment is the reward" bridge
as the TRL adapter, in verifiers' single-completion shape.

Zero-dependency: produces a plain callable using only Crucible + the stdlib, so it's
usable and testable without `verifiers` installed. Register it in a Rubric:

    import verifiers as vf
    rubric = vf.Rubric(funcs=[env_reward_fn(make_env)], weights=[1.0])
"""

from __future__ import annotations

from typing import Any, Callable

from ..env import Environment


def _default_extract(completion: Any) -> Any:
    """Pull the action text from a completion. Verifiers completions are chat-style
    (a list of ``{"role", "content"}`` messages), so take the last message's content;
    otherwise use the completion as-is."""
    if isinstance(completion, list) and completion and isinstance(completion[-1], dict):
        return completion[-1].get("content", "")
    return completion


def env_reward_fn(
    env_factory: Callable[[], Environment],
    *,
    parse_completion: Callable[[Any], Any] = lambda text: text,
    extract: Callable[[Any], Any] = _default_extract,
    seed: int = 0,
) -> Callable[..., float]:
    """Wrap a Crucible environment as a verifiers reward function.

    Returns ``reward(completion=None, **kwargs) -> float``: extract the completion's
    text, parse it into an environment action, step a fresh environment, and return
    the step reward. Suits single-step, verifiable-reward environments; supply
    ``parse_completion`` to map text to a non-text action shape.
    """

    def reward(completion: Any = None, **kwargs) -> float:
        action = parse_completion(extract(completion))
        env = env_factory()
        env.reset(seed)
        return float(env.step(action).reward)

    return reward
