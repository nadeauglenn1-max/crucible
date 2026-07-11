"""The Environment contract: real software wrapped as a trainable, gradable world.

An environment is the vessel. It exposes a minimal rollout protocol an agent can be
driven through, and it honors the determinism contract that makes episodes
replayable and rewards auditable.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

# Observations and actions are rich by design — text, dicts, structured data — not
# just tensors. They must be JSON-serializable so a trajectory can be recorded and
# replayed; the trajectory layer enforces that.
Observation = Any
Action = Any


@dataclass
class StepResult:
    """The outcome of one step: what the agent sees next, what it earned, whether
    the episode is over, and any side information (kept out of the reward path)."""

    observation: Observation
    reward: float
    done: bool
    info: dict = field(default_factory=dict)


class Environment(ABC):
    """A Crucible environment.

    Determinism contract (this is law, see CONVENTIONS): ``reset(seed)`` fully
    determines the episode. The same seed followed by the same sequence of actions
    must reproduce the same observations, rewards, and state digests. An environment
    that cannot honor this is not replayable — and ``replay`` will say so rather than
    pretend otherwise.
    """

    #: A stable identifier for the environment, used to label trajectories.
    env_id: str = ""

    @abstractmethod
    def reset(self, seed: int) -> Observation:
        """Start a fresh episode determined entirely by ``seed`` and return the
        initial observation."""

    @abstractmethod
    def step(self, action: Action) -> StepResult:
        """Advance the world by one action and return the resulting StepResult."""

    def digest(self) -> str:
        """A stable hash of the environment's observable state, used by ``replay`` to
        confirm a re-run reproduced the same world. The default is empty, which means
        replay verifies observations and rewards but not internal state; environments
        with hidden state should override this to make replay strict."""
        return ""

    def name(self) -> str:
        """The environment's identifier, defaulting to the class name."""
        return self.env_id or type(self).__name__

    def config(self) -> dict:
        """A JSON-serializable dict that ``registry.make(name, config)`` can use to
        reconstruct an identical environment. The default is empty; an environment
        that wants to be CLI-replayable overrides this to return its constructor
        arguments. An environment carrying a live callable simply can't, and says so
        by leaving this empty."""
        return {}
