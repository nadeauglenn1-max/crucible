"""Example agents — the thing Crucible trains and grades.

These are deliberately trivial scripted policies. A real agent (an LLM, a learned
policy) implements the same ``reset`` / ``act`` protocol and drops in unchanged.
"""

from __future__ import annotations

from crucible.env import Action, Observation


class BinarySearchAgent:
    """Solves ``GuessEnv`` optimally by binary search over the remaining range."""

    def reset(self) -> None:
        self.low: int | None = None
        self.high: int | None = None
        self.last: int | None = None

    def act(self, observation: Observation) -> Action:
        feedback = observation.get("feedback")
        if feedback == "start":
            self.low, self.high = observation["low"], observation["high"]
        elif feedback == "higher":  # secret is above the last guess
            self.low = (self.last or 0) + 1
        elif feedback == "lower":  # secret is below the last guess
            self.high = (self.last or 0) - 1
        self.last = (self.low + self.high) // 2
        return self.last


class ScriptedAgent:
    """Replays a fixed list of actions, cycling if the episode runs long. Handy for
    demos (feed it the right SQL) and tests (feed it a losing move)."""

    def __init__(self, actions: list[Action]) -> None:
        if not actions:
            raise ValueError("ScriptedAgent needs at least one action")
        self.actions = list(actions)
        self.i = 0

    def reset(self) -> None:
        self.i = 0

    def act(self, observation: Observation) -> Action:
        action = self.actions[self.i % len(self.actions)]
        self.i += 1
        return action
