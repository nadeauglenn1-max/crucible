"""Reward composition: build one scalar reward from several weighted checks.

Real tasks are partial-credit — "tests pass **and** style clean **and** no secret
leaked" — not a single boolean. A `Rubric` scores a state against weighted criteria
and returns both the scalar reward (the weighted fraction passed, in [0, 1]) and a
per-criterion breakdown for `info`.

Every check must be a deterministic, pure function of the state, or the composed
reward won't replay. This stays strictly programmatic on purpose: learned reward for
non-verifiable tasks is a separate, careful design (see `docs/BACKLOG.md`) and is not
here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

Check = Callable[[Any], bool]


@dataclass(frozen=True)
class Criterion:
    """One weighted, named check over a state."""

    name: str
    weight: float
    check: Check


class Rubric:
    """A weighted set of criteria. `score(state)` returns the weighted fraction of
    criteria that passed, in [0, 1], plus a name → passed breakdown."""

    def __init__(self, criteria: list[Criterion]) -> None:
        if not criteria:
            raise ValueError("a rubric needs at least one criterion")
        if any(c.weight < 0 for c in criteria):
            raise ValueError("criterion weights must be non-negative")
        total = sum(c.weight for c in criteria)
        if total <= 0:
            raise ValueError("criterion weights must sum to a positive value")
        self.criteria = list(criteria)
        self._total = total

    def score(self, state: Any) -> tuple[float, dict]:
        earned = 0.0
        breakdown: dict[str, bool] = {}
        for criterion in self.criteria:
            passed = bool(criterion.check(state))
            breakdown[criterion.name] = passed
            if passed:
                earned += criterion.weight
        return earned / self._total, breakdown


def rubric(*criteria: tuple[str, float, Check]) -> Rubric:
    """Build a Rubric from ``(name, weight, check)`` tuples."""
    return Rubric([Criterion(name, weight, check) for (name, weight, check) in criteria])
