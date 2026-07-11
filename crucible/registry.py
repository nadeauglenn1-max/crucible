"""A registry of environments, so a saved trajectory can be reconstructed by name.

A trajectory records its environment's registered name and a serializable config
(see `Environment.config`). Given those two, `make` rebuilds an identical environment
— which is what lets `crucible replay <file>` re-run a saved episode without the
original Python objects in hand.

Only environments whose *entire* reconstruction is captured by a JSON-serializable
config can be registered this way. An environment carrying a live callable (e.g.
`CodeTaskEnv`'s grader) is still fully usable from the library; it just isn't
CLI-replayable until its behavior is expressed as data.
"""

from __future__ import annotations

from typing import Callable

from .env import Environment

_REGISTRY: dict[str, Callable[..., Environment]] = {}


def register(name: str) -> Callable[[type], type]:
    """Class decorator: register an environment factory under ``name`` and stamp the
    class's ``env_id`` so its trajectories carry that name."""

    def decorate(cls: type) -> type:
        if name in _REGISTRY:
            raise ValueError(f"environment {name!r} is already registered")
        cls.env_id = name
        _REGISTRY[name] = cls
        return cls

    return decorate


def make(name: str, config: dict) -> Environment:
    """Reconstruct a registered environment from its name and config."""
    if name not in _REGISTRY:
        raise KeyError(f"unknown environment {name!r}; registered: {registered()}")
    return _REGISTRY[name](**config)


def registered() -> list[str]:
    """The names of all registered environments, sorted."""
    return sorted(_REGISTRY)
