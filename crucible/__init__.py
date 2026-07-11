"""Crucible — turn any real software into a trainable, gradable, replayable RL
environment for AI agents.

The core is tiny and dependency-free: an ``Environment`` contract, a replayable
``Trajectory``, and ``rollout`` / ``replay``. Everything else is an environment or
an agent that plugs in.
"""

from .env import Action, Environment, Observation, StepResult
from .rollout import Agent, ReplayReport, replay, rollout
from .trajectory import Trajectory, Transition

__all__ = [
    "Environment",
    "StepResult",
    "Observation",
    "Action",
    "Trajectory",
    "Transition",
    "Agent",
    "rollout",
    "replay",
    "ReplayReport",
]

__version__ = "0.1.0"
