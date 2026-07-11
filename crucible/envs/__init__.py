"""Example Crucible environments.

`GuessEnv` is a tiny fully-deterministic environment — the clean proof that rollout
and replay reproduce an episode exactly. `SQLTaskEnv` wraps a real SQLite database
into a verifiable-reward environment: the thesis in ~60 lines.
"""

from .code import CodeTaskEnv
from .guess import GuessEnv
from .sql import SQLTaskEnv

__all__ = ["GuessEnv", "SQLTaskEnv", "CodeTaskEnv"]
