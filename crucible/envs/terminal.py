"""TerminalEnv — a stateful command-line session.

Where `CommandEnv` runs each command against a fixed set of files, `TerminalEnv` keeps
a **persistent working directory across steps**, so commands accumulate state: make a
directory in one step, write into it the next. This is the terminal-agent shape
(the TerminalBench / "endless terminal" lineage).

Reward: a goal check over the working directory's files — `+1.0` and done when it
passes, a small step cost otherwise.

Determinism / replay: the workdir starts from fixed files each episode and commands
apply in order, so a deterministic command sequence gives a deterministic episode.
Paths are normalized so file keys are OS-independent. A command whose *output* embeds
the (random) workdir path or wall-clock time is not replayable — and replay will say
so, now that it verifies the observation chain.

Security: commands run in a subprocess inside a temp dir (the same isolation level as
`CodeTaskEnv`'s grader). Real isolation (container / seccomp) is a documented
follow-up; don't point this at an untrusted agent without it.
"""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path
from typing import Callable

from ..env import Action, Environment, Observation, StepResult
from ..sandbox import SubprocessSandbox, materialize

Goal = Callable[[dict[str, str]], bool]


class TerminalEnv(Environment):
    """A persistent shell session. The agent's action is a command (argv); state
    carries across steps; reward comes from ``goal`` over the workdir's files."""

    def __init__(
        self,
        files: dict[str, str],
        task: str,
        goal: Goal,
        timeout: float = 10.0,
    ) -> None:
        self.files = dict(files)
        self.task = task
        self.goal = goal
        self.timeout = timeout
        self._sandbox = SubprocessSandbox(timeout=timeout)
        self._tmp: tempfile.TemporaryDirectory | None = None
        self._root: Path | None = None
        self._solved = False

    def reset(self, seed: int) -> Observation:
        if self._tmp is not None:
            self._tmp.cleanup()
        self._tmp = tempfile.TemporaryDirectory()
        self._root = Path(self._tmp.name)
        materialize(self._root, self.files)
        self._solved = False
        return {"task": self.task, "files": dict(self.files)}

    def step(self, action: Action) -> StepResult:
        assert self._root is not None, "reset() must be called before step()"
        result = self._sandbox.run_in(self._root, list(action))
        self._solved = bool(self.goal(self._workdir()))
        return StepResult(
            observation={
                "stdout": result.stdout,
                "stderr": result.stderr,
                "exit_code": result.exit_code,
            },
            reward=1.0 if self._solved else -0.05,
            done=self._solved,
            info={"solved": self._solved},
        )

    def _workdir(self) -> dict[str, str]:
        """The current working directory as {relative-posix-path: contents}."""
        assert self._root is not None
        files: dict[str, str] = {}
        for p in sorted(self._root.rglob("*")):
            if p.is_file():
                key = p.relative_to(self._root).as_posix()
                files[key] = p.read_text(encoding="utf-8")
        return files

    def digest(self) -> str:
        body = "\n".join(f"{k}={v}" for k, v in sorted(self._workdir().items()))
        return hashlib.sha256(f"{body}|{self._solved}".encode("utf-8")).hexdigest()[:16]
