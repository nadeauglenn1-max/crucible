"""CommandEnv — wrap a command-line environment.

The agent's action is a command (a list of argv tokens); the reward is programmatic
and verifiable: run the command over a fixed set of files in the sandbox, and succeed
when it exits cleanly *and* its stdout matches the expected output. Because its whole
configuration is data (files, task, expected output), it is registerable and
CLI-replayable — unlike environments that carry a live grader callable.

Determinism: the files are fixed each episode and the command is the only variable,
so a deterministic command yields a deterministic episode. A command that embeds
wall-clock time or randomness in its output isn't replayable — and `replay` will say
so rather than hide it.
"""

from __future__ import annotations

import hashlib

from ..env import Action, Environment, Observation, StepResult
from ..registry import register
from ..sandbox import SubprocessSandbox


@register("command")
class CommandEnv(Environment):
    """Run the agent's command over fixed ``files``; reward when it exits 0 and its
    stripped stdout equals ``expected_stdout``. ``+1.0`` and done on success, ``-0.1``
    otherwise (the episode continues up to the caller's ``max_steps``)."""

    def __init__(
        self,
        files: dict[str, str],
        task: str,
        expected_stdout: str,
        timeout: float = 10.0,
    ) -> None:
        self.files = dict(files)
        self.task = task
        self.expected_stdout = expected_stdout
        self.timeout = timeout
        self._sandbox = SubprocessSandbox(timeout=timeout)
        self._solved = False
        self._attempts = 0

    def reset(self, seed: int) -> Observation:
        self._solved = False
        self._attempts = 0
        return {"task": self.task, "files": dict(self.files)}

    def step(self, action: Action) -> StepResult:
        command = list(action)
        self._attempts += 1
        result = self._sandbox.run(self.files, command)
        # Require a clean exit AND a matching output, so a crash with empty stdout
        # can't accidentally "match" an empty expected output.
        correct = result.exit_code == 0 and result.stdout.strip() == self.expected_stdout
        self._solved = correct
        return StepResult(
            observation={
                "stdout": result.stdout,
                "stderr": result.stderr,
                "exit_code": result.exit_code,
            },
            reward=1.0 if correct else -0.1,
            done=correct,
            info={"correct": correct, "attempts": self._attempts},
        )

    def digest(self) -> str:
        raw = f"{self.task}:{self._solved}:{self._attempts}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    def config(self) -> dict:
        return {
            "files": self.files,
            "task": self.task,
            "expected_stdout": self.expected_stdout,
            "timeout": self.timeout,
        }
