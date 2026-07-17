"""CodeTaskEnv — wrap a small code repo + a grader into an environment.

This is the shape everyone wants from an agent-training environment (the SWE-agent
loop) and the first place the reward *writes itself*: the test suite IS the reward
function. The agent edits files; the grader runs the tests; green is the reward.

Determinism: the initial files are fixed, edits apply deterministically, and the
grader must be a pure function of the file state — which a deterministic test suite
is. The digest covers the file contents plus the last grade, so replay is strict.

Security note: grading runs code derived from the files under test. v1 materializes
the files to a temp directory and calls the grader in-process, which is fine for the
scripted example and tests here. A real deployment MUST sandbox the grader
(subprocess / container / seccomp) before pointing it at an untrusted agent. That
isolation is a named roadmap item, deliberately not faked here.
"""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path
from typing import Callable

from ..env import Action, Environment, Observation, StepResult
from ..sandbox import materialize

Grader = Callable[[Path], bool]


class CodeTaskEnv(Environment):
    """A code task: start from ``files``, make the ``grader`` pass.

    ``files`` is the initial repo snapshot (relative path -> contents). The agent's
    action is a dict of edits (path -> new contents) applied over the current files.
    Reward is ``+1.0`` and done when the grader passes; ``-0.1`` otherwise (the
    episode continues so the agent can try again up to the caller's ``max_steps``).
    """

    def __init__(self, files: dict[str, str], task: str, grader: Grader) -> None:
        if not files:
            raise ValueError("CodeTaskEnv needs at least one initial file")
        self.files = dict(files)
        self.task = task
        self.grader = grader
        self._current: dict[str, str] = {}
        self._passed = False

    def reset(self, seed: int) -> Observation:
        # The content is fixed; seed is part of the contract but unused here.
        self._current = dict(self.files)
        self._passed = False
        return {"task": self.task, "files": dict(self._current)}

    def step(self, action: Action) -> StepResult:
        for path, content in dict(action).items():
            self._current[path] = content
        self._passed = self._grade()
        return StepResult(
            observation={"files": dict(self._current)},
            reward=1.0 if self._passed else -0.1,
            done=self._passed,
            info={"passed": self._passed},
        )

    def _grade(self) -> bool:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            try:
                # File keys come from the agent's edits; materialize refuses any that
                # would escape the temp dir (an unsafe path is just a failed grade).
                materialize(root, self._current)
                return bool(self.grader(root))
            except Exception:
                # A grader that raises (syntax error in the agent's code, failed
                # import, failing assertion) — or an unsafe edit path — is simply a
                # non-passing grade.
                return False

    def digest(self) -> str:
        body = "\n".join(f"{k}={self._current[k]}" for k in sorted(self._current))
        return hashlib.sha256(f"{body}|{self._passed}".encode("utf-8")).hexdigest()[:16]
