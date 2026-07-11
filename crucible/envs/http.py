"""HttpTaskEnv — wrap a recorded HTTP service as an environment.

An HTTP service is the one thing that *can't* be deterministic against a live network,
so — as the backlog calls for — this wraps a **recording**: a fixed map of
request → response (the VCR / cassette pattern). The agent's action is a request
(`{"method", "path"}`); the env returns the recorded response (or a 404), and reward
is `+1.0` when the response body matches the expected body.

Because the whole service is data, `HttpTaskEnv` is deterministic, replayable, and
registerable / CLI-replayable — you can train an agent to find the right endpoint
against a recorded API trace without touching a network. Wrapping a *live* WSGI app
behind the same shape (deterministic only if the app is pure) is a later option.
"""

from __future__ import annotations

import hashlib

from ..env import Action, Environment, Observation, StepResult
from ..registry import register


@register("http-task")
class HttpTaskEnv(Environment):
    """Route the agent's request against a recorded response map; reward when the
    response body equals ``expected_body``."""

    def __init__(
        self,
        recording: dict,
        task: str,
        expected_body,
        not_found: dict | None = None,
    ) -> None:
        # recording maps "METHOD path" -> {"status": int, "body": ...}
        self.recording = dict(recording)
        self.task = task
        self.expected_body = expected_body
        self.not_found = not_found if not_found is not None else {"status": 404, "body": None}
        self._solved = False
        self._attempts = 0

    @staticmethod
    def _key(action: Action) -> str:
        method = str(action.get("method", "GET")).upper()
        path = str(action.get("path", "/"))
        return f"{method} {path}"

    def reset(self, seed: int) -> Observation:
        self._solved = False
        self._attempts = 0
        return {"task": self.task, "endpoints": sorted(self.recording)}

    def step(self, action: Action) -> StepResult:
        self._attempts += 1
        response = self.recording.get(self._key(action), self.not_found)
        self._solved = response.get("body") == self.expected_body
        return StepResult(
            observation={"response": response},
            reward=1.0 if self._solved else -0.1,
            done=self._solved,
            info={"correct": self._solved, "attempts": self._attempts},
        )

    def digest(self) -> str:
        raw = f"{self.task}:{self._solved}:{self._attempts}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    def config(self) -> dict:
        return {
            "recording": self.recording,
            "task": self.task,
            "expected_body": self.expected_body,
            "not_found": self.not_found,
        }
