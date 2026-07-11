"""GuessEnv — a fully deterministic number-guessing environment.

Small on purpose: it is the clean proof that an episode is reproducible. The secret
is fixed by the seed, feedback is a pure function of the guess, and the state digest
covers everything hidden, so a replay must reproduce the episode exactly.
"""

from __future__ import annotations

import hashlib
import random

from ..env import Action, Environment, Observation, StepResult


class GuessEnv(Environment):
    """Guess the secret integer in ``[low, high]``.

    Reward: ``+1.0`` for the correct guess (ends the episode); ``-0.05`` per wrong
    guess. The episode also ends, unsolved, after ``max_guesses``. The observation
    is a hint dict — ``feedback`` is ``"start"``, ``"higher"``, ``"lower"``, or
    ``"correct"``, plus the current bounds — enough for a binary-search agent.
    """

    def __init__(self, low: int = 1, high: int = 100, max_guesses: int = 10) -> None:
        if low > high:
            raise ValueError("low must be <= high")
        self.low = low
        self.high = high
        self.max_guesses = max_guesses
        self._secret = 0
        self._guesses = 0
        self._last: int | None = None

    def reset(self, seed: int) -> Observation:
        self._secret = random.Random(seed).randint(self.low, self.high)
        self._guesses = 0
        self._last = None
        return {"feedback": "start", "low": self.low, "high": self.high}

    def step(self, action: Action) -> StepResult:
        guess = int(action)
        self._guesses += 1
        self._last = guess

        if guess == self._secret:
            return StepResult(
                observation={"feedback": "correct"},
                reward=1.0,
                done=True,
                info={"secret": self._secret, "guesses": self._guesses},
            )

        done = self._guesses >= self.max_guesses
        feedback = "higher" if guess < self._secret else "lower"
        return StepResult(
            observation={"feedback": feedback, "low": self.low, "high": self.high},
            reward=-0.05,
            done=done,
            info={"guesses": self._guesses},
        )

    def digest(self) -> str:
        # Cover the hidden state (secret + counters) so replay is strict.
        raw = f"{self._secret}:{self._guesses}:{self._last}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
