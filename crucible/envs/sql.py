"""SQLTaskEnv — wrap a real SQLite database into a verifiable-reward environment.

This is the thesis in one file. The agent is handed a schema and a task; its action
is a SQL query; the reward is *programmatic and verifiable* — run the query against a
real database and compare the result set to the expected answer. No learned reward
model, no human grader: a reward anyone can re-check by replaying the episode.
"""

from __future__ import annotations

import hashlib
import sqlite3

from ..env import Action, Environment, Observation, StepResult
from ..registry import register


@register("sql-task")
class SQLTaskEnv(Environment):
    """A SQL task over an in-memory SQLite database built fresh each episode.

    ``schema_sql`` creates the tables, ``seed_sql`` fills them, ``task`` is the
    natural-language ask, and ``expected_rows`` is the correct result (a list of
    row-lists). Reward: ``+1.0`` and done when the agent's query returns exactly the
    expected rows; ``-0.1`` for a wrong result or a SQL error (the episode continues
    so the agent can try again up to the caller's ``max_steps``).

    Determinism: the database is rebuilt from the same DDL/DML every ``reset``, and
    queries are read-only, so an episode is a pure function of the actions taken.
    """

    def __init__(
        self,
        schema_sql: str,
        seed_sql: str,
        task: str,
        expected_rows: list[list],
    ) -> None:
        self.schema_sql = schema_sql
        self.seed_sql = seed_sql
        self.task = task
        self.expected_rows = expected_rows
        self._conn: sqlite3.Connection | None = None
        self._solved = False
        self._attempts = 0

    def reset(self, seed: int) -> Observation:
        if self._conn is not None:
            self._conn.close()
        self._conn = sqlite3.connect(":memory:")
        self._conn.executescript(self.schema_sql)
        self._conn.executescript(self.seed_sql)
        self._solved = False
        self._attempts = 0
        return {"task": self.task, "schema": self.schema_sql}

    def step(self, action: Action) -> StepResult:
        assert self._conn is not None, "reset() must be called before step()"
        query = str(action)
        self._attempts += 1

        try:
            rows = [list(r) for r in self._conn.execute(query).fetchall()]
        except sqlite3.Error as exc:
            return StepResult(
                observation={"error": str(exc)},
                reward=-0.1,
                done=False,
                info={"error": str(exc), "attempts": self._attempts},
            )

        correct = rows == self.expected_rows
        self._solved = correct
        return StepResult(
            observation={"rows": rows},
            reward=1.0 if correct else -0.1,
            done=correct,
            info={"correct": correct, "attempts": self._attempts},
        )

    def digest(self) -> str:
        raw = f"{self.task}:{self._solved}:{self._attempts}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    def config(self) -> dict:
        return {
            "schema_sql": self.schema_sql,
            "seed_sql": self.seed_sql,
            "task": self.task,
            "expected_rows": self.expected_rows,
        }
