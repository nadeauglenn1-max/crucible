"""The flagship SWE-agent shape, end to end against *real* pytest.

A repo with a bug and a failing test; the agent edits the source; the reward is the
test suite going green (run as real `pytest` in the subprocess sandbox). No new code —
just `CodeTaskEnv` + `command_grader` — which is the point: the composition already
gives you the shape everyone wants.
"""

import sys

import pytest

from crucible import SubprocessSandbox, command_grader, replay, rollout
from crucible.envs import CodeTaskEnv
from examples.agents import ScriptedAgent

BUGGY = "def add(a, b):\n    return a - b\n"
FIXED = "def add(a, b):\n    return a + b\n"
TEST = (
    "from calc import add\n\n"
    "def test_add():\n"
    "    assert add(2, 3) == 5\n"
    "    assert add(0, 0) == 0\n"
)
PYTEST = [sys.executable, "-m", "pytest", "-q"]


def make_repo_env() -> CodeTaskEnv:
    grader = command_grader(PYTEST, sandbox=SubprocessSandbox(timeout=120))
    return CodeTaskEnv(
        files={"calc.py": BUGGY, "test_calc.py": TEST},
        task="Make the failing tests in test_calc.py pass by fixing calc.py.",
        grader=grader,
    )


def test_fixing_the_bug_turns_pytest_green():
    env = make_repo_env()
    env.reset(0)
    result = env.step({"calc.py": FIXED})
    assert result.reward == 1.0
    assert result.done is True


def test_still_broken_stays_red():
    env = make_repo_env()
    env.reset(0)
    result = env.step({"calc.py": "def add(a, b):\n    return a * b\n"})
    assert result.reward == -0.1
    assert result.done is False


def test_episode_replays_through_real_pytest():
    traj = rollout(make_repo_env(), ScriptedAgent([{"calc.py": FIXED}]), seed=0, max_steps=3)
    assert traj.transitions[-1].done is True
    assert replay(make_repo_env(), traj).ok
