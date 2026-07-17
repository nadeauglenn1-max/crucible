import importlib.util
from pathlib import Path

import pytest

from crucible import replay, rollout
from crucible.envs import CodeTaskEnv
from examples.agents import ScriptedAgent

BUGGY = "def add(a, b):\n    return a - b\n"
FIXED = "def add(a, b):\n    return a + b\n"


def test_code_env_rejects_escaping_edit_path():
    # An agent edit that names a path outside the sandbox must fail the grade and
    # never write outside the temp dir — even when the grader would otherwise pass.
    env = CodeTaskEnv(files={"a.py": "x = 1\n"}, task="t", grader=lambda root: True)
    env.reset(0)
    result = env.step({"../escape.py": "pwned"})
    assert not result.done
    assert result.reward < 0


def grade_add(root: Path) -> bool:
    """The 'test suite': import the agent's file and check the behavior. This is a
    pure function of the file contents, which is what keeps the episode replayable."""
    spec = importlib.util.spec_from_file_location("solution", root / "solution.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.add(2, 3) == 5 and module.add(-1, 1) == 0


def make_env() -> CodeTaskEnv:
    return CodeTaskEnv({"solution.py": BUGGY}, "Fix add so it adds.", grade_add)


def test_fix_makes_the_test_go_green():
    env = make_env()
    env.reset(0)
    result = env.step({"solution.py": FIXED})
    assert result.reward == 1.0
    assert result.done is True
    assert result.info["passed"] is True


def test_wrong_edit_is_penalized():
    env = make_env()
    env.reset(0)
    result = env.step({"solution.py": "def add(a, b):\n    return a * b\n"})
    assert result.reward == -0.1
    assert result.done is False


def test_broken_code_grades_false_not_crash():
    env = make_env()
    env.reset(0)
    result = env.step({"solution.py": "def add(a, b):\n    return a +\n"})  # syntax error
    assert result.reward == -0.1
    assert result.done is False


def test_requires_initial_files():
    with pytest.raises(ValueError):
        CodeTaskEnv({}, "task", grade_add)


def test_wrong_then_right_recovers_and_replays():
    # The reward writes itself: the same test suite scores a failed attempt and the
    # fix, and the whole episode replays byte-for-byte.
    traj = rollout(
        make_env(),
        ScriptedAgent([{"solution.py": BUGGY}, {"solution.py": FIXED}]),
        seed=0,
        max_steps=5,
    )
    assert traj.total_reward == pytest.approx(-0.1 + 1.0)
    assert traj.transitions[-1].done is True
    assert replay(make_env(), traj).ok
