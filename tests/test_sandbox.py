import sys

import pytest

from crucible import (
    GradeResult,
    Sandbox,
    SubprocessSandbox,
    command_grader,
    replay,
    rollout,
)
from crucible.envs import CodeTaskEnv
from examples.agents import ScriptedAgent

FIXED = "def add(a, b):\n    return a + b\n"
BUGGY = "def add(a, b):\n    return a - b\n"
# A check command that runs *in the subprocess*, importing the agent's file.
CHECK = [sys.executable, "-c", "import solution; assert solution.add(2, 3) == 5"]


def test_subprocess_sandbox_is_a_sandbox():
    assert isinstance(SubprocessSandbox(), Sandbox)


def test_sandbox_passes_correct_code():
    result = SubprocessSandbox(timeout=15).run({"solution.py": FIXED}, CHECK)
    assert isinstance(result, GradeResult)
    assert result.passed
    assert result.exit_code == 0


def test_sandbox_fails_wrong_code():
    result = SubprocessSandbox(timeout=15).run({"solution.py": BUGGY}, CHECK)
    assert not result.passed
    assert result.exit_code != 0


def test_sandbox_times_out_and_fails_closed():
    loop = [sys.executable, "-c", "while True: pass"]
    result = SubprocessSandbox(timeout=1.0).run({"x.py": "# noop\n"}, loop)
    assert not result.passed
    assert "timed out" in result.stderr


def test_sandbox_launch_error_is_a_grade_not_a_crash():
    result = SubprocessSandbox().run({}, ["definitely-not-a-real-binary-xyz-123"])
    assert not result.passed
    assert "launch error" in result.stderr


def test_sandbox_rejects_bad_timeout():
    with pytest.raises(ValueError):
        SubprocessSandbox(timeout=0)


def test_sandbox_rejects_absolute_path_escape(tmp_path):
    # An absolute-path file key must not write outside the sandbox; run fails closed.
    escape = tmp_path / "escaped.txt"
    result = SubprocessSandbox().run({str(escape): "pwned"}, [sys.executable, "-c", "pass"])
    assert not result.passed
    assert not escape.exists()


def test_sandbox_rejects_parent_traversal(tmp_path):
    result = SubprocessSandbox().run({"../escape.txt": "pwned"}, [sys.executable, "-c", "pass"])
    assert not result.passed


def test_materialize_refuses_escape_but_writes_safe_paths(tmp_path):
    from crucible.sandbox import materialize

    for bad in ("../evil.txt", "sub/../../evil.txt"):
        with pytest.raises(ValueError):
            materialize(tmp_path, {bad: "x"})
    # A nested but contained path is written normally.
    materialize(tmp_path, {"sub/ok.txt": "hi"})
    assert (tmp_path / "sub" / "ok.txt").read_text(encoding="utf-8") == "hi"


def test_command_grader_drives_codetaskenv_safely():
    # The untrusted "agent code" runs in a subprocess via command_grader, not in
    # this process — and the whole episode still replays byte-for-byte.
    grader = command_grader(CHECK, sandbox=SubprocessSandbox(timeout=15))

    def make_env() -> CodeTaskEnv:
        return CodeTaskEnv({"solution.py": BUGGY}, "Fix add so it adds.", grader)

    traj = rollout(
        make_env(),
        ScriptedAgent([{"solution.py": BUGGY}, {"solution.py": FIXED}]),
        seed=0,
        max_steps=5,
    )
    assert traj.total_reward == pytest.approx(-0.1 + 1.0)
    assert traj.transitions[-1].done is True
    assert replay(make_env(), traj).ok
