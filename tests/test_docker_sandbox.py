from pathlib import Path

import pytest

from crucible import DockerSandbox, GradeResult, Sandbox


def test_docker_sandbox_is_a_sandbox():
    assert isinstance(DockerSandbox(), Sandbox)


def test_docker_command_is_isolated():
    cmd = DockerSandbox(image="python:3.12-slim").docker_command(
        Path("/work-host"), ["python", "-c", "print(1)"]
    )
    assert cmd[:3] == ["docker", "run", "--rm"]
    # Network off, memory + pids capped -> real isolation, not just a subprocess.
    assert cmd[cmd.index("--network") + 1] == "none"
    assert "--memory" in cmd
    assert "--pids-limit" in cmd
    # The image precedes the command.
    i = cmd.index("python:3.12-slim")
    assert cmd[i + 1 :] == ["python", "-c", "print(1)"]


def test_fails_closed_when_docker_unavailable():
    result = DockerSandbox(docker="definitely-not-docker-xyz").run(
        {"x.py": "print(1)"}, ["python", "x.py"]
    )
    assert isinstance(result, GradeResult)
    assert not result.passed
    assert "not available" in result.stderr


def test_rejects_bad_timeout():
    with pytest.raises(ValueError):
        DockerSandbox(timeout=0)
