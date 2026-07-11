"""Sandboxed grading — run a check *command* over the agent's files in isolation.

`CodeTaskEnv`'s in-process grader (import the file, call it) is fine for trusted,
scripted use. It is reckless against a real model writing real code: you'd be
executing a stranger's program inside your own process. This module is the safe
path: the agent's code runs in a **subprocess**, in a fresh temp directory, with a
minimal environment and a hard timeout — and passing is simply "the command exited 0."

The `Sandbox` seam has one method (`run`); the `SubprocessSandbox` here is the first
adapter. A stronger adapter (container / seccomp / nsjail) slots in behind the same
interface without touching any environment. `command_grader` adapts a sandbox into a
`CodeTaskEnv` grader, so switching an environment from trusted to sandboxed grading
is a one-line change.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

# A minimal, cross-platform environment allowlist: enough for an interpreter to
# start, nothing that leaks the host's configuration into the graded run.
_ENV_ALLOWLIST = ("PATH", "SYSTEMROOT", "SystemRoot", "TEMP", "TMP", "LD_LIBRARY_PATH")


@dataclass
class GradeResult:
    """The outcome of a sandboxed command: whether it passed (exit 0), the exit code,
    and captured output. ``passed`` is the only thing reward should depend on."""

    passed: bool
    exit_code: int
    stdout: str
    stderr: str


@runtime_checkable
class Sandbox(Protocol):
    """Runs a command over a set of files in isolation and reports pass/fail."""

    def run(self, files: dict[str, str], command: list[str]) -> GradeResult: ...


class SubprocessSandbox:
    """Materialize ``files`` into a fresh temp directory and run ``command`` there in
    a subprocess, with a minimal environment and a hard timeout. Anything that isn't
    a clean exit-0 — a failing test, a crash, a timeout, a launch error — is a
    non-passing grade, never a hang or an exception that escapes (fail closed).
    """

    def __init__(self, timeout: float = 30.0) -> None:
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        self.timeout = timeout

    def run(self, files: dict[str, str], command: list[str]) -> GradeResult:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for path, content in files.items():
                dest = root / path
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(content, encoding="utf-8")
            return self.run_in(root, command)

    def run_in(self, root: Path, command: list[str]) -> GradeResult:
        """Run ``command`` in an already-populated directory. Determinism: the
        environment is stripped to the allowlist and ``PYTHONHASHSEED`` is pinned, so
        a deterministic test is a pure function of the files under it."""
        env = {k: os.environ[k] for k in _ENV_ALLOWLIST if k in os.environ}
        env["PYTHONHASHSEED"] = "0"
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        try:
            proc = subprocess.run(
                command,
                cwd=str(root),
                env=env,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
        except subprocess.TimeoutExpired:
            return GradeResult(False, -1, "", f"timed out after {self.timeout}s")
        except (OSError, ValueError) as exc:
            return GradeResult(False, -1, "", f"launch error: {exc}")
        return GradeResult(proc.returncode == 0, proc.returncode, proc.stdout, proc.stderr)


def command_grader(command: list[str], sandbox: Sandbox | None = None):
    """Adapt a sandboxed ``command`` into a ``CodeTaskEnv`` grader.

    The command runs over the environment's current files (passing = exit 0). Use
    this instead of an in-process grader whenever the agent is untrusted: its code
    executes in the sandbox's subprocess, never in yours. Defaults to a
    ``SubprocessSandbox`` with a 30s timeout.
    """
    box = sandbox or SubprocessSandbox()

    def grade(root: Path) -> bool:
        files = {
            str(p.relative_to(root)): p.read_text(encoding="utf-8")
            for p in root.rglob("*")
            if p.is_file()
        }
        return box.run(files, command).passed

    return grade
