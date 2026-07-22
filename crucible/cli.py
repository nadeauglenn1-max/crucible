"""The ``crucible`` command-line tool.

- ``crucible show <file>`` — load a saved trajectory, summarize it, and
  integrity-check it, with no environment needed.
- ``crucible replay <file>`` — rebuild the environment from the registry and re-run
  the episode against it.

``replay`` is strictly the stronger check: it runs the same self-contained integrity
check ``show`` does (via ``Trajectory.integrity_mismatches``) and *then* re-derives
the episode. The two commands can never reach opposite verdicts on one file.
"""

from __future__ import annotations

import argparse

from . import envs as _envs  # noqa: F401  (import populates the environment registry)
from .registry import make, registered
from .rollout import replay
from .trajectory import Trajectory


def _cmd_show(path: str) -> int:
    try:
        traj = Trajectory.load(path)
    except (OSError, ValueError) as exc:
        print(f"crucible: cannot read trajectory {path!r}: {exc}")
        return 2

    problems = traj.integrity_mismatches()

    print(f"env         : {traj.env_id}")
    print(f"seed        : {traj.seed}")
    print(f"steps       : {traj.steps}")
    print(f"total reward: {traj.total_reward:+.3f}")
    print(f"fingerprint : {traj.fingerprint()}")
    print(f"integrity   : {'ok' if not problems else 'MISMATCH'}")
    for p in problems:
        print(f"  - {p}")
    return 0 if not problems else 1


def _cmd_replay(path: str) -> int:
    try:
        traj = Trajectory.load(path)
    except (OSError, ValueError) as exc:
        print(f"crucible: cannot read trajectory {path!r}: {exc}")
        return 2

    try:
        env = make(traj.env_id, traj.env_config)
    except (KeyError, TypeError) as exc:
        print(
            f"crucible: cannot rebuild environment {traj.env_id!r}: {exc}\n"
            f"          registered: {registered()} "
            f"(environments carrying a live callable aren't CLI-replayable)"
        )
        return 2

    report = replay(env, traj)
    if report.ok:
        print(f"replay: reproduced OK ({report.steps} steps)")
        return 0
    print(f"replay: MISMATCH ({len(report.mismatches)} issue(s) over {report.steps} steps)")
    for m in report.mismatches:
        print(f"  - {m}")
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="crucible", description="Crucible trajectory tools.")
    sub = parser.add_subparsers(dest="command", required=True)

    show = sub.add_parser("show", help="summarize and integrity-check a saved trajectory")
    show.add_argument("path", help="path to a saved trajectory JSON file")

    rep = sub.add_parser("replay", help="rebuild the environment and re-run a saved episode")
    rep.add_argument("path", help="path to a saved trajectory JSON file")

    args = parser.parse_args(argv)
    if args.command == "replay":
        return _cmd_replay(args.path)
    return _cmd_show(args.path)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
