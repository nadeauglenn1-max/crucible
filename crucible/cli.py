"""The ``crucible`` command-line tool.

V1 ships ``crucible show <file>`` — load a saved trajectory, summarize it, and
integrity-check it (the recorded total reward must equal the sum of the step
rewards). Env-bound replay from the CLI (re-running a saved episode against a fresh
environment) needs an environment registry and arrives post-V1; today, replay lives
in the library (``crucible.replay``) where the environment is in hand.
"""

from __future__ import annotations

import argparse

from .trajectory import Trajectory


def _cmd_show(path: str) -> int:
    try:
        traj = Trajectory.load(path)
    except (OSError, ValueError) as exc:
        print(f"crucible: cannot read trajectory {path!r}: {exc}")
        return 2

    recomputed = sum(t.reward for t in traj.transitions)
    integrity_ok = abs(recomputed - traj.total_reward) < 1e-9

    print(f"env         : {traj.env_id}")
    print(f"seed        : {traj.seed}")
    print(f"steps       : {traj.steps}")
    print(f"total reward: {traj.total_reward:+.3f}")
    print(f"fingerprint : {traj.fingerprint()}")
    print(f"integrity   : {'ok' if integrity_ok else 'MISMATCH (total != sum of steps)'}")
    return 0 if integrity_ok else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="crucible", description="Crucible trajectory tools.")
    sub = parser.add_subparsers(dest="command", required=True)

    show = sub.add_parser("show", help="summarize and integrity-check a saved trajectory")
    show.add_argument("path", help="path to a saved trajectory JSON file")

    args = parser.parse_args(argv)
    # A subcommand is required, so ``show`` is the only path here today.
    return _cmd_show(args.path)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
