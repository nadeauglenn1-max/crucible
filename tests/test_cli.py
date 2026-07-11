import json

import pytest

from crucible import Trajectory, Transition, rollout
from crucible.cli import main
from crucible.envs import GuessEnv
from examples.agents import BinarySearchAgent


def saved(tmp_path, traj: Trajectory):
    path = tmp_path / "e.trajectory.json"
    traj.save(path)
    return str(path)


def test_show_summarizes_a_good_trajectory(tmp_path, capsys):
    traj = rollout(GuessEnv(), BinarySearchAgent(), seed=5, max_steps=12)
    rc = main(["show", saved(tmp_path, traj)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "fingerprint :" in out
    assert traj.fingerprint() in out
    assert "integrity   : ok" in out


def test_show_reports_missing_file(tmp_path, capsys):
    rc = main(["show", str(tmp_path / "nope.json")])
    assert rc == 2
    assert "cannot read trajectory" in capsys.readouterr().out


def test_show_reports_bad_version(tmp_path, capsys):
    path = tmp_path / "future.json"
    path.write_text(json.dumps({"version": 999, "trajectory": {}}), encoding="utf-8")
    rc = main(["show", str(path)])
    assert rc == 2


def test_show_flags_integrity_mismatch(tmp_path, capsys):
    # A trajectory whose total_reward disagrees with its steps is caught.
    traj = Trajectory(
        env_id="Tampered",
        seed=0,
        initial_observation=None,
        transitions=[Transition(observation=None, action=1, reward=0.1, done=True, info={}, digest="")],
        total_reward=99.0,  # a lie
    )
    rc = main(["show", saved(tmp_path, traj)])
    assert rc == 1
    assert "MISMATCH" in capsys.readouterr().out


def test_requires_a_subcommand():
    with pytest.raises(SystemExit):
        main([])
