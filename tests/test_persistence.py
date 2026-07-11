import json

import pytest

from crucible import Trajectory, rollout
from crucible.envs import GuessEnv
from crucible.trajectory import FORMAT_VERSION
from examples.agents import BinarySearchAgent


def a_trajectory() -> Trajectory:
    return rollout(GuessEnv(), BinarySearchAgent(), seed=11, max_steps=12)


def test_save_load_roundtrip(tmp_path):
    traj = a_trajectory()
    path = tmp_path / "episode.trajectory.json"
    traj.save(path)
    assert path.exists()

    restored = Trajectory.load(path)
    assert restored.to_json() == traj.to_json()
    assert restored.fingerprint() == traj.fingerprint()


def test_saved_file_is_versioned_envelope(tmp_path):
    path = tmp_path / "e.json"
    a_trajectory().save(path)
    envelope = json.loads(path.read_text(encoding="utf-8"))
    assert envelope["version"] == FORMAT_VERSION
    assert "trajectory" in envelope


def test_load_rejects_unknown_version(tmp_path):
    path = tmp_path / "future.json"
    path.write_text(json.dumps({"version": 999, "trajectory": {}}), encoding="utf-8")
    with pytest.raises(ValueError):
        Trajectory.load(path)
