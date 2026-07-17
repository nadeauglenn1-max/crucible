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


def test_replay_is_invariant_under_save_load(tmp_path):
    """Replaying a *saved* trajectory must be as faithful as replaying one in memory,
    even when observations use JSON-nonnative types (tuples) a round-trip coerces to
    lists. Reproducibility must not depend on whether the record touched disk."""
    from crucible import replay
    from crucible.env import Environment, StepResult

    class TupleObsEnv(Environment):
        env_id = "tuple-obs"

        def reset(self, seed):
            self._n = 0
            return (0, 0)

        def step(self, action):
            self._n += 1
            return StepResult((self._n, self._n), reward=1.0, done=self._n >= 2, info={})

    class Scripted:
        def reset(self):
            pass

        def act(self, observation):
            return "x"

    traj = rollout(TupleObsEnv(), Scripted(), seed=0, max_steps=5)
    assert replay(TupleObsEnv(), traj).ok  # in memory

    path = tmp_path / "t.json"
    traj.save(path)
    assert replay(TupleObsEnv(), Trajectory.load(path)).ok  # after a disk round-trip


def test_same_compares_across_json_coercion_and_falls_back():
    from crucible.rollout import _same

    assert _same((1, 2), [1, 2])  # tuple vs list: equal under canonical JSON
    assert _same({"a": 1}, {"a": 1})  # identical
    assert not _same([1, 2], [1, 3])  # genuinely different
    assert not _same(object(), object())  # non-JSON-serializable, unequal -> fallback
