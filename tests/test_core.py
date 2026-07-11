import copy

import pytest

from crucible import Agent, Environment, ReplayReport, StepResult, Trajectory, replay, rollout
from crucible.envs import GuessEnv
from examples.agents import BinarySearchAgent, ScriptedAgent


class _MinimalEnv(Environment):
    """An environment that relies on the base default digest (empty) — proof that
    an env can skip digests and still be replayed on observations and rewards."""

    def reset(self, seed: int):
        return 0

    def step(self, action):
        return StepResult(observation=action, reward=1.0, done=True)


def solved_trajectory(seed: int = 42) -> Trajectory:
    return rollout(GuessEnv(), BinarySearchAgent(), seed=seed, max_steps=12)


def test_rollout_records_and_solves():
    traj = solved_trajectory()
    assert traj.steps >= 1
    # Binary search over 1..100 solves in <= 7 guesses, ending on a correct guess.
    last = traj.transitions[-1]
    assert last.done is True
    assert last.reward == 1.0
    # total_reward is the sum of the shaped penalties plus the final +1.
    assert traj.total_reward == pytest.approx(sum(t.reward for t in traj.transitions))


def test_rollout_respects_max_steps():
    traj = rollout(GuessEnv(), ScriptedAgent([200]), seed=1, max_steps=1)
    assert traj.steps == 1
    assert traj.transitions[-1].done is False  # not solved, stopped by max_steps


def test_rollout_rejects_bad_max_steps():
    with pytest.raises(ValueError):
        rollout(GuessEnv(), BinarySearchAgent(), seed=1, max_steps=0)


def test_replay_reproduces():
    traj = solved_trajectory()
    report = replay(GuessEnv(), traj)
    assert isinstance(report, ReplayReport)
    assert report.ok
    assert report.mismatches == []
    assert report.steps == traj.steps


def test_replay_detects_reward_tampering():
    traj = solved_trajectory()
    traj.transitions[0].reward += 0.5
    report = replay(GuessEnv(), traj)
    assert not report.ok
    assert any("reward" in m for m in report.mismatches)


def test_replay_detects_done_tampering():
    traj = solved_trajectory()
    traj.transitions[-1].done = False  # lie about the episode ending
    report = replay(GuessEnv(), traj)
    assert not report.ok
    assert any("done" in m for m in report.mismatches)


def test_replay_detects_digest_tampering():
    traj = solved_trajectory()
    traj.transitions[0].digest = "0" * 16
    report = replay(GuessEnv(), traj)
    assert not report.ok
    assert any("digest" in m for m in report.mismatches)


def test_replay_detects_initial_observation_tampering():
    traj = solved_trajectory()
    traj.initial_observation = {"feedback": "bogus"}
    report = replay(GuessEnv(), traj)
    assert not report.ok
    assert any("initial observation" in m for m in report.mismatches)


def test_replay_detects_early_end():
    traj = solved_trajectory()
    # Append a phantom transition after the real (done) one: replay ends early.
    traj.transitions.append(copy.deepcopy(traj.transitions[-1]))
    report = replay(GuessEnv(), traj)
    assert not report.ok
    assert any("ended early" in m for m in report.mismatches)


def test_trajectory_json_roundtrip_and_fingerprint():
    traj = solved_trajectory()
    restored = Trajectory.from_json(traj.to_json())
    assert restored.to_json() == traj.to_json()
    assert restored.fingerprint() == traj.fingerprint()
    # A change to any recorded value changes the fingerprint.
    restored.transitions[0].reward += 1.0
    assert restored.fingerprint() != traj.fingerprint()


def test_agent_protocol_is_runtime_checkable():
    assert isinstance(BinarySearchAgent(), Agent)
    assert isinstance(ScriptedAgent([1]), Agent)


def test_scripted_agent_requires_actions():
    with pytest.raises(ValueError):
        ScriptedAgent([])


def test_default_digest_is_empty_and_still_replays():
    traj = rollout(_MinimalEnv(), ScriptedAgent([7]), seed=0, max_steps=1)
    assert traj.transitions[0].digest == ""  # the base default, no override
    assert replay(_MinimalEnv(), traj).ok
