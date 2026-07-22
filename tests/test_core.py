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


class _FlakyObsEnv(Environment):
    """Deterministic rewards/digests, but a non-deterministic observation at step 1 —
    exactly the case replay must catch now that it verifies the observation chain."""

    def reset(self, seed: int):
        self._n = 0
        return "start"

    def step(self, action):
        import random

        self._n += 1
        obs = random.random() if self._n == 1 else "end"
        return StepResult(observation=obs, reward=0.0, done=self._n >= 2)


def test_replay_detects_observation_divergence():
    traj = rollout(_FlakyObsEnv(), ScriptedAgent([0, 0]), seed=0, max_steps=3)
    report = replay(_FlakyObsEnv(), traj)
    assert not report.ok
    assert any("observation diverged" in m for m in report.mismatches)


# --- every recorded claim is bound -----------------------------------------------
#
# Reported by @sterngold (issue #14) against a 13-class tamper matrix: replay checked
# the step-by-step numbers but left the trajectory's other claims — the evidence in
# `info`, the headline total, the environment it says it ran in, and the observation
# the episode ended on — unverified. A recorded field nothing checks is exactly the
# field worth tampering with, so each one gets a test that tampers with it.


def test_rollout_records_the_final_observation():
    traj = solved_trajectory()
    # The episode ends on a correct guess; that terminal observation used to be
    # dropped on the floor by the rollout loop.
    assert traj.final_observation == {"feedback": "correct"}


def test_replay_detects_final_observation_tampering():
    traj = solved_trajectory()
    traj.final_observation = {"feedback": "bogus"}
    report = replay(GuessEnv(), traj)
    assert not report.ok
    assert any("final observation" in m for m in report.mismatches)


def test_replay_detects_info_tampering():
    traj = solved_trajectory()
    # `info` carries the evidence — a Rubric's per-criterion breakdown lands here.
    traj.transitions[-1].info = {"secret": 1, "guesses": 999}
    report = replay(GuessEnv(), traj)
    assert not report.ok
    assert any("info" in m for m in report.mismatches)


def test_replay_detects_total_reward_tampering():
    traj = solved_trajectory()
    traj.total_reward = 99.0  # the headline number, unmoored from the steps
    report = replay(GuessEnv(), traj)
    assert not report.ok
    assert any("total reward" in m for m in report.mismatches)


def test_replay_detects_a_different_environment():
    traj = solved_trajectory()
    traj.env_id = "some-other-world"
    report = replay(GuessEnv(), traj)
    assert not report.ok
    assert any("environment:" in m for m in report.mismatches)


def test_replay_detects_a_different_environment_config():
    traj = solved_trajectory()
    traj.env_config = {"low": 1, "high": 1000, "max_guesses": 10}
    report = replay(GuessEnv(), traj)
    assert not report.ok
    assert any("environment config" in m for m in report.mismatches)


def test_integrity_mismatches_is_the_one_check_replay_shares():
    traj = solved_trajectory()
    assert traj.integrity_mismatches() == []
    traj.total_reward += 1.0
    # The same message `crucible show` prints is the one replay reports — one
    # guarantee, one implementation, so the two can't disagree about a file.
    assert traj.integrity_mismatches()
    assert set(traj.integrity_mismatches()) <= set(replay(GuessEnv(), traj).mismatches)


def test_the_declared_version_is_declared_once():
    """`pyproject.toml` and `crucible.__version__` must agree.

    Two places state the version, so there are two chances to change one and
    forget the other — and the failure is silent: the wheel says one thing and the
    library says another, which is the shape of the defect this whole release was
    about. Nobody derives it, so something has to check it.
    """
    import tomllib
    from pathlib import Path

    import crucible

    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    declared = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]["version"]
    assert declared == crucible.__version__, (
        f"pyproject says {declared}, crucible.__version__ says {crucible.__version__}"
    )


def test_the_changelog_covers_the_current_version():
    """A release with no changelog entry is a contract change nobody was told about.

    0.2.0 tightened what `replay` verifies, which fails environments that passed
    before. The reply to the user who found it flagged that; the package flagged it
    to nobody, because the version stayed 0.1.0 and there was no changelog at all.
    """
    from pathlib import Path

    import crucible

    changelog = Path(__file__).resolve().parent.parent / "CHANGELOG.md"
    assert f"## {crucible.__version__}" in changelog.read_text(encoding="utf-8"), (
        f"CHANGELOG.md has no entry for {crucible.__version__}"
    )
