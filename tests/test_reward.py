import hashlib

import pytest

from crucible import Environment, StepResult, replay, rollout, rubric
from examples.agents import ScriptedAgent


def a_rubric():
    return rubric(
        ("nonempty", 1.0, lambda s: len(s) > 0),
        ("has_x", 1.0, lambda s: "x" in s),
    )


def test_all_criteria_pass_scores_one():
    score, breakdown = a_rubric().score("xy")
    assert score == 1.0
    assert breakdown == {"nonempty": True, "has_x": True}


def test_partial_credit_is_the_weighted_fraction():
    score, breakdown = a_rubric().score("hi")  # nonempty passes, has_x fails
    assert score == 0.5
    assert breakdown == {"nonempty": True, "has_x": False}


def test_no_criteria_pass_scores_zero():
    score, _ = a_rubric().score("")
    assert score == 0.0


def test_weights_matter():
    r = rubric(
        ("cheap", 1.0, lambda s: True),
        ("valuable", 3.0, lambda s: False),
    )
    score, _ = r.score("anything")
    assert score == pytest.approx(1.0 / 4.0)  # only the weight-1 criterion passed


def test_rubric_validation():
    with pytest.raises(ValueError):
        rubric()  # empty
    with pytest.raises(ValueError):
        rubric(("neg", -1.0, lambda s: True))  # negative weight
    with pytest.raises(ValueError):
        rubric(("zero", 0.0, lambda s: True))  # weights sum to zero


class RubricEnv(Environment):
    """A tiny environment that grades with a rubric — partial credit, breakdown in
    info, still fully replayable."""

    def __init__(self):
        self.rubric = a_rubric()

    def reset(self, seed):
        self._last = ""
        return {"task": "produce a non-empty string containing 'x'"}

    def step(self, action):
        s = str(action)
        self._last = s
        score, breakdown = self.rubric.score(s)
        return StepResult(
            observation={"you_said": s},
            reward=score,
            done=score == 1.0,
            info={"breakdown": breakdown},
        )

    def digest(self):
        return hashlib.sha256(self._last.encode("utf-8")).hexdigest()[:16]


def test_environment_using_a_rubric_replays():
    traj = rollout(RubricEnv(), ScriptedAgent(["hi", "hix"]), seed=0, max_steps=5)
    # "hi" -> 0.5 (partial), "hix" -> 1.0 (done).
    assert [t.reward for t in traj.transitions] == [0.5, 1.0]
    assert traj.transitions[-1].done is True
    assert traj.transitions[0].info["breakdown"] == {"nonempty": True, "has_x": False}
    assert replay(RubricEnv(), traj).ok
