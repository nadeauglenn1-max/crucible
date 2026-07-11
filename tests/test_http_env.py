from crucible import make, replay, rollout
from crucible.envs import HttpTaskEnv
from examples.agents import ScriptedAgent

RECORDING = {
    "GET /users/5": {"status": 200, "body": {"id": 5, "name": "Ada"}},
    "GET /users": {"status": 200, "body": [{"id": 5}]},
}
EXPECTED = {"id": 5, "name": "Ada"}
RIGHT = {"method": "GET", "path": "/users/5"}
WRONG = {"method": "GET", "path": "/users"}
MISSING = {"method": "GET", "path": "/nope"}


def make_env() -> HttpTaskEnv:
    return HttpTaskEnv(RECORDING, task="fetch user 5", expected_body=EXPECTED)


def test_right_request_is_rewarded():
    env = make_env()
    env.reset(0)
    result = env.step(RIGHT)
    assert result.reward == 1.0
    assert result.done is True
    assert result.observation["response"]["status"] == 200


def test_wrong_endpoint_is_penalized():
    env = make_env()
    env.reset(0)
    result = env.step(WRONG)
    assert result.reward == -0.1
    assert result.done is False


def test_unknown_path_returns_404_not_a_crash():
    env = make_env()
    env.reset(0)
    result = env.step(MISSING)
    assert result.reward == -0.1
    assert result.observation["response"]["status"] == 404


def test_wrong_then_right_recovers_and_replays():
    traj = rollout(make_env(), ScriptedAgent([WRONG, RIGHT]), seed=0, max_steps=5)
    assert [t.reward for t in traj.transitions] == [-0.1, 1.0]
    assert replay(make_env(), traj).ok


def test_registerable_and_cli_replayable():
    original = make_env()
    rebuilt = make("http-task", original.config())
    assert isinstance(rebuilt, HttpTaskEnv)
    traj = rollout(original, ScriptedAgent([RIGHT]), seed=0, max_steps=2)
    assert replay(make("http-task", traj.env_config), traj).ok
