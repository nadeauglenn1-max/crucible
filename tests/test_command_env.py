import sys

from crucible import make, replay, rollout
from crucible.envs import CommandEnv
from examples.agents import ScriptedAgent

RIGHT = [sys.executable, "-c", "print(2 + 3)"]
WRONG = [sys.executable, "-c", "print(2 + 2)"]


def make_env() -> CommandEnv:
    return CommandEnv(files={}, task="print the sum of 2 and 3", expected_stdout="5", timeout=15)


def test_correct_command_is_rewarded():
    env = make_env()
    env.reset(0)
    result = env.step(RIGHT)
    assert result.reward == 1.0
    assert result.done is True
    assert result.observation["exit_code"] == 0


def test_wrong_output_is_penalized():
    env = make_env()
    env.reset(0)
    result = env.step(WRONG)
    assert result.reward == -0.1
    assert result.done is False


def test_launch_error_is_not_a_false_pass():
    # A crashing command with empty stdout must not "match" an empty expected output.
    env = CommandEnv(files={}, task="do nothing", expected_stdout="", timeout=15)
    env.reset(0)
    result = env.step(["definitely-not-a-real-binary-xyz"])
    assert result.reward == -0.1
    assert result.done is False


def test_wrong_then_right_recovers_and_replays():
    traj = rollout(make_env(), ScriptedAgent([WRONG, RIGHT]), seed=0, max_steps=5)
    assert [t.reward for t in traj.transitions] == [-0.1, 1.0]
    assert traj.transitions[-1].done is True
    assert replay(make_env(), traj).ok


def test_command_env_is_registerable_and_cli_replayable():
    original = make_env()
    rebuilt = make("command", original.config())
    assert isinstance(rebuilt, CommandEnv)
    # A registered env rebuilt from config reproduces a recorded episode.
    traj = rollout(original, ScriptedAgent([RIGHT]), seed=0, max_steps=2)
    assert replay(make("command", traj.env_config), traj).ok
