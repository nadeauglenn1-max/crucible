import sys

from crucible import replay, rollout
from crucible.envs import TerminalEnv
from examples.agents import ScriptedAgent

MKDIR = [sys.executable, "-c", "import os; os.makedirs('sub', exist_ok=True)"]
WRITE = [sys.executable, "-c", "open('sub/x.txt', 'w').write('ok')"]
BADWRITE = [sys.executable, "-c", "open('sub/x.txt', 'w').write('nope')"]


def reached_goal(files: dict[str, str]) -> bool:
    return files.get("sub/x.txt") == "ok"


def make_env() -> TerminalEnv:
    return TerminalEnv(
        files={},
        task="create sub/, then write 'ok' into sub/x.txt",
        goal=reached_goal,
        timeout=15,
    )


def test_state_accumulates_across_steps():
    # The write in step 2 depends on the directory made in step 1 — proof the session
    # is stateful, not per-command.
    traj = rollout(make_env(), ScriptedAgent([MKDIR, WRITE]), seed=0, max_steps=5)
    assert [t.reward for t in traj.transitions] == [-0.05, 1.0]
    assert traj.transitions[-1].done is True


def test_episode_replays_byte_for_byte():
    traj = rollout(make_env(), ScriptedAgent([MKDIR, WRITE]), seed=0, max_steps=5)
    assert replay(make_env(), traj).ok


def test_unreached_goal_keeps_going():
    env = make_env()
    env.reset(0)
    env.step(MKDIR)
    result = env.step(BADWRITE)  # writes the wrong contents
    assert result.reward == -0.05
    assert result.done is False


def test_initial_files_are_materialized():
    env = TerminalEnv(
        files={"input.txt": "5\n"},
        task="copy input.txt to output.txt",
        goal=lambda files: files.get("output.txt") == "5\n",
        timeout=15,
    )
    env.reset(0)
    copy = [sys.executable, "-c", "open('output.txt','w').write(open('input.txt').read())"]
    result = env.step(copy)
    assert result.done is True
    assert result.reward == 1.0


def test_reset_starts_a_clean_session():
    env = make_env()
    env.reset(0)
    env.step(MKDIR)
    env.step(WRITE)  # solved
    solved_digest = env.digest()
    env.reset(0)  # a fresh session must forget the previous one
    assert env.digest() != solved_digest
