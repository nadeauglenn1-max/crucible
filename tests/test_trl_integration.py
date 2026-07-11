from crucible import rollout
from crucible.envs import GuessEnv, SQLTaskEnv
from crucible.integrations.trl import env_reward_func, to_prompt_dataset
from examples.agents import BinarySearchAgent

SCHEMA = "CREATE TABLE t (n INTEGER);"
SEED = "INSERT INTO t VALUES (2),(3);"
RIGHT = "SELECT SUM(n) FROM t"
WRONG = "SELECT n FROM t"
EXPECTED = [[5]]


def make_sql_env() -> SQLTaskEnv:
    return SQLTaskEnv(SCHEMA, SEED, "sum n", EXPECTED)


def test_to_prompt_dataset_shape():
    traj = rollout(GuessEnv(), BinarySearchAgent(), seed=1, max_steps=12)
    rows = to_prompt_dataset([traj])
    assert len(rows) == traj.steps
    assert set(rows[0]) == {"prompt", "env_id"}
    assert rows[0]["env_id"] == "guess"
    assert isinstance(rows[0]["prompt"], str)


def test_env_reward_func_scores_completions():
    # A SQL environment becomes the reward function: the completion is the query.
    reward = env_reward_func(make_sql_env)
    rewards = reward(prompts=["sum n", "sum n"], completions=[RIGHT, WRONG])
    assert rewards == [1.0, -0.1]
    assert all(isinstance(r, float) for r in rewards)


def test_env_reward_func_handles_no_completions():
    reward = env_reward_func(make_sql_env)
    assert reward(prompts=[], completions=[]) == []
    assert reward() == []  # TRL may call with no kwargs during setup


def test_env_reward_func_uses_parse_completion():
    # A parser maps the model's text to the environment's action shape.
    reward = env_reward_func(make_sql_env, parse_completion=lambda c: c.strip())
    assert reward(completions=[f"  {RIGHT}  "]) == [1.0]
