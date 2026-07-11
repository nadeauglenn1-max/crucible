from crucible.envs import SQLTaskEnv
from crucible.integrations.verifiers import env_reward_fn

SCHEMA = "CREATE TABLE t (n INTEGER);"
SEED = "INSERT INTO t VALUES (2),(3);"
RIGHT = "SELECT SUM(n) FROM t"
WRONG = "SELECT n FROM t"
EXPECTED = [[5]]


def make_sql_env() -> SQLTaskEnv:
    return SQLTaskEnv(SCHEMA, SEED, "sum n", EXPECTED)


def test_reward_from_chat_style_completion():
    reward = env_reward_fn(make_sql_env)
    assert reward(completion=[{"role": "assistant", "content": RIGHT}]) == 1.0
    assert reward(completion=[{"role": "assistant", "content": WRONG}]) == -0.1


def test_reward_from_plain_string_completion():
    reward = env_reward_fn(make_sql_env)
    assert reward(completion=RIGHT) == 1.0


def test_parse_completion_is_applied():
    reward = env_reward_fn(make_sql_env, parse_completion=lambda t: t.strip())
    assert reward(completion=f"  {RIGHT}  ") == 1.0


def test_empty_chat_completion_falls_back():
    # An empty list isn't a chat message list; it passes through as the action and
    # simply scores as a wrong query (no crash).
    reward = env_reward_fn(make_sql_env)
    assert reward(completion=[]) == -0.1
