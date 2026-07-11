"""Unit tests for the importable, dependency-light parts of examples/train_grpo.

The training itself (`main`) needs a GPU + trl and is excluded from coverage; what we
test here is the glue that decides *correctness*: parsing a model completion into a
SQL string, and that a correct query actually pays out through the real environment.
"""

from examples.train_grpo import (
    EXPECTED,
    extract_sql,
    make_env,
    prompt_messages,
    sql_reward,
)

GOLDEN = "SELECT name, salary FROM employees ORDER BY salary DESC LIMIT 1 OFFSET 1"


def test_extract_sql_plain_string():
    assert extract_sql("  SELECT 1;  ") == "SELECT 1"


def test_extract_sql_conversational_list():
    completion = [{"role": "assistant", "content": "SELECT 1;"}]
    assert extract_sql(completion) == "SELECT 1"


def test_extract_sql_strips_sql_fence():
    text = "Here you go:\n```sql\nSELECT 1;\n```"
    assert extract_sql(text) == "SELECT 1"


def test_extract_sql_strips_bare_fence():
    assert extract_sql("```\nSELECT 1\n```") == "SELECT 1"


def test_extract_sql_strips_label():
    assert extract_sql("SQL: SELECT 1;") == "SELECT 1"


def test_extract_sql_empty():
    assert extract_sql("") == ""
    assert extract_sql([]) == ""
    assert extract_sql(None) == ""


def test_golden_query_solves_the_env():
    # A correct query, extracted and stepped through the real SQLite env, pays out.
    env = make_env()
    env.reset(0)
    result = env.step(extract_sql(GOLDEN))
    assert result.reward > 0
    assert result.info["correct"] is True
    assert result.observation["rows"] == EXPECTED


def test_sql_reward_matches_expected_shape():
    # The reward function GRPO calls returns one float per completion.
    rewards = sql_reward(completions=[GOLDEN, "SELECT name FROM employees"])
    assert len(rewards) == 2
    assert rewards[0] > 0  # correct
    assert rewards[1] < 0  # wrong shape → penalty


def test_prompt_messages_shape():
    msgs = prompt_messages()
    assert msgs[0]["role"] == "user"
    assert "SQL" in msgs[0]["content"]
