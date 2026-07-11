import pytest

from crucible import replay, rollout
from crucible.envs import GuessEnv, SQLTaskEnv
from examples.agents import BinarySearchAgent, ScriptedAgent

SCHEMA = (
    "CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT);"
    "CREATE TABLE orders (id INTEGER PRIMARY KEY, customer_id INTEGER, amount INTEGER);"
)
SEED_SQL = (
    "INSERT INTO customers VALUES (1,'Ada'),(2,'Lin'),(3,'Bo');"
    "INSERT INTO orders VALUES (1,1,100),(2,1,50),(3,2,200),(4,3,30);"
)
TASK = "total per customer, highest first"
EXPECTED = [["Lin", 200], ["Ada", 150], ["Bo", 30]]
RIGHT_SQL = (
    "SELECT c.name, SUM(o.amount) FROM customers c "
    "JOIN orders o ON o.customer_id = c.id GROUP BY c.id ORDER BY SUM(o.amount) DESC"
)


def make_sql_env() -> SQLTaskEnv:
    return SQLTaskEnv(SCHEMA, SEED_SQL, TASK, EXPECTED)


# --- GuessEnv -------------------------------------------------------------------

def test_guess_deterministic_from_seed():
    a, b = GuessEnv(), GuessEnv()
    a.reset(7)
    b.reset(7)
    # Same seed => identical hidden state and identical responses to the same guess.
    ra, rb = a.step(50), b.step(50)
    assert (ra.reward, ra.done, ra.observation) == (rb.reward, rb.done, rb.observation)
    assert a.digest() == b.digest()


def test_guess_solvable_by_binary_search():
    traj = rollout(GuessEnv(), BinarySearchAgent(), seed=3, max_steps=12)
    assert traj.transitions[-1].reward == 1.0
    assert traj.transitions[-1].done is True
    assert replay(GuessEnv(), traj).ok


def test_guess_exhausts_when_never_correct():
    # Guessing above the range yields "lower" forever, so the episode ends unsolved
    # exactly at max_guesses.
    traj = rollout(GuessEnv(max_guesses=5), ScriptedAgent([200]), seed=1, max_steps=50)
    assert traj.steps == 5
    assert traj.transitions[-1].done is True
    assert traj.transitions[-1].reward == -0.05  # a penalty, not a win


def test_guess_rejects_bad_bounds():
    with pytest.raises(ValueError):
        GuessEnv(low=100, high=1)


def test_env_name_defaults_to_class_then_env_id():
    env = GuessEnv()
    assert env.name() == "GuessEnv"  # falls back to the class name
    env.env_id = "guess-v1"
    assert env.name() == "guess-v1"  # an explicit id wins


# --- SQLTaskEnv -----------------------------------------------------------------

def test_sql_correct_query_is_rewarded():
    env = make_sql_env()
    env.reset(0)
    result = env.step(RIGHT_SQL)
    assert result.reward == 1.0
    assert result.done is True
    assert result.info["correct"] is True
    assert result.observation["rows"] == EXPECTED


def test_sql_wrong_result_is_penalized():
    env = make_sql_env()
    env.reset(0)
    result = env.step("SELECT name FROM customers")
    assert result.reward == -0.1
    assert result.done is False


def test_sql_error_is_penalized_not_crashed():
    env = make_sql_env()
    env.reset(0)
    result = env.step("SELECT nope FROM missing_table")
    assert result.reward == -0.1
    assert result.done is False
    assert "error" in result.info


def test_sql_reset_rebuilds_clean_state():
    env = make_sql_env()
    env.reset(0)
    env.step(RIGHT_SQL)  # solves -> state advances
    solved_digest = env.digest()

    env.reset(0)  # a fresh episode must return to clean state
    fresh = make_sql_env()
    fresh.reset(0)
    assert env.digest() == fresh.digest()
    assert env.digest() != solved_digest


def test_sql_wrap_real_software_end_to_end():
    # The thesis in one test: a scripted agent recovers from a wrong query to a
    # right one, and the whole episode replays byte-for-byte.
    traj = rollout(
        make_sql_env(),
        ScriptedAgent(["SELECT 1", RIGHT_SQL]),
        seed=0,
        max_steps=5,
    )
    assert traj.total_reward == pytest.approx(-0.1 + 1.0)
    assert replay(make_sql_env(), traj).ok
