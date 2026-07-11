"""A runnable walk-through: drive agents through two environments, then replay each
episode to prove it reproduces byte-for-byte.

    python -m examples.demo
"""

from __future__ import annotations

from typing import Callable

from crucible import Environment, replay, rollout
from crucible.envs import GuessEnv, SQLTaskEnv
from examples.agents import BinarySearchAgent, ScriptedAgent

# A small real database and a verifiable task: total ordered per customer, desc.
SCHEMA = """
CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT);
CREATE TABLE orders (id INTEGER PRIMARY KEY, customer_id INTEGER, amount INTEGER);
"""
SEED = """
INSERT INTO customers VALUES (1,'Ada'),(2,'Lin'),(3,'Bo');
INSERT INTO orders VALUES (1,1,100),(2,1,50),(3,2,200),(4,3,30);
"""
TASK = "For each customer, the total amount ordered, highest total first: name, total."
EXPECTED = [["Lin", 200], ["Ada", 150], ["Bo", 30]]
WRONG_SQL = "SELECT name FROM customers"
RIGHT_SQL = (
    "SELECT c.name, SUM(o.amount) FROM customers c "
    "JOIN orders o ON o.customer_id = c.id "
    "GROUP BY c.id ORDER BY SUM(o.amount) DESC"
)


def run(title: str, make_env: Callable[[], Environment], agent, seed: int) -> None:
    traj = rollout(make_env(), agent, seed=seed, max_steps=12)
    report = replay(make_env(), traj)
    verdict = "reproduced OK" if report.ok else f"MISMATCH: {report.mismatches}"
    print(f"\n=== {title} ===")
    print(f"  steps        : {traj.steps}")
    print(f"  total reward : {traj.total_reward:+.2f}")
    print(f"  fingerprint  : {traj.fingerprint()[:16]}")
    print(f"  replay       : {verdict}")


def main() -> None:
    print("Crucible - forge agents against real software, then replay the episode.")

    run("GuessEnv - binary-search agent", GuessEnv, BinarySearchAgent(), seed=42)

    run(
        "SQLTaskEnv - real SQLite, verifiable reward (wrong then right)",
        lambda: SQLTaskEnv(SCHEMA, SEED, TASK, EXPECTED),
        ScriptedAgent([WRONG_SQL, RIGHT_SQL]),
        seed=0,
    )


if __name__ == "__main__":
    main()
