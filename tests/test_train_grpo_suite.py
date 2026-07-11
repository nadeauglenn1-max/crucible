"""Guard the generalization battery's ground truth.

Each `Task.expected` in the suite is hand-computed from the seed data; a wrong value
would silently cap that task at 0% on a GPU run and waste the whole battery. Here a
known-correct query per task is stepped through the real environment and must pay out,
so any drift in the expected rows fails in CI on a CPU — no GPU or trl needed.
"""

from crucible.envs import SQLTaskEnv
from examples.train_grpo import SCHEMA, SEED, extract_sql
from examples.train_grpo_suite import TASKS

# One known-correct query per task (by name) — the answer key for the answer key.
GOLDEN = {
    "second_highest": "SELECT name, salary FROM employees ORDER BY salary DESC LIMIT 1,1",
    "dept_payroll": (
        "SELECT department, SUM(salary) FROM employees "
        "GROUP BY department ORDER BY SUM(salary) DESC"
    ),
    "dept_headcount": (
        "SELECT department, COUNT(*) FROM employees "
        "GROUP BY department HAVING COUNT(*) > 1 ORDER BY department"
    ),
    "dept_avg_top": (
        "SELECT department, AVG(salary) FROM employees "
        "GROUP BY department ORDER BY AVG(salary) DESC LIMIT 1"
    ),
}


def test_every_task_has_a_golden_query():
    assert {t.name for t in TASKS} == set(GOLDEN)


def test_golden_query_solves_each_task():
    for task in TASKS:
        env = SQLTaskEnv(SCHEMA, SEED, task.ask, task.expected)
        env.reset(0)
        result = env.step(extract_sql(GOLDEN[task.name]))
        assert result.reward > 0, f"{task.name}: expected rows are wrong"
        assert result.info["correct"] is True


def test_tasks_are_distinct_skills():
    # the whole point of the battery — no duplicated task
    assert len({t.skill for t in TASKS}) == len(TASKS)
    assert len({t.ask for t in TASKS}) == len(TASKS)
