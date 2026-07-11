"""Guard the cross-modality battery's ground truth (CPU, no GPU/trl).

Each modality's environment must actually pay out for a known-correct solution, or a
GPU run would chase an unreachable reward. This steps a golden solution through the
*real* environment for all three types — SQL row match, shell exit-0 + stdout, and a
passing test — so any drift in the tasks fails in CI on a CPU.
"""

from examples.train_xmodal import (
    MODALITIES,
    parse_code,
    parse_command,
    parse_sql,
)

GOLDEN = {
    "sql_above_dept_avg": (
        "SELECT name FROM employees e WHERE salary > "
        "(SELECT AVG(salary) FROM employees d WHERE d.department = e.department) "
        "ORDER BY name",
        parse_sql,
    ),
    "cmd_count_lines": (
        "python -c \"print(len(open('log.txt').readlines()))\"",
        parse_command,
    ),
    "code_second_largest": (
        "def solve(xs):\n    return sorted(set(xs), reverse=True)[1]",
        parse_code,
    ),
}


def test_every_modality_has_a_golden_solution():
    assert {m.name for m in MODALITIES} == set(GOLDEN)


def test_golden_solution_solves_each_modality():
    for mod in MODALITIES:
        solution, parse = GOLDEN[mod.name]
        env = mod.make_env()
        env.reset(0)
        result = env.step(parse(solution))
        assert result.reward > 0, f"{mod.name}: golden solution did not pay out"
        assert result.done is True


def test_modalities_are_three_distinct_env_types():
    # the whole point: different environment classes, not three of the same
    kinds = {type(m.make_env()).__name__ for m in MODALITIES}
    assert kinds == {"SQLTaskEnv", "CommandEnv", "CodeTaskEnv"}


def test_wrong_answers_do_not_pay_out():
    # a sanity floor: an empty/garbage action is penalized, not rewarded
    for mod in MODALITIES:
        env = mod.make_env()
        env.reset(0)
        assert env.step(mod.parse("definitely not the answer")).reward < 0
