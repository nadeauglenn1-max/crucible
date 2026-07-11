"""Beyond SQL: one training loop, three *different environment types*.

The SQL battery (`train_grpo_suite.py`) proves the method isn't one lucky task — but
it's all one env class. The sharper question is whether Crucible trains *different
kinds of agents*. This runs the identical GRPO loop over three genuinely different
Crucible environment types, each with its own verifiable reward:

  - SQLTaskEnv   — a **database** agent: write a query, graded by running it on SQLite
  - CommandEnv   — a **shell** agent: emit a command, graded by its exit code + stdout
  - CodeTaskEnv  — a **coding** agent: write a function, graded by **real execution**

Same `env_reward_func` seam, same trainer, one model (Qwen2.5-1.5B — big enough that
these harder modalities have real headroom). Each task trains a fresh copy from the
same pretrained weights. The reward is always "the environment paid out" — a SQL row
match, a shell exit-0 + stdout match, or a passing test.

    pip install "crucible-rl[train]"        # + a CUDA GPU (~8GB)
    python -m examples.train_xmodal

Writes per-task baseline/trained/curve to `examples/results/xmodal.json`; render the
chart (no GPU) with `python -m examples.results.plot_xmodal`.
"""

from __future__ import annotations

import gc
import json
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from crucible.env import Environment
from crucible.envs import CodeTaskEnv, CommandEnv, SQLTaskEnv
from crucible.integrations.trl import env_reward_func
from examples.train_grpo import accuracy, grpo_config, lora_config, prompt_messages

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_STEPS = 60
EVAL_N = 20
NUM_GEN = 6  # 1.5B needs a smaller group than the 0.5B runs to fit 8GB
OUT = Path(__file__).resolve().parent / "results" / "xmodal.json"


# --- completion parsing, shared -------------------------------------------------

def _text(completion: Any) -> str:
    if isinstance(completion, list):
        return completion[-1]["content"] if completion else ""
    return completion or ""


def _unfence(text: str) -> str:
    """Strip a ```...``` code fence (and any language tag) if present."""
    text = str(text).strip()
    if "```" in text:
        block = text.split("```", 2)[1]
        block = re.sub(r"^[a-zA-Z]+\n", "", block)  # drop a leading ```python tag
        text = block.strip()
    return text


def parse_sql(completion: Any) -> str:
    text = _unfence(_text(completion))
    for label in ("SQL:", "Query:", "sql:"):
        if text.startswith(label):
            text = text[len(label):].strip()
    return text.rstrip(";").strip()


def parse_command(completion: Any) -> list[str]:
    line = next((ln for ln in _unfence(_text(completion)).splitlines() if ln.strip()), "")
    try:
        argv = shlex.split(line, posix=True)
    except ValueError:
        argv = []
    return argv or ["__no_command__"]  # a non-command fails closed in the sandbox


def parse_code(completion: Any) -> dict:
    return {"solution.py": _unfence(_text(completion))}


# --- the three tasks, one per environment type ----------------------------------

@dataclass
class Modality:
    name: str
    kind: str                                   # short label for the chart
    make_env: Callable[[], Environment]
    parse: Callable[[Any], Any]
    instruction: str
    max_completion_length: int = 100


SCHEMA = "CREATE TABLE employees (id INTEGER PRIMARY KEY, name TEXT, department TEXT, salary INTEGER);"
SEED = (
    "INSERT INTO employees VALUES (1,'Ada','Engineering',120000),(2,'Lin','Engineering',135000),"
    "(3,'Bo','Sales',90000),(4,'Cy','Sales',105000),(5,'Dev','Marketing',80000);"
)
SQL_ASK = ("the names of employees who earn strictly more than the average salary of "
           "their own department, ordered by name")
SQL_EXPECTED = [["Cy"], ["Lin"]]

NUMS = "3\n1\n4\n1\n5\n9\n2\n6\n"  # sum = 31

CODE_STUB = "def solve(xs):\n    # return the second largest DISTINCT value in xs\n    pass\n"
CODE_ASSERTS = [
    "assert solve([5,2,8,8,1])==5",
    "assert solve([1,2,3])==2",
    "assert solve([10,9])==9",
    "assert solve([4,4,7])==4",
]


def _code_grader(root: Path) -> bool:
    """Run the agent's solution.py against the asserts in a subprocess (fail closed)."""
    harness = "from solution import solve\n" + "\n".join(CODE_ASSERTS)
    try:
        done = subprocess.run(
            [sys.executable, "-c", harness], cwd=root, capture_output=True, timeout=10
        )
        return done.returncode == 0
    except Exception:
        return False


MODALITIES = [
    Modality(
        name="sql_above_dept_avg",
        kind="database (SQL)",
        make_env=lambda: SQLTaskEnv(SCHEMA, SEED, SQL_ASK, SQL_EXPECTED),
        parse=parse_sql,
        instruction=(
            f"You are given a SQLite database with this schema:\n\n{SCHEMA}\n\n"
            f"Write a single SQL query that returns {SQL_ASK}.\n"
            "Return one column: name.\n"
            "Reply with ONLY the SQL query, no explanation, no markdown."
        ),
    ),
    Modality(
        name="cmd_count_lines",
        kind="shell (command)",
        make_env=lambda: CommandEnv({"log.txt": "".join(f"line {i}\n" for i in range(13))},
                                    "count the lines in log.txt", "13"),
        parse=parse_command,
        instruction=(
            "There is a file log.txt in the current directory. Print the NUMBER OF "
            "LINES it contains (just the number).\n"
            "There is no shell: pipes and tools like wc do not work. You must use a "
            "single python command of the form:\n  python -c \"...\"\n"
            "Reply with ONLY that command, no explanation."
        ),
    ),
    Modality(
        name="code_second_largest",
        kind="code (Python)",
        make_env=lambda: CodeTaskEnv({"solution.py": CODE_STUB},
                                     "second largest distinct value", _code_grader),
        parse=parse_code,
        instruction=(
            "Complete this Python file:\n\n" + CODE_STUB + "\n"
            "Implement solve so it returns the second largest DISTINCT value in the "
            "list xs. Reply with ONLY the full Python file, no explanation."
        ),
        max_completion_length=160,
    ),
]


def run_modality(mod: Modality, tokenizer) -> dict:  # pragma: no cover - GPU + trl
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM
    from trl import GRPOTrainer

    reward_fn = env_reward_func(mod.make_env, parse_completion=mod.parse)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to("cuda")

    print(f"\n=== {mod.name} — {mod.kind} ===")
    before = accuracy(model, tokenizer, n=EVAL_N, reward_fn=reward_fn, instruction=mod.instruction)
    print(f"  baseline: {before:.0%}")

    dataset = Dataset.from_list([{"prompt": prompt_messages(mod.instruction)}] * (MAX_STEPS * NUM_GEN))
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_fn,
        args=grpo_config(f"runtime/xmodal-{mod.name}", max_steps=MAX_STEPS,
                         num_generations=NUM_GEN, max_completion_length=mod.max_completion_length),
        train_dataset=dataset,
        peft_config=lora_config(),
    )
    trainer.train()

    after = accuracy(trainer.model, tokenizer, n=EVAL_N, show=1,
                     reward_fn=reward_fn, instruction=mod.instruction)
    print(f"  trained : {after:.0%}")

    curve = [round(float(h["reward"]), 4) for h in trainer.state.log_history if "reward" in h]
    del trainer, model
    gc.collect()
    torch.cuda.empty_cache()
    return {"name": mod.name, "kind": mod.kind, "baseline": before, "trained": after,
            "reward_curve": curve}


def main() -> None:  # pragma: no cover - needs a GPU + the RL stack
    from transformers import AutoTokenizer

    print(f"loading tokenizer for {MODEL} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    results = {"model": MODEL, "max_steps": MAX_STEPS, "eval_n": EVAL_N, "num_generations": NUM_GEN,
               "tasks": []}
    for mod in MODALITIES:
        results["tasks"].append(run_modality(mod, tokenizer))
        OUT.write_text(json.dumps(results, indent=2))
        print(f"  wrote {OUT}")

    print("\n=== cross-modality summary ===")
    for r in results["tasks"]:
        print(f"  {r['name']:20s} {r['kind']:18s} {r['baseline']:>4.0%} -> {r['trained']:>4.0%}")


if __name__ == "__main__":
    main()
