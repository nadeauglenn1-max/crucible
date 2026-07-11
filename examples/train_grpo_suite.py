"""Generalization battery: prove the GRPO result isn't a one-off.

`train_grpo.py` shows one task go 5% -> 100%. The obvious skeptic's question is
"cherry-picked task?". This trains a **fresh** copy of the same 0.5B model, from the
same pretrained weights, on several *distinct* SQL tasks — different skills, different
baselines — each with a Crucible `SQLTaskEnv` as the only reward. If they all climb,
it's the method that works, not one lucky task.

Each task reuses the exact training machinery from `train_grpo` (same GRPO/LoRA config,
same eval), so this file only supplies the tasks and the loop.

    pip install "trl>=0.15" peft transformers accelerate torch   # + a CUDA GPU (~8GB)
    python -m examples.train_grpo_suite

It writes per-task baseline/trained accuracy + reward curves to
`examples/results/grpo_suite.json`; render the chart (no GPU) with
`python -m examples.results.plot_grpo_suite`.
"""

from __future__ import annotations

import gc
import json
from dataclasses import dataclass
from pathlib import Path

from crucible.envs import SQLTaskEnv
from crucible.integrations.trl import env_reward_func
from examples.train_grpo import (
    MODEL,
    SCHEMA,
    SEED,
    accuracy,
    build_instruction,
    extract_sql,
    grpo_config,
    lora_config,
    prompt_messages,
)

MAX_STEPS = 60
EVAL_N = 20
OUT = Path(__file__).resolve().parent / "results" / "grpo_suite.json"


@dataclass
class Task:
    name: str
    ask: str          # the natural-language task, dropped into the instruction
    columns: str      # required output column order
    expected: list    # the exact correct rows (the whole reward)
    skill: str        # short label for the chart


# Four distinct SQL skills over the shared employees table. Expected rows are the
# ground truth the environment checks against — computed by hand from SEED.
TASKS = [
    Task(
        name="second_highest",
        ask="the name and salary of the employee with the SECOND highest salary",
        columns="name, salary",
        expected=[["Ada", 120000]],
        skill="subquery / OFFSET",
    ),
    Task(
        name="dept_payroll",
        ask="the total salary paid per department, highest total first, "
            "returned as department and total",
        columns="department, total",
        expected=[["Engineering", 255000], ["Sales", 195000], ["Marketing", 80000]],
        skill="GROUP BY + SUM + ORDER",
    ),
    Task(
        name="dept_headcount",
        ask="each department and its employee count, only for departments with more "
            "than one employee, ordered by department name, as department and count",
        columns="department, count",
        expected=[["Engineering", 2], ["Sales", 2]],
        skill="GROUP BY + HAVING",
    ),
    Task(
        name="dept_avg_top",
        ask="the department with the highest average salary and that average, "
            "returned as department and avg_salary",
        columns="department, avg_salary",
        expected=[["Engineering", 127500.0]],
        skill="GROUP BY + AVG",
    ),
]


def run_task(task: Task, tokenizer) -> dict:  # pragma: no cover - needs a GPU + trl
    """Train one fresh model on one task; return baseline/trained/curve/learned query."""
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM
    from trl import GRPOTrainer

    instruction = build_instruction(task.ask, task.columns)

    def make_env():
        return SQLTaskEnv(SCHEMA, SEED, task.ask, task.expected)

    reward_fn = env_reward_func(make_env, parse_completion=extract_sql)

    # Fresh pretrained weights every task — no leakage between runs.
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to("cuda")

    print(f"\n=== {task.name} ({task.skill}) ===")
    before = accuracy(model, tokenizer, n=EVAL_N, reward_fn=reward_fn, instruction=instruction)
    print(f"  baseline: {before:.0%}")

    dataset = Dataset.from_list([{"prompt": prompt_messages(instruction)}] * (MAX_STEPS * 8))
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_fn,
        args=grpo_config(f"runtime/grpo-{task.name}", max_steps=MAX_STEPS),
        train_dataset=dataset,
        peft_config=lora_config(),
    )
    trainer.train()

    after = accuracy(
        trainer.model, tokenizer, n=EVAL_N, show=1, reward_fn=reward_fn, instruction=instruction
    )
    print(f"  trained : {after:.0%}")

    curve = [round(float(h["reward"]), 4) for h in trainer.state.log_history if "reward" in h]
    learned = _sample_query(trainer.model, tokenizer, instruction)

    # free the GPU before the next task
    del trainer, model
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "name": task.name,
        "skill": task.skill,
        "ask": task.ask,
        "baseline": before,
        "trained": after,
        "learned_query": learned,
        "reward_curve": curve,
    }


def _sample_query(model, tokenizer, instruction) -> str:  # pragma: no cover - GPU
    """One low-temperature sample, to record what the trained policy converged to."""
    import torch

    model.eval()
    model.config.use_cache = True
    text = tokenizer.apply_chat_template(
        prompt_messages(instruction), tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=100, do_sample=True, temperature=0.3, top_p=0.95,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    completion = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return extract_sql(completion)


def main() -> None:  # pragma: no cover - needs a GPU + the RL stack
    from transformers import AutoTokenizer

    print(f"loading tokenizer for {MODEL} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    results = {"model": MODEL, "max_steps": MAX_STEPS, "eval_n": EVAL_N, "tasks": []}
    for task in TASKS:
        results["tasks"].append(run_task(task, tokenizer))
        OUT.write_text(json.dumps(results, indent=2))  # checkpoint after every task
        print(f"  wrote {OUT}")

    print("\n=== generalization summary ===")
    for r in results["tasks"]:
        print(f"  {r['name']:16s} {r['skill']:22s} {r['baseline']:>4.0%} -> {r['trained']:>4.0%}")


if __name__ == "__main__":
    main()
