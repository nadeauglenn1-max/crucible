"""Real GRPO fine-tune where the reward *is* a Crucible environment.

This is the end-to-end proof of the TRL adapter: we take a small instruct model,
give it a SQL task, and let `trl.GRPOTrainer` improve it — with **no hand-labelled
data and no hand-written reward code**. The reward function is
`crucible.integrations.trl.env_reward_func` wrapping a `SQLTaskEnv`: the model writes
a query, Crucible runs it against real SQLite, and "did the rows match?" is the whole
signal. GRPO generates a group of completions per prompt, scores each with the
environment, and pushes the policy toward the ones that scored.

Run it (needs a GPU; ~8GB is enough with the LoRA + bf16 config here):

    pip install "trl>=0.15" peft transformers accelerate torch
    python -m examples.train_grpo

It prints accuracy on a held sample **before** and **after** training. The delta is
the point: a Crucible environment, unmodified, taught a model to do a real task.

Everything above `main()` is importable and dependency-light so it can be unit-tested
without pulling in torch/trl (see tests/test_train_grpo.py).
"""

from __future__ import annotations

from typing import Any, Callable

from crucible.envs import SQLTaskEnv
from crucible.integrations.trl import env_reward_func

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

# --- the task -------------------------------------------------------------------
# One real, verifiable SQL task. Small enough that a 0.5B model gets it *sometimes*
# at baseline (so there's signal to reinforce) but not reliably (so there's room to
# improve). The reward is entirely "run the query, compare the rows".

SCHEMA = (
    "CREATE TABLE employees ("
    "  id INTEGER PRIMARY KEY,"
    "  name TEXT,"
    "  department TEXT,"
    "  salary INTEGER"
    ");"
)
SEED = (
    "INSERT INTO employees VALUES "
    "(1, 'Ada',   'Engineering', 120000),"
    "(2, 'Lin',   'Engineering', 135000),"
    "(3, 'Bo',    'Sales',        90000),"
    "(4, 'Cy',    'Sales',       105000),"
    "(5, 'Dev',   'Marketing',    80000);"
)
TASK = "the name and salary of the employee with the SECOND highest salary"
EXPECTED = [["Ada", 120000]]
COLUMNS = "name, salary"


def build_instruction(task_text: str, columns: str) -> str:
    """The conversational instruction for a SQL task over the shared schema."""
    return (
        "You are given a SQLite database with this schema:\n\n"
        f"{SCHEMA}\n\n"
        f"Write a single SQL query that returns {task_text}.\n"
        f"Return columns in the order: {columns}.\n"
        "Reply with ONLY the SQL query, no explanation, no markdown."
    )


INSTRUCTION = build_instruction(TASK, COLUMNS)


def make_env() -> SQLTaskEnv:
    """A fresh instance of the task environment (GRPO resets one per completion)."""
    return SQLTaskEnv(SCHEMA, SEED, TASK, EXPECTED)


def extract_sql(completion: Any) -> str:
    """Pull a bare SQL string out of a model completion.

    Handles both TRL completion shapes — a plain string, or the conversational
    ``[{"role": "assistant", "content": ...}]`` list — and strips ```sql fences and a
    trailing semicolon so the query reaches the environment clean.
    """
    if isinstance(completion, list):
        text = completion[-1]["content"] if completion else ""
    else:
        text = completion or ""
    text = str(text).strip()

    if "```" in text:
        # take the content of the first fenced block
        block = text.split("```", 2)[1]
        if block.lower().startswith("sql"):
            block = block[3:]
        text = block.strip()

    # some models still prefix a label
    for label in ("SQL:", "Query:", "sql:"):
        if text.startswith(label):
            text = text[len(label):].strip()

    return text.rstrip(";").strip()


# The reward function GRPO calls: env_reward_func resets a fresh SQLTaskEnv per
# completion, runs extract_sql(completion) against it, and returns the step reward.
sql_reward = env_reward_func(make_env, parse_completion=extract_sql)


def prompt_messages(instruction: str = INSTRUCTION) -> list[dict]:
    """The conversational prompt GRPO trains on (one task → many generations)."""
    return [{"role": "user", "content": instruction}]


def accuracy(
    model,
    tokenizer,
    n: int = 20,
    *,
    temperature: float = 0.9,
    show: int = 0,
    reward_fn: Callable[..., list[float]] = sql_reward,
    instruction: str = INSTRUCTION,
) -> float:
    """Fraction of `n` sampled completions that solve the task (reward > 0).

    Uses the same env reward the trainer uses, so "accuracy" here means exactly
    "the environment paid out". Imported by main() for the before/after measurement.
    `show` prints that many (completion, extracted SQL, reward) samples for diagnosis.
    `reward_fn`/`instruction` let the generalization suite reuse this for other tasks.
    """
    import torch

    # Generation must not run under the training-time gradient-checkpointing/no-cache
    # regime, or the model degrades. Put it in eval mode with the KV cache on.
    was_training = model.training
    model.eval()
    model.config.use_cache = True

    messages = prompt_messages(instruction)
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    solved = 0
    for i in range(n):
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,
                temperature=temperature,
                top_p=0.95,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        completion = tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        # score it through the environment — the one source of truth for "correct"
        reward = reward_fn(completions=[completion])[0]
        if reward > 0:
            solved += 1
        if i < show:
            print(f"    [{reward:+.1f}] {extract_sql(completion)!r}")

    if was_training:
        model.train()
    return solved / n


def grpo_config(
    output_dir: str,
    max_steps: int = 80,
    *,
    num_generations: int = 8,
    max_completion_length: int = 100,
):
    """The GRPO hyperparameters, sized for a 0.5B model on an 8GB laptop GPU.

    Factored out so the generalization suite trains every task the same way. Larger
    models / longer completions can lower `num_generations` to fit the same VRAM.
    Note: trl's GRPOConfig has no `max_prompt_length` — passing it raises TypeError.
    """
    from trl import GRPOConfig

    return GRPOConfig(
        output_dir=output_dir,
        num_generations=num_generations,
        per_device_train_batch_size=num_generations,
        gradient_accumulation_steps=1,
        max_completion_length=max_completion_length,
        temperature=0.9,
        learning_rate=1e-5,
        max_steps=max_steps,
        logging_steps=1,
        save_strategy="no",
        bf16=True,
        gradient_checkpointing=True,
        report_to="none",
    )


def lora_config():
    """LoRA on the attention projections — the only trainable weights."""
    from peft import LoraConfig

    return LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )


def main() -> None:  # pragma: no cover - needs a GPU + the RL stack
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOTrainer

    print(f"loading {MODEL} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16)
    model.to("cuda")

    print("measuring baseline accuracy (n=20) ...")
    before = accuracy(model, tokenizer, n=20, show=3)
    print(f"  baseline: {before:.0%}")

    # GRPO trains on prompts; the same task repeated is fine — it generates a fresh
    # group of completions per step and scores each through the environment.
    dataset = Dataset.from_list([{"prompt": prompt_messages()}] * 640)

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=sql_reward,
        args=grpo_config("runtime/grpo-sql", max_steps=80),
        train_dataset=dataset,
        peft_config=lora_config(),
    )
    print("training ...")
    trainer.train()

    trainer.model.save_pretrained("runtime/grpo-sql/adapter")
    print("measuring post-training accuracy (n=20) ...")
    after = accuracy(trainer.model, tokenizer, n=20, show=3)

    print("\n=== Crucible GRPO result ===")
    print(f"  baseline : {before:.0%}")
    print(f"  trained  : {after:.0%}")
    print(f"  delta    : {after - before:+.0%}")


if __name__ == "__main__":
    main()
