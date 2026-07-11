# Result: a model learns a real task from a Crucible environment

This is the end-to-end proof of the whole thesis — **a Crucible environment, unmodified,
used directly as a reinforcement-learning reward, taught a language model to do a real
task with no labelled data and no hand-written reward code.**

![GRPO on a Crucible SQL environment](../../docs/assets/grpo_sql.png)

## What happened

- **Model:** `Qwen/Qwen2.5-0.5B-Instruct` (LoRA `r=16` on the attention projections, bf16).
- **Task:** given a real SQLite schema, write the query returning *the second-highest-paid
  employee*. Not trivial for a 0.5B model — it needs the `LIMIT offset, count` idiom.
- **Reward = the environment.** `crucible.envs.SQLTaskEnv` runs the model's query against a
  real database and pays `+1.0` only if the rows match exactly, `-0.1` otherwise. That is
  the entire training signal. The bridge is one line —
  `env_reward_func(make_env, parse_completion=extract_sql)` from
  [`crucible/integrations/trl.py`](../../crucible/integrations/trl.py).
- **Trainer:** `trl.GRPOTrainer`, 8 generations/step, 80 steps, on an RTX 5070 Laptop (8GB).

| | task solved |
| --- | --- |
| before training | **5%** |
| after 80 GRPO steps | **100%** |

The model **discovered** the query on its own, guided only by the environment's reward:

```sql
SELECT name, salary FROM employees ORDER BY salary DESC LIMIT 1, 1
```

Accuracy here is measured *by the same environment* — "the reward paid out" — so the metric
and the training signal are the identical, auditable check. That is the point Crucible
exists to make: **the environment is the reward, and it is verifiable.**

## Reproduce it

```bash
pip install "trl>=0.15" peft transformers accelerate torch   # + a CUDA GPU (~8GB)
python -m examples.train_grpo        # trains, prints before/after, saves the LoRA adapter
```

The run prints its baseline and post-training accuracy and writes the LoRA adapter to
`runtime/grpo-sql/adapter`. The exact per-step rewards from the recorded run are in
[`grpo_sql_rewards.json`](grpo_sql_rewards.json); regenerate the chart from that data
(no GPU needed) with:

```bash
pip install matplotlib
python -m examples.results.plot_grpo        # writes docs/assets/grpo_sql.png
```

> Note on measurement: generation for the before/after eval runs with the model in
> `eval()` mode and the KV cache on. Evaluating while the trainer's
> gradient-checkpointing/no-cache regime is still active silently degrades generation and
> will read a false 0% — the accuracy check in `train_grpo.py` restores eval mode first.
