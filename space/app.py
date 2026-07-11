"""Crucible — a Hugging Face Space.

An in-browser taste of Crucible: pick a scenario, run a scripted agent through the
environment, and see the recorded trajectory *replayed byte-for-byte*. All the logic
is the tested `crucible` core; this file only wires it to Gradio widgets.

Run locally:  pip install -r requirements.txt && python app.py
"""

from __future__ import annotations

import gradio as gr

from crucible import replay, rollout
from crucible.envs import GuessEnv, SQLTaskEnv


# Small scripted agents, inlined so the Space needs only `crucible` (not the repo's
# examples package). A real agent implements the same reset/act protocol.
class BinarySearchAgent:
    def reset(self):
        self.low = self.high = self.last = None

    def act(self, observation):
        fb = observation.get("feedback")
        if fb == "start":
            self.low, self.high = observation["low"], observation["high"]
        elif fb == "higher":
            self.low = (self.last or 0) + 1
        elif fb == "lower":
            self.high = (self.last or 0) - 1
        self.last = (self.low + self.high) // 2
        return self.last


class ScriptedAgent:
    def __init__(self, actions):
        self.actions = list(actions)
        self.i = 0

    def reset(self):
        self.i = 0

    def act(self, observation):
        action = self.actions[self.i % len(self.actions)]
        self.i += 1
        return action

# --- scenarios: (factory, agent) pairs, all deterministic so the demo is reliable ---

SQL_SCHEMA = (
    "CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT);"
    "CREATE TABLE orders (id INTEGER PRIMARY KEY, customer_id INTEGER, amount INTEGER);"
)
SQL_SEED = (
    "INSERT INTO customers VALUES (1,'Ada'),(2,'Lin'),(3,'Bo');"
    "INSERT INTO orders VALUES (1,1,100),(2,1,50),(3,2,200),(4,3,30);"
)
SQL_RIGHT = (
    "SELECT c.name, SUM(o.amount) FROM customers c "
    "JOIN orders o ON o.customer_id = c.id GROUP BY c.id ORDER BY SUM(o.amount) DESC"
)
SQL_EXPECTED = [["Lin", 200], ["Ada", 150], ["Bo", 30]]


def _sql_scenario():
    env = SQLTaskEnv(SQL_SCHEMA, SQL_SEED, "total per customer, highest first", SQL_EXPECTED)
    return env, ScriptedAgent(["SELECT name FROM customers", SQL_RIGHT])


SCENARIOS = {
    "GuessEnv — binary-search agent": lambda: (GuessEnv(), BinarySearchAgent()),
    "SQLTaskEnv — real SQLite (wrong then right)": _sql_scenario,
}


def run_scenario(name: str) -> str:
    env, agent = SCENARIOS[name]()
    traj = rollout(env, agent, seed=42, max_steps=12)

    # Rebuild a fresh environment of the same kind to replay against.
    fresh_env, _ = SCENARIOS[name]()
    report = replay(fresh_env, traj)

    lines = [
        f"environment : {traj.env_id}",
        f"seed        : {traj.seed}",
        f"steps       : {traj.steps}",
        f"total reward: {traj.total_reward:+.2f}",
        f"fingerprint : {traj.fingerprint()[:16]}",
        f"replay      : {'reproduced OK ✅' if report.ok else 'MISMATCH: ' + '; '.join(report.mismatches)}",
        "",
        "steps:",
    ]
    for i, t in enumerate(traj.transitions):
        lines.append(f"  {i}: action={t.action!r}  reward={t.reward:+.2f}  done={t.done}")
    return "\n".join(lines)


with gr.Blocks(title="Crucible") as demo:
    gr.Markdown(
        "# Crucible\n"
        "Turn any real software into a trainable, gradable, **replayable** RL "
        "environment for AI agents. Pick a scenario, run an agent, and watch the "
        "episode reproduce byte-for-byte.\n\n"
        "[Code on GitHub](https://github.com/nadeauglenn1-max/crucible)"
    )
    choice = gr.Dropdown(choices=list(SCENARIOS), value=list(SCENARIOS)[0], label="Scenario")
    run_btn = gr.Button("Forge & replay", variant="primary")
    output = gr.Code(label="Trajectory + replay", language=None)
    run_btn.click(run_scenario, inputs=choice, outputs=output)

if __name__ == "__main__":
    demo.launch()
