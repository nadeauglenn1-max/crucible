"""Render the GRPO-on-a-Crucible-environment result chart from the recorded run.

Reads `grpo_sql_rewards.json` (the data captured from `python -m examples.train_grpo`)
and writes `docs/assets/grpo_sql.png` — the before/after bars plus the per-step reward
curve. Kept separate from the training script so the chart is reproducible from data
without a GPU:

    pip install matplotlib
    python -m examples.results.plot_grpo
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
DATA = HERE / "grpo_sql_rewards.json"
OUT = ROOT / "docs" / "assets" / "grpo_sql.png"

INK = "#1b1b2f"
ACCENT = "#e94560"
GOOD = "#2ec4b6"
MUTE = "#8d99ae"


def main() -> None:
    d = json.loads(DATA.read_text())
    curve = d["mean_group_reward_per_step"]
    before = d["baseline_accuracy"] * 100
    after = d["trained_accuracy"] * 100

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4), width_ratios=[1, 1.7])
    fig.suptitle(
        "A 0.5B model learns a real SQL task — taught only by a Crucible environment",
        fontsize=13, fontweight="bold", color=INK,
    )

    # --- before / after bars ---
    bars = ax1.bar(
        ["before\ntraining", "after\n80 GRPO steps"],
        [before, after],
        color=[MUTE, GOOD], width=0.6,
    )
    ax1.set_ylim(0, 108)
    ax1.set_ylabel("task solved (%)", color=INK)
    ax1.set_title("environment-scored accuracy", fontsize=10, color=INK)
    for bar, val in zip(bars, (before, after)):
        ax1.text(bar.get_x() + bar.get_width() / 2, val + 2, f"{val:.0f}%",
                 ha="center", va="bottom", fontweight="bold", color=INK)
    ax1.spines[["top", "right"]].set_visible(False)

    # --- reward curve ---
    ax2.plot(range(1, len(curve) + 1), curve, color=ACCENT, lw=2)
    ax2.axhline(1.0, color=GOOD, ls="--", lw=1, alpha=0.7)
    ax2.set_ylim(-0.15, 1.1)
    ax2.set_xlim(1, len(curve))
    ax2.set_xlabel("GRPO step", color=INK)
    ax2.set_ylabel("mean reward / group", color=INK)
    ax2.set_title("reward climbs as the policy discovers the query", fontsize=10, color=INK)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.text(
        len(curve), 0.02,
        f"learned:  {d['learned_query']}",
        ha="right", va="bottom", fontsize=8, color=INK, family="monospace",
        bbox=dict(boxstyle="round,pad=0.4", fc="#f4f4f8", ec=MUTE, lw=0.8),
    )

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, facecolor="white")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
