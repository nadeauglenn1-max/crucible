"""Render the generalization-battery chart from the recorded suite run.

Reads `grpo_suite.json` (from `python -m examples.train_grpo_suite`) and writes
`docs/assets/grpo_suite.png` — before/after bars for every task plus each task's
reward curve, the evidence that the GRPO result generalizes across SQL skills.
GPU-free:

    pip install matplotlib
    python -m examples.results.plot_grpo_suite
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
DATA = HERE / "grpo_suite.json"
OUT = ROOT / "docs" / "assets" / "grpo_suite.png"

INK = "#1b1b2f"
MUTE = "#8d99ae"
GOOD = "#2ec4b6"
CURVE = ["#e94560", "#f5a623", "#3a86ff", "#8338ec"]


def main() -> None:
    d = json.loads(DATA.read_text())
    tasks = d["tasks"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8), width_ratios=[1.15, 1])
    fig.suptitle(
        "Not a one-off: four SQL skills, four fresh models, each taught only by a "
        "Crucible environment",
        fontsize=13, fontweight="bold", color=INK,
    )
    fig.text(
        0.5, 0.925, f"{d['model']}  ·  LoRA + GRPO, {d['max_steps']} steps each",
        ha="center", va="top", fontsize=9, color=MUTE,
    )

    # --- grouped before/after bars ---
    import numpy as np

    x = np.arange(len(tasks))
    w = 0.38
    before = [t["baseline"] * 100 for t in tasks]
    after = [t["trained"] * 100 for t in tasks]
    ax1.bar(x - w / 2, before, w, label="before", color=MUTE)
    ax1.bar(x + w / 2, after, w, label="after 60 GRPO steps", color=GOOD)
    for xi, b, a in zip(x, before, after):
        ax1.text(xi - w / 2, b + 1.5, f"{b:.0f}", ha="center", va="bottom", fontsize=8, color=INK)
        ax1.text(xi + w / 2, a + 1.5, f"{a:.0f}", ha="center", va="bottom", fontsize=8,
                 fontweight="bold", color=INK)
    ax1.set_xticks(x)
    ax1.set_xticklabels([t["skill"].replace(" + ", "\n+ ") for t in tasks], fontsize=8)
    ax1.set_ylim(0, 108)
    ax1.set_ylabel("task solved (%)", color=INK)
    ax1.set_title("environment-scored accuracy", fontsize=10, color=INK)
    ax1.legend(frameon=False, fontsize=8, loc="upper left")
    ax1.spines[["top", "right"]].set_visible(False)

    # --- reward curves, one per task ---
    for t, c in zip(tasks, CURVE):
        curve = t["reward_curve"]
        ax2.plot(range(1, len(curve) + 1), curve, color=c, lw=1.6, label=t["name"])
    ax2.axhline(1.0, color=GOOD, ls="--", lw=1, alpha=0.6)
    ax2.set_ylim(-0.15, 1.12)
    ax2.set_xlim(1, max(len(t["reward_curve"]) for t in tasks))
    ax2.set_xlabel("GRPO step", color=INK)
    ax2.set_ylabel("mean reward / group", color=INK)
    ax2.set_title("reward climbs on every task", fontsize=10, color=INK)
    ax2.legend(frameon=False, fontsize=8, loc="lower right")
    ax2.spines[["top", "right"]].set_visible(False)

    fig.tight_layout(rect=(0, 0, 1, 0.90))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, facecolor="white")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
