"""Export trajectories to training-ready records.

RL trainers — TRL's GRPO/PPO trainers, verifiers, prime-rl — all consume, per step,
roughly the same triple: the **prompt** the policy saw, the **completion** it
produced, and the scalar **reward**. That is exactly a `Transition`. This module
flattens trajectories into that shape as plain dicts and JSONL, with a `render` hook
so rich observations and actions become the text a trainer wants.

Keeping the output plain dicts / JSONL means **zero dependency on any specific
trainer**: a thin, trainer-specific adapter maps these neutral records onto a
concrete schema (e.g. a TRL dataset) without this module — or the core — taking on
that dependency.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Iterable

from .trajectory import Trajectory

Render = Callable[[Any], str]


def _default_render(value: Any) -> str:
    """Text left as-is; anything else canonically JSON-encoded."""
    return value if isinstance(value, str) else json.dumps(value, sort_keys=True)


def to_records(
    traj: Trajectory,
    *,
    render_observation: Render | None = None,
    render_action: Render | None = None,
) -> list[dict]:
    """Flatten one trajectory into ``{prompt, completion, reward, ...}`` records — one
    per step. Rich observations/actions are rendered to text by the hooks (default:
    strings as-is, everything else canonical JSON)."""
    render_obs = render_observation or _default_render
    render_act = render_action or _default_render
    records = []
    for step, t in enumerate(traj.transitions):
        records.append(
            {
                "env_id": traj.env_id,
                "seed": traj.seed,
                "step": step,
                "prompt": render_obs(t.observation),
                "completion": render_act(t.action),
                "reward": t.reward,
            }
        )
    return records


def write_jsonl(
    trajectories: Iterable[Trajectory],
    path: str | Path,
    *,
    render_observation: Render | None = None,
    render_action: Render | None = None,
) -> int:
    """Write the flattened records of many trajectories to a JSONL file (one record
    per line). Returns the number of records written — the training-example count."""
    lines = []
    for traj in trajectories:
        for record in to_records(
            traj,
            render_observation=render_observation,
            render_action=render_action,
        ):
            lines.append(json.dumps(record, sort_keys=True))
    Path(path).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return len(lines)
