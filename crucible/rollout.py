"""Driving agents through environments, and verifying the result by replay.

``rollout`` runs an agent through an environment and records a Trajectory.
``replay`` re-runs that trajectory's actions against a fresh environment and checks
that the world reproduces exactly — the property that makes Crucible trajectories
reproducible training data and auditable rewards.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from .env import Action, Environment, Observation
from .trajectory import Trajectory, Transition


@runtime_checkable
class Agent(Protocol):
    """Anything that can act. The thing Crucible exists to train and grade — kept
    deliberately outside the core so any policy (scripted, an LLM, a learned model)
    plugs in."""

    def reset(self) -> None:
        """Prepare for a new episode."""

    def act(self, observation: Observation) -> Action:
        """Choose an action given the current observation."""


def rollout(
    env: Environment,
    agent: Agent,
    *,
    seed: int,
    max_steps: int = 100,
) -> Trajectory:
    """Run ``agent`` through ``env`` for one episode and record a Trajectory.

    The loop is the familiar reset/act/step cycle; the point is what it captures —
    every observation the agent acted on, its action, the reward, and the
    environment's post-step digest — so the episode can be replayed and audited.
    """
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")

    agent.reset()
    observation = env.reset(seed)
    traj = Trajectory(
        env_id=env.name(),
        seed=seed,
        initial_observation=observation,
        env_config=env.config(),
    )

    for _ in range(max_steps):
        action = agent.act(observation)
        result = env.step(action)
        traj.add(
            Transition(
                observation=observation,
                action=action,
                reward=result.reward,
                done=result.done,
                info=result.info,
                digest=env.digest(),
            )
        )
        observation = result.observation
        if result.done:
            break

    return traj


@dataclass
class ReplayReport:
    """The result of replaying a trajectory: whether the world reproduced exactly,
    how many steps were checked, and every mismatch found (empty when ``ok``)."""

    ok: bool
    steps: int
    mismatches: list[str] = field(default_factory=list)


def replay(env: Environment, traj: Trajectory) -> ReplayReport:
    """Re-run ``traj``'s actions against a fresh ``env`` and verify reproducibility.

    A fresh environment is reset with the trajectory's seed, then each recorded
    action is applied and its reward, done-flag, and state digest are compared to the
    record. Any divergence is reported — loudly, never hidden — because a
    non-reproducible environment is a bug, not a footnote.
    """
    mismatches: list[str] = []
    observation = env.reset(traj.seed)
    if observation != traj.initial_observation:
        mismatches.append(
            f"initial observation: replay {observation!r} != recorded {traj.initial_observation!r}"
        )

    for i, t in enumerate(traj.transitions):
        result = env.step(t.action)
        if result.reward != t.reward:
            mismatches.append(f"step {i}: reward {result.reward} != recorded {t.reward}")
        if result.done != t.done:
            mismatches.append(f"step {i}: done {result.done} != recorded {t.done}")
        digest = env.digest()
        if digest != t.digest:
            mismatches.append(f"step {i}: digest {digest!r} != recorded {t.digest!r}")
        if result.done:
            # The trajectory should end exactly here; any trailing transitions mean
            # the recorded episode and the replay diverged in length.
            if i != len(traj.transitions) - 1:
                mismatches.append(
                    f"step {i}: replay ended early ({i + 1}/{len(traj.transitions)} steps)"
                )
            break

    return ReplayReport(ok=not mismatches, steps=traj.steps, mismatches=mismatches)
