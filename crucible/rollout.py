"""Driving agents through environments, and verifying the result by replay.

``rollout`` runs an agent through an environment and records a Trajectory.
``replay`` re-runs that trajectory's actions against a fresh environment and checks
that the world reproduces exactly — the property that makes Crucible trajectories
reproducible training data and auditable rewards.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from .env import Action, Environment, Observation
from .trajectory import UNRECORDED, Trajectory, Transition


def _same(a: object, b: object) -> bool:
    """Whether two observations are the same value.

    A trajectory loaded from disk has been through JSON, which turns tuples into
    lists, int dict-keys into strings, and so on. A fresh environment produces the
    native objects. So a fast identity/equality check is backed by a canonical-JSON
    comparison: replaying a *saved* trajectory is then exactly as faithful as
    replaying one in memory — reproducibility must not depend on whether the record
    took a round-trip through disk. Values that aren't JSON-serializable (and so
    could never be in a saved trajectory) fall back to plain inequality.
    """
    if a == b:
        return True
    try:
        return json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    except TypeError:
        return False


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

    # The last observation is part of the episode too — it is what the agent was
    # left looking at, and for many tasks it is where the answer is. Recording it
    # is what lets replay verify the episode all the way to its end.
    traj.final_observation = observation
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
    action is applied and its reward, done-flag, ``info`` and state digest are
    compared to the record. Any divergence is reported — loudly, never hidden —
    because a non-reproducible environment is a bug, not a footnote.

    The rule is that **every claim the trajectory makes gets bound**: the world it
    says it ran in, the numbers it reports, the evidence it carries, and the
    observations at both ends of the episode. A recorded field that nothing verifies
    is not a harmless omission — it is the one field worth tampering with.
    """
    mismatches = list(traj.integrity_mismatches())

    # Bind the world first: verifying a faithful replay of the *wrong* environment
    # is a green tick that means nothing.
    if env.name() != traj.env_id:
        mismatches.append(
            f"environment: replaying {env.name()!r} against a trajectory "
            f"recorded in {traj.env_id!r}"
        )
    if not _same(env.config(), traj.env_config):
        mismatches.append(
            f"environment config: replay {env.config()!r} != recorded {traj.env_config!r}"
        )

    observation = env.reset(traj.seed)
    if not _same(observation, traj.initial_observation):
        mismatches.append(
            f"initial observation: replay {observation!r} != recorded {traj.initial_observation!r}"
        )

    for i, t in enumerate(traj.transitions):
        # The observation the agent acted on must reproduce too — otherwise
        # "byte-for-byte" would be a claim about rewards and digests only.
        if not _same(observation, t.observation):
            mismatches.append(f"step {i}: observation diverged from the record")
        result = env.step(t.action)
        if result.reward != t.reward:
            mismatches.append(f"step {i}: reward {result.reward} != recorded {t.reward}")
        if result.done != t.done:
            mismatches.append(f"step {i}: done {result.done} != recorded {t.done}")
        # `info` is where a Rubric puts its per-criterion breakdown — the *reason* for
        # the reward. Leaving it unbound would make the evidence the one part of the
        # record anyone could rewrite. It is a deterministic function of seed and
        # actions like everything else, so it is held to the same standard.
        if not _same(result.info, t.info):
            mismatches.append(f"step {i}: info {result.info!r} != recorded {t.info!r}")
        digest = env.digest()
        if digest != t.digest:
            mismatches.append(f"step {i}: digest {digest!r} != recorded {t.digest!r}")
        observation = result.observation
        if result.done:
            # The trajectory should end exactly here; any trailing transitions mean
            # the recorded episode and the replay diverged in length.
            if i != len(traj.transitions) - 1:
                mismatches.append(
                    f"step {i}: replay ended early ({i + 1}/{len(traj.transitions)} steps)"
                )
            break

    # A v1/v2 file never recorded a final observation; there is nothing to check, and
    # we say nothing rather than manufacture a verdict about it.
    if traj.final_observation is not UNRECORDED and not _same(
        observation, traj.final_observation
    ):
        mismatches.append(
            f"final observation: replay {observation!r} != recorded {traj.final_observation!r}"
        )

    return ReplayReport(ok=not mismatches, steps=traj.steps, mismatches=mismatches)
