"""The Trajectory: a self-contained, replayable record of one episode.

A trajectory is the artifact Crucible produces. It is enough, on its own, to re-run
an episode against a fresh environment and confirm — byte for byte — the same
observations, rewards, and state digests. That reproducibility is what makes a
trajectory shareable training data and an auditable reward.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

#: On-disk format versions. A saved trajectory is wrapped in an envelope carrying
#: the current version, so the format can evolve without silently misreading old
#: files. Version 2 added ``env_config`` (for CLI replay); version 3 added
#: ``final_observation`` (the last thing the agent saw, which v1/v2 dropped on the
#: floor). Older files still load — what they never recorded stays UNRECORDED.
FORMAT_VERSION = 3
_SUPPORTED_VERSIONS = (1, 2, 3)


class _Unrecorded:
    """The value of a field the record never carried.

    Distinct from ``None``, which is a legitimate observation. A v1/v2 file predates
    ``final_observation``; loading one must not invent a value that ``replay`` would
    then solemnly verify against thin air.
    """

    def __repr__(self) -> str:
        return "<unrecorded>"


#: Singleton marker — compare with ``is``.
UNRECORDED = _Unrecorded()


@dataclass
class Transition:
    """One recorded step: the observation the agent acted on, the action it took, and
    the environment's response — plus the environment's state digest *after* the
    step, so replay can verify the world, not just the numbers."""

    observation: Any
    action: Any
    reward: float
    done: bool
    info: dict
    digest: str


@dataclass
class Trajectory:
    """A full episode: the seed that determined it, the environment it ran in, the
    observations either side of the run, and the ordered transitions. Serializes to
    and from JSON so episodes are portable.

    Everything recorded here is *claim-bearing* — it is what the trajectory asserts
    happened — so ``replay`` binds all of it. A field the record carries but nothing
    verifies is a place to hide a lie.
    """

    env_id: str
    seed: int
    initial_observation: Any
    env_config: dict = field(default_factory=dict)
    transitions: list[Transition] = field(default_factory=list)
    total_reward: float = 0.0
    #: The observation left when the episode ended — the last thing the agent saw,
    #: and often where the answer is. Set by ``rollout``; UNRECORDED in v1/v2 files.
    final_observation: Any = UNRECORDED

    def add(self, transition: Transition) -> None:
        self.transitions.append(transition)
        self.total_reward += transition.reward

    @property
    def steps(self) -> int:
        return len(self.transitions)

    def integrity_mismatches(self) -> list[str]:
        """Everything the record can check about *itself*, with no environment in
        hand: the recorded ``total_reward`` must be the sum of the step rewards.

        This lives here, once, because both ``crucible show`` and ``replay`` need it.
        Two copies of one guarantee is how a file comes to pass one command and fail
        the other.
        """
        recomputed = sum(t.reward for t in self.transitions)
        if abs(recomputed - self.total_reward) < 1e-9:
            return []
        return [
            f"total reward: recorded {self.total_reward} "
            f"!= sum of step rewards {recomputed}"
        ]

    def to_dict(self) -> dict:
        # Hand-rolled rather than asdict() so an unrecorded final observation is
        # *absent* from the encoding, not serialized as a sentinel or as a null that
        # would be indistinguishable from a genuine None.
        data: dict[str, Any] = {
            "env_id": self.env_id,
            "seed": self.seed,
            "initial_observation": self.initial_observation,
            "env_config": self.env_config,
            "transitions": [asdict(t) for t in self.transitions],
            "total_reward": self.total_reward,
        }
        if self.final_observation is not UNRECORDED:
            data["final_observation"] = self.final_observation
        return data

    def to_json(self, *, indent: int | None = None) -> str:
        # sort_keys keeps the encoding canonical, so fingerprints are stable.
        return json.dumps(self.to_dict(), sort_keys=True, indent=indent)

    @classmethod
    def from_dict(cls, data: dict) -> "Trajectory":
        transitions = [Transition(**t) for t in data.get("transitions", [])]
        return cls(
            env_id=data["env_id"],
            seed=data["seed"],
            initial_observation=data["initial_observation"],
            env_config=data.get("env_config", {}),  # absent in v1 files
            transitions=transitions,
            total_reward=data.get("total_reward", 0.0),
            final_observation=data.get("final_observation", UNRECORDED),  # v3+
        )

    @classmethod
    def from_json(cls, text: str) -> "Trajectory":
        return cls.from_dict(json.loads(text))

    def fingerprint(self) -> str:
        """A content hash over the canonical JSON — a stable id for this exact
        episode. Two trajectories with the same fingerprint are the same episode."""
        return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()

    def save(self, path: str | Path) -> None:
        """Write the trajectory to disk as a versioned JSON envelope. The trajectory
        is the artifact the whole product exists to make; this is how it leaves
        memory to be shared, cached, or replayed later."""
        envelope = {"version": FORMAT_VERSION, "trajectory": self.to_dict()}
        Path(path).write_text(
            json.dumps(envelope, sort_keys=True, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: str | Path) -> "Trajectory":
        """Read a trajectory saved by ``save``, rejecting an unrecognized format
        version rather than silently misreading it."""
        envelope = json.loads(Path(path).read_text(encoding="utf-8"))
        version = envelope.get("version")
        if version not in _SUPPORTED_VERSIONS:
            raise ValueError(
                f"unsupported trajectory format version {version!r} "
                f"(this build reads versions {list(_SUPPORTED_VERSIONS)})"
            )
        return cls.from_dict(envelope["trajectory"])
