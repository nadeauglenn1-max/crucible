"""Watch a policy *learn* against a Crucible environment — on your laptop, no GPU.

Crucible grades; a trainer learns. This shows the whole loop end to end with the
simplest possible learner: a contextual bandit whose reward comes straight from a
Crucible environment. Over episodes its average reward climbs from chance to
near-optimal — the same **generate → grade → reinforce** loop GRPO runs at LLM scale,
just small enough to watch in a second.

    python -m examples.learn
"""

from __future__ import annotations

import random

from crucible import Environment, StepResult, rollout


class PickEnv(Environment):
    """A tiny learnable task. Each episode presents a *context* (0..k-1); the agent
    picks one of n actions; reward is 1.0 iff it picks the context's hidden correct
    action. The answers are fixed per environment, so the context→action mapping is
    there to be learned — chance alone scores 1/n."""

    def __init__(self, k_contexts: int = 5, n_actions: int = 4, answer_seed: int = 0) -> None:
        self.k = k_contexts
        self.n = n_actions
        rng = random.Random(answer_seed)
        self.answers = [rng.randrange(n_actions) for _ in range(k_contexts)]
        self._ctx = 0

    def reset(self, seed: int):
        self._ctx = seed % self.k  # the seed selects which context this episode shows
        return self._ctx

    def step(self, action):
        correct = int(action) == self.answers[self._ctx]
        return StepResult(observation=self._ctx, reward=1.0 if correct else 0.0, done=True)


class QAgent:
    """Epsilon-greedy learner: tracks the average reward of each (context, action) and
    picks the best-so-far, exploring occasionally. It learns *between* episodes from
    the recorded trajectory — the reward is the only teacher."""

    def __init__(self, n_actions: int, epsilon: float = 0.1, seed: int = 0) -> None:
        self.n = n_actions
        self.eps = epsilon
        self.rng = random.Random(seed)
        self.q: dict[int, list[list[float]]] = {}

    def reset(self) -> None:
        pass  # keep what it has learned across episodes

    def _row(self, ctx: int) -> list[list[float]]:
        return self.q.setdefault(ctx, [[0.0, 0] for _ in range(self.n)])

    def act(self, observation):
        row = self._row(observation)
        if self.rng.random() < self.eps:
            return self.rng.randrange(self.n)
        avgs = [total / count if count else 0.0 for total, count in row]
        return max(range(self.n), key=lambda a: avgs[a])

    def learn(self, trajectory) -> None:
        for t in trajectory.transitions:
            row = self._row(t.observation)
            row[t.action][0] += t.reward
            row[t.action][1] += 1


def train(env: Environment, agent: QAgent, episodes: int) -> list[float]:
    """Run the full loop: each episode Crucible grades the agent's choice, and the
    agent learns from the reward. Returns the per-episode rewards."""
    rewards = []
    for episode in range(episodes):
        traj = rollout(env, agent, seed=episode, max_steps=1)
        agent.learn(traj)
        rewards.append(traj.total_reward)
    return rewards


def _avg(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def main() -> None:
    learner = QAgent(n_actions=4, epsilon=0.1, seed=0)
    learned = _avg(train(PickEnv(answer_seed=7), learner, episodes=3000)[-300:])

    # Control: a policy that never learns (always random) stays at chance.
    baseline = _avg(train(PickEnv(answer_seed=7), QAgent(n_actions=4, epsilon=1.0, seed=1), episodes=3000))

    print("Watch a policy learn against a Crucible environment (no GPU)")
    print("  task     : pick the right action for each of 5 contexts (chance = 1/4 = 0.25)")
    print(f"  random   : avg reward, never-learning policy = {baseline:.2f}")
    print(f"  learned  : avg reward, epsilon-greedy learner = {learned:.2f}")
    print("  -> the environment's reward signal alone taught the policy the mapping.")


if __name__ == "__main__":
    main()
