"""The test that matters: a policy actually *learns* from a Crucible reward.

This exercises the whole loop — environment grades, learner improves — and asserts
the reward climbs from chance to near-optimal. It's the CPU-runnable proof that
Crucible's reward signal is a real teacher; GRPO on an LLM is the same loop at scale.
"""

from examples.learn import PickEnv, QAgent, train


def _avg(xs):
    return sum(xs) / len(xs)


def test_policy_learns_from_the_environment_reward():
    # The learner: epsilon-greedy, updates from each episode's reward.
    learner = QAgent(n_actions=4, epsilon=0.1, seed=0)
    late = _avg(train(PickEnv(answer_seed=7), learner, episodes=3000)[-300:])

    # The control: a policy that never learns (always random) stays at chance (1/4).
    baseline = _avg(train(PickEnv(answer_seed=7), QAgent(n_actions=4, epsilon=1.0, seed=1), episodes=3000))

    assert 0.15 < baseline < 0.35, f"random baseline should hover at chance, got {baseline:.2f}"
    assert late > 0.75, f"the learner should master the task, got {late:.2f}"
    assert late > baseline + 0.4, "learning should decisively beat not-learning"


def test_learning_is_deterministic():
    # Same seeds -> same learning curve, so this proof is a stable, reproducible test.
    def run():
        return train(PickEnv(answer_seed=7), QAgent(n_actions=4, seed=0), episodes=500)

    assert run() == run()
