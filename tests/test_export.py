import json

from crucible import rollout, to_records, write_jsonl
from crucible.envs import GuessEnv
from examples.agents import BinarySearchAgent


def a_trajectory(seed: int = 3):
    return rollout(GuessEnv(), BinarySearchAgent(), seed=seed, max_steps=12)


def test_to_records_shape():
    traj = a_trajectory()
    records = to_records(traj)
    assert len(records) == traj.steps
    first = records[0]
    assert set(first) == {"env_id", "seed", "step", "prompt", "completion", "reward"}
    assert first["env_id"] == "guess"
    assert first["reward"] == traj.transitions[0].reward
    # Non-string observation/action are rendered to canonical JSON text by default.
    assert isinstance(first["prompt"], str)
    assert isinstance(first["completion"], str)


def test_render_hooks_are_applied():
    traj = a_trajectory()
    records = to_records(
        traj,
        render_observation=lambda o: "OBS",
        render_action=lambda a: f"guess={a}",
    )
    assert records[0]["prompt"] == "OBS"
    assert records[0]["completion"].startswith("guess=")


def test_default_render_keeps_strings_and_json_encodes_others():
    from crucible.export import _default_render

    assert _default_render("hello") == "hello"
    assert _default_render({"b": 1, "a": 2}) == '{"a": 2, "b": 1}'  # sorted keys


def test_write_jsonl_roundtrips(tmp_path):
    trajs = [a_trajectory(1), a_trajectory(2)]
    path = tmp_path / "data.jsonl"
    count = write_jsonl(trajs, path)

    lines = path.read_text(encoding="utf-8").strip().split("\n")
    assert count == len(lines)
    assert count == trajs[0].steps + trajs[1].steps
    for line in lines:
        record = json.loads(line)  # every line is valid JSON
        assert "reward" in record


def test_write_jsonl_empty(tmp_path):
    path = tmp_path / "empty.jsonl"
    assert write_jsonl([], path) == 0
    assert path.read_text(encoding="utf-8") == ""
