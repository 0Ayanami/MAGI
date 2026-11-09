"""CPU regressions spanning evidence ingestion, revision and offline replay."""

import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from vlm_orchestrator.remm_memory.runtime import MemoryRuntime, RemmConfig
from vlm_orchestrator.remm_memory.store import MemoryStore
from vlm_orchestrator.remm_memory.types import Predicate, Truth

PLAN = {
    "objects": [
        {"id": "red", "description": "red block"},
        {"id": "blue", "description": "blue block"},
        {"id": "green", "description": "green block"},
    ],
    "goals": [
        {"subgoal_index": 0, "predicates": [{"name": "on", "args": ["red", "blue"]}]},
        {"subgoal_index": 1, "predicates": [{"name": "holding", "args": ["green"]}]},
    ],
}


def runtime(mode="episodic"):
    memory = MemoryRuntime(RemmConfig(mode=mode))
    memory.reset(1)
    memory.set_plan(deepcopy(PLAN), ["stack red", "pick green"])
    return memory


def observation(step, *, held=False, moved=False):
    x = 0.5 if moved else 0.0
    return {
        "observation/remm_objects": {
            "coordinate_frame": "world",
            "units": "m",
            "source": "rgbd",
            "objects": [
                {
                    "id": "red",
                    "position": [x, 0, 0.15],
                    "bounds": [[x - 0.025, -0.025, 0.1], [x + 0.025, 0.025, 0.2]],
                    "holding": held,
                    "frame_id": f"frame{step}",
                },
                {
                    "id": "blue",
                    "position": [0, 0, 0.05],
                    "bounds": [[-0.05, -0.05, 0], [0.05, 0.05, 0.1]],
                    "frame_id": f"frame{step}",
                },
                {
                    "id": "green",
                    "position": [1, 0, 0.1],
                    "holding": True,
                    "frame_id": f"frame{step}",
                },
            ],
        },
    }


def test_physical_completion_is_retracted_without_erasing_evidence():
    memory = runtime()
    memory.observe(observation(0, held=True), 0)
    assert memory.check_completion(0, 0)["status"] == "FALSE"
    memory.observe(observation(5), 5)
    assert not memory.check_completion(0, 5)["allowed"]
    memory.observe(observation(10), 10)
    assert memory.check_completion(0, 10)["allowed"]
    completed = memory.mark_complete(0)
    assert completed.step == 10
    assert completed.parents
    original_events = memory.store.events
    memory.observe(observation(15, moved=True), 15)
    assert memory.broken_goals == ("plan1:subgoal0",)
    assert memory.store.events[: len(original_events)] == original_events
    assert not memory.check_completion(1, 15)["allowed"]


def test_current_frame_ablation_does_not_use_temporal_confirmation():
    current, episodic = runtime("current"), runtime()
    obs = observation(0)
    current.observe(obs, 0)
    episodic.observe(obs, 0)
    assert current.check_completion(0, 0)["allowed"]
    assert not episodic.check_completion(0, 0)["allowed"]
    current.observe({}, 5)
    assert not current.check_completion(0, 5)["allowed"]
    assert all(
        obj.last_trusted_position is None for obj in current.store.objects.values()
    )


def test_episodic_prompts_expose_referenced_evidence_and_revisions():
    memory = runtime()
    memory.observe(observation(0), 0)
    memory.observe(observation(5), 5)
    memory.mark_complete(0)
    memory.observe(observation(10, moved=True), 10)
    context = json.loads(memory.context())["memory"]
    assert context["evidence_history"]
    assert any(row["status"] == "invalidated" for row in context["assertion_revisions"])
    assert context["history_window"]["included_events"] <= 32
    current = runtime("current")
    current.observe(observation(0), 0)
    assert "evidence_history" not in json.loads(current.context())["memory"]


def test_occlusion_preserves_history_with_unknown_current_completion():
    memory = runtime()
    memory.observe(observation(0), 0)
    memory.observe(observation(5), 5)
    memory.mark_complete(0)
    memory.observe({}, 10)
    assert not memory.store.objects["red"].visible
    assert memory.store.objects["red"].last_trusted_position == (0, 0, 0.15)
    assert memory.store.current_truth(Predicate("on", ("red", "blue"))) == Truth.TRUE
    assert memory.check_completion(0, 10)["status"] == "UNKNOWN"
    assert not memory.broken_goals


def test_gt_cannot_change_memory_or_supply_missing_perception():
    first, second = runtime(), runtime()
    obs = observation(0)
    first.observe(obs, 0)
    second.observe(dict(obs, gt_state={"objects": "fake"}, ground_truth_done=True), 0)
    assert first.snapshot() == second.snapshot()
    second.observe({"gt_state": {"satisfied": True}, "gt_seg/red": [[1]]}, 5)
    assert second.check_completion(0, 5)["status"] == "UNKNOWN"


def test_replan_retains_broken_obligations_until_physical_repair():
    memory = runtime()
    memory.observe(observation(0), 0)
    memory.observe(observation(5), 5)
    memory.mark_complete(0)
    memory.observe(observation(10, moved=True), 10)
    memory.set_plan(
        {
            "objects": [PLAN["objects"][2]],
            "goals": [
                {
                    "subgoal_index": 0,
                    "predicates": [{"name": "holding", "args": ["green"]}],
                }
            ],
        },
        ["pick green"],
    )
    memory.observe(observation(15, moved=True), 15)
    assert memory.broken_goals
    assert not memory.check_completion(0, 15)["allowed"]
    memory.observe(observation(20), 20)
    memory.observe(observation(25), 25)
    assert not memory.broken_goals
    assert memory.check_completion(0, 25)["allowed"]


def test_identity_rebinding_and_malformed_plan_disable_coverage():
    memory = runtime()
    changed = deepcopy(PLAN)
    changed["objects"][0]["description"] = "orange block"
    memory.set_plan(changed, ["a", "b"])
    assert memory.plan_error
    assert not memory.contracts
    memory.set_plan(None, ["a"])
    assert not memory.check_completion(0, 0)["allowed"]


def test_rejected_batch_cannot_partially_mutate_supported_state(monkeypatch):
    memory = runtime()
    memory.observe(observation(0), 0)
    before = memory.store.snapshot()
    original = MemoryStore.assert_fact

    def reject_relation(self, predicate, *args, **kwargs):
        if predicate.name == "on":
            raise ValueError("injected late validation failure")
        return original(self, predicate, *args, **kwargs)

    monkeypatch.setattr(MemoryStore, "assert_fact", reject_relation)
    with pytest.raises(ValueError, match="late validation"):
        memory.observe(observation(5, moved=True), 5)
    assert memory.store.snapshot() == before
    assert not memory.check_completion(0, 5)["allowed"]


def test_stale_observation_and_duplicate_evidence_are_rejected():
    memory = runtime()
    memory.observe(observation(5), 5)
    memory.observe(observation(5), 5)
    assert len(memory.store.events) == 3
    with pytest.raises(ValueError, match="Conflicting"):
        memory.observe(observation(5, moved=True), 5)
    assert not memory.check_completion(0, 5)["allowed"]
    with pytest.raises(ValueError, match="out-of-order"):
        memory.observe(observation(0), 0)


def test_reused_old_negative_frame_cannot_revoke_fresh_completion():
    memory = runtime()
    old_negative = observation(0, moved=True)
    memory.observe(old_negative, 0)
    memory.observe(observation(5), 5)
    memory.observe(observation(10), 10)
    memory.mark_complete(0)
    memory.observe(old_negative, 15)
    assert memory.check_completion(0, 15)["status"] == "UNKNOWN"
    assert not memory.broken_goals
    assert memory.store.current_truth(Predicate("on", ("red", "blue"))) == Truth.TRUE


def test_cpu_replay_command_exports_evidence_and_revocation(tmp_path):
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps({"memory": PLAN, "subgoals": ["stack red", "pick green"]})
    )
    frames = tmp_path / "frames.jsonl"
    rows = [
        {"step": 0, "observation": observation(0), "check_subgoal": 0},
        {
            "step": 5,
            "observation": observation(5),
            "check_subgoal": 0,
            "commit_completion": True,
        },
        {"step": 10, "observation": observation(10, moved=True), "check_subgoal": 1},
    ]
    frames.write_text("\n".join(json.dumps(row) for row in rows))
    output = tmp_path / "result.jsonl"
    subprocess.run(
        [
            sys.executable,
            "-B",
            "-m",
            "vlm_orchestrator.remm_memory.replay",
            "--plan",
            str(plan),
            "--input",
            str(frames),
            "--output",
            str(output),
        ],
        check=True,
        cwd=Path(__file__).resolve().parents[1],
    )
    results = [json.loads(line) for line in output.read_text().splitlines()]
    assert results[1]["decision"]["allowed"]
    assert not results[2]["decision"]["allowed"]
    assert results[-1]["final_memory"]["broken_goals"] == ["plan1:subgoal0"]
    assert len(results[-1]["final_memory"]["events"]) >= 10
