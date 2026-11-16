"""CPU integration checks for opt-in EviMem strategy behavior."""

import json

import numpy as np
import pytest

from vlm_orchestrator import cli
from vlm_orchestrator.evimem.runtime import MemoryRuntime, EviMemConfig
from vlm_orchestrator.strategies import evimem_subgoal
from vlm_orchestrator.strategies.base import SessionState, StrategyContext
from vlm_orchestrator.strategies.tasksys_subgoal import TaskSysSubgoalStrategy
from vlm_orchestrator.strategies.subgoal import SubgoalConfig, SubgoalStrategy
from vlm_orchestrator.strategies.subgoal_base import SubgoalBaseStrategy
from vlm_orchestrator.vlm import PassthroughVLM


class ControlledMemory:
    """A verifier double exposing gate and session boundaries to these tests."""

    def __init__(self, config):
        self.config = config
        self.allowed = False
        self.broken_goals = []
        self.marked = []
        self.checks = []
        self.resets = []
        self.invalidated = []
        self.plans = []

    def reset(self, episode_id):
        self.resets.append(episode_id)
        self.marked.clear()

    def set_plan(self, payload, subgoals):
        self.plans.append((payload, list(subgoals)))
        self.broken_goals = []

    def observe(self, obs, step):
        if obs.get("malformed"):
            raise ValueError("invalid online observation")

    def invalidate_current(self, step, reason):
        self.invalidated.append((step, reason))
        self.allowed = False

    def context(self):
        return "EviMem evidence for current session"

    def check_completion(self, index, step):
        self.checks.append((index, step))
        return {
            "allowed": self.allowed,
            "reason": "controlled verifier",
            "status": "TRUE" if self.allowed else "UNKNOWN",
            "evidence_ids": ["e1"] if self.allowed else [],
        }

    def mark_complete(self, index):
        self.marked.append(index)

    def snapshot(self):
        return {"marked": list(self.marked), "broken_goals": self.broken_goals}


def observation(**kwargs):
    return {
        "prompt": "put both blocks in the bin",
        "observation/image": np.zeros((16, 16, 3), dtype=np.uint8),
        **kwargs,
    }


PLAN = {
    "subgoals": ["place red block in bin", "place blue block in bin"],
    "ordered": True,
    "memory": {
        "objects": [{"id": "red", "description": "red block"}],
        "goals": [
            {"subgoal_index": 0, "predicates": [{"name": "holding", "args": ["red"]}]}
        ],
    },
}
NEXT = {"status": "complete", "action": "next", "reason": "looks complete"}


def strategy_fixture(monkeypatch, *, gate="enforce", retries=3, responses=None):
    calls = []
    replies = iter(responses or [PLAN, NEXT, NEXT, NEXT])

    def call(self, prompt, content, max_tokens=None):
        calls.append((prompt, content, max_tokens))
        return json.dumps(next(replies))

    monkeypatch.setattr(evimem_subgoal, "MemoryRuntime", ControlledMemory)
    monkeypatch.setattr(SubgoalBaseStrategy, "_vlm_call", call)
    strategy = evimem_subgoal.EviMemSubgoalStrategy(
        StrategyContext(vlm=PassthroughVLM(), image_key="observation/image"),
        SubgoalConfig(check_interval=1),
        memory_config=EviMemConfig(gate=gate, max_reobservations=retries),
        failure_monitor="vlm",
        recovery_mode="replan",
    )
    state = SessionState()
    strategy.process(observation(), state)
    state.infer_count = 1
    return strategy, state, calls


def step(strategy, state, obs=None):
    state.episode_step += 1
    result = strategy.process(obs or observation(), state)
    state.infer_count += 1
    return result


@pytest.mark.parametrize("use_tasksys,merge_final", [(False, False), (True, False), (True, True)])
def test_model_advances_with_image_and_retained_coordinates_after_track_loss(monkeypatch, use_tasksys, merge_final):
    objects = [
        {"id": "red", "description": "red block"},
        {"id": "blue", "description": "blue block"},
        {"id": "bin", "description": "black bin"},
    ]
    effects = [{"name": "in", "args": [object_id, "bin"]} for object_id in ("red", "blue")]
    if use_tasksys:
        plan = {
            "objects": objects,
            "nodes": [
                {"id": effect["args"][0], "skill": "place_in", "object_ids": effect["args"],
                 "preconditions": [], "expected_effects": [effect]}
                for effect in effects
            ],
            "final_conditions": effects,
        }
    else:
        plan = {**PLAN, "memory": {**PLAN["memory"], "objects": objects}}
    replies = iter([plan, {"status": "in_progress", "action": "continue", "reason": "cannot tell"},
                    NEXT, NEXT, NEXT])
    calls = []

    def call(self, prompt, content, max_tokens=None):
        calls.append(content)
        reply = next(replies)
        if use_tasksys and reply.get("action") == "next":
            monitor = json.loads(content[0]["text"])
            confirmed = monitor["final_conditions"] if merge_final and monitor["final_conditions"] else monitor["expected_effects"]
            reply = {**reply, "confirmed_conditions": confirmed,
                     "evidence_refs": [monitor["scene_ref"]]}
        return json.dumps(reply)

    monkeypatch.setattr(SubgoalBaseStrategy, "_vlm_call", call)
    strategy_type = TaskSysSubgoalStrategy if use_tasksys else evimem_subgoal.EviMemSubgoalStrategy
    strategy = strategy_type(
        StrategyContext(vlm=PassthroughVLM(), image_key="observation/image"),
        SubgoalConfig(check_interval=1),
        memory_config=EviMemConfig(gate="enforce", verification="vlm"),
        failure_monitor="vlm", recovery_mode="replan",
    )
    state = SessionState()
    strategy.process(observation(**{"observation/evimem_objects": {
        "coordinate_frame": "world", "units": "m", "source": "rgbd",
        "objects": [{"id": "red", "position": [0.2, 0, 0.1], "frame_id": "initial", "confidence": 0.9}],
    }}), state)
    state.infer_count = 1

    step(strategy, state)
    assert state.current_subgoal_idx == 0
    out, _ = step(strategy, state)
    assert state.current_subgoal_idx == 1
    assert "blue block" in out["prompt"]
    assert any(part["type"] == "image_url" for part in calls[-1])
    context = json.loads(calls[-1][0 if use_tasksys else -1]["text"])
    if use_tasksys:
        context = context["evidence"]
    assert context["current_step"] == 2
    red = context["objects"][0]
    assert red["visible"] is False
    assert red["last_trusted_position"] == [0.2, 0, 0.1]
    assert red["last_trusted_step"] == 0
    assert red["last_trusted_age_steps"] == 2
    assert red["trusted_samples"][0]["confidence"] == 0.9
    assert not state.evimem.contracts
    assert not state.evimem.store.assertions
    assert state.evimem.snapshot()["verification"] == []
    if use_tasksys:
        step(strategy, state)
        assert state.tasksys.final_verified == merge_final
        step(strategy, state)
        assert state.tasksys.final_verified and state.evimem_hold
        assert len(calls) == (4 if merge_final else 5)


def test_tasksys_rechecks_on_release_and_repairs_a_changed_dependency(monkeypatch):
    support = {"name": "on", "args": ["red", "base"]}
    placement = {"name": "on", "args": ["blue", "red"]}
    first = {"id": "first", "skill": "place_on", "object_ids": ["red", "base"],
             "preconditions": [], "expected_effects": [support]}
    second = {"id": "second", "skill": "place_on", "object_ids": ["blue", "red", "base"],
              "preconditions": [support], "expected_effects": [placement]}
    plan = {"objects": [{"id": object_id, "description": f"{object_id} block"}
                        for object_id in ("red", "base", "blue")],
            "nodes": [first, second], "final_conditions": [support, placement]}
    monitors = []

    def call(self, prompt, content, max_tokens=None):
        text = content[0]["text"]
        if text.startswith("Task:"):
            return json.dumps(plan)
        request = json.loads(text)
        if "basis" in request:
            assert request["required"] == [support]
            assert request["affected_nodes"] == ["__goal__", "second"]
            return json.dumps({"basis": request["basis"], "repair_nodes": [{**first, "id": "restore"}]})
        monitors.append(request)
        step = request["step"]
        if step == 2:
            assert request["evidence"]["last_open_step"] == 1
            ref = request["evidence"]["objects"][0]["trusted_samples"][0]["ref"]
            assert request["evidence"]["objects"][0]["trusted_samples"][0]["step"] == 0
            return json.dumps({**NEXT, "confirmed_conditions": [support], "evidence_refs": [ref]})
        if step in (8, 9):
            action = "next" if step == 8 else "continue"
            return json.dumps({"action": action, "reason": "current support observed",
                               "confirmed_conditions": [support], "evidence_refs": [request["scene_ref"]]})
        assert step == 11 and request["conditions_to_review"] == [support]
        return json.dumps({"action": "replan", "reason": "red block moved away",
                           "evidence_refs": [request["scene_ref"]]})

    def sample(step, gripper, red_x=0.0):
        return observation(**{
            "observation/gripper_position": [gripper],
            "observation/evimem_objects": {
                "coordinate_frame": "world", "units": "m", "source": "rgbd",
                "objects": [{"id": "red", "position": [red_x, 0, 0.1], "frame_id": f"frame:{step}"}],
            },
        })

    monkeypatch.setattr(SubgoalBaseStrategy, "_vlm_call", call)
    strategy = TaskSysSubgoalStrategy(
        StrategyContext(vlm=PassthroughVLM(), image_key="observation/image"),
        SubgoalConfig(check_interval=4), memory_config=EviMemConfig(gate="enforce", verification="vlm"),
        failure_monitor="vlm", recovery_mode="replan",
    )
    state = SessionState()
    strategy.process(sample(0, 0.9), state)
    state.infer_count = 1
    for current in range(1, 9):
        step(strategy, state, sample(current, 0.1))
        if current < 8:
            assert state.current_subgoal_idx == 0
    assert state.current_subgoal_idx == 1 and state.evimem_hold
    step(strategy, state, sample(9, 0.1))
    assert state.tasksys.active and not state.evimem_hold
    step(strategy, state, sample(10, 0.1, red_x=0.1))
    assert state.evimem_hold
    step(strategy, state, sample(11, 0.1, red_x=0.1))
    assert [row["step"] for row in monitors] == [2, 8, 9, 11]
    assert [node.id for node in state.tasksys.graph.nodes] == ["first", "restore", "second"]
    assert state.tasksys.graph.version == 2


def test_plan_and_monitor_receive_memory_without_additional_calls(monkeypatch):
    strategy, state, calls = strategy_fixture(monkeypatch)
    step(strategy, state)
    assert len(calls) == 2
    assert '"memory" object' in calls[0][0]
    assert calls[0][2] >= 2048
    assert calls[1][1][-1]["text"] == "EviMem evidence for current session"
    assert state.evimem.plans == [(PLAN["memory"], PLAN["subgoals"])]


def test_unknown_gate_blocks_advancement(monkeypatch):
    strategy, state, _ = strategy_fixture(monkeypatch)
    step(strategy, state)
    assert state.current_subgoal_idx == 0
    assert state.evimem.marked == []
    assert state.evimem_blocked_counts[0] == 1


def test_verified_advance_marks_completion_once(monkeypatch):
    strategy, state, _ = strategy_fixture(monkeypatch)
    state.evimem.allowed = True
    step(strategy, state)
    assert state.current_subgoal_idx == 1
    assert state.evimem.marked == [0]
    assert state.evimem.checks == [(0, 1)]


def test_shadow_logs_rejection_and_preserves_original_next(monkeypatch):
    strategy, state, _ = strategy_fixture(monkeypatch, gate="shadow")
    step(strategy, state)
    assert state.current_subgoal_idx == 1
    checks = [
        entry for entry in state.log_entries if entry.get("event") == "completion_check"
    ]
    assert checks[-1]["decision"]["allowed"] is False
    assert checks[-1]["memory_gate"] == "shadow"


def test_final_next_is_also_gated(monkeypatch):
    strategy, state, calls = strategy_fixture(monkeypatch)
    state.current_subgoal_idx = 1
    state.rewritten_instruction = state.subgoals[1]
    step(strategy, state)
    assert len(calls) == 2
    assert strategy._recycle_count == 0
    assert state.evimem_blocked_counts == {1: 1}


def test_reobservation_budget_requests_one_replan(monkeypatch):
    strategy, state, _ = strategy_fixture(monkeypatch, retries=0)
    replans = []
    monkeypatch.setattr(
        strategy, "_recycle", lambda *args: replans.append(True) or False
    )
    step(strategy, state)
    step(strategy, state)
    assert replans == [True]
    assert state.current_subgoal_idx == 0
    assert state.evimem_hold
    assert state.flush_actions


def test_broken_completion_retries_are_bounded(monkeypatch):
    strategy, state, _ = strategy_fixture(monkeypatch, retries=1)
    state.evimem.broken_goals = [0]
    replans = []
    monkeypatch.setattr(
        strategy, "_recycle", lambda *args: replans.append(True) or False
    )
    step(strategy, state)
    step(strategy, state)
    step(strategy, state)
    assert replans == [True, True]
    assert state.evimem_hold


def test_new_plan_releases_hold(monkeypatch):
    strategy, state, _ = strategy_fixture(
        monkeypatch, retries=0, responses=[PLAN, NEXT, PLAN]
    )
    step(strategy, state)
    assert len(state.evimem.plans) == 2
    assert state.evimem_hold is False
    assert state.flush_actions


def test_exhausted_repair_holds_until_current_evidence_verifies(monkeypatch):
    strategy, state, calls = strategy_fixture(monkeypatch, retries=0)
    monkeypatch.setattr(strategy, "_recycle", lambda *args: False)
    step(strategy, state)
    assert state.evimem_hold
    step(strategy, state)
    assert state.evimem_hold
    assert len(calls) == 2
    state.evimem.allowed = True
    step(strategy, state)
    assert state.evimem_hold is False
    assert state.current_subgoal_idx == 1
    assert len(calls) == 3


def test_restored_fact_releases_hold_after_failed_repair(monkeypatch):
    strategy, state, _ = strategy_fixture(monkeypatch)
    state.evimem.broken_goals = [0]
    monkeypatch.setattr(strategy, "_recycle", lambda *args: False)
    step(strategy, state)
    assert state.evimem_hold
    state.evimem.broken_goals = []
    step(strategy, state)
    assert state.evimem_hold is False


def test_broken_prior_fact_blocks_even_verified_final_goal(monkeypatch):
    strategy, state, _ = strategy_fixture(monkeypatch)
    state.evimem.allowed = True
    state.evimem.broken_goals = [0]
    state.current_subgoal_idx = 1
    state.rewritten_instruction = state.subgoals[1]
    monkeypatch.setattr(strategy, "_recycle", lambda *args: False)
    step(strategy, state)
    assert state.current_subgoal_idx == 1
    assert state.evimem.marked == []


def test_successful_repair_does_not_reschedule_same_pending_obligation(monkeypatch):
    strategy, state, calls = strategy_fixture(monkeypatch, responses=[PLAN, PLAN, NEXT])
    state.evimem.broken_goals = ["old-goal"]
    original = state.evimem.set_plan

    def set_plan(payload, subgoals):
        original(payload, subgoals)
        state.evimem.broken_goals = ["old-goal"]

    monkeypatch.setattr(state.evimem, "set_plan", set_plan)
    step(strategy, state)
    assert len(calls) == 2
    assert state.evimem_hold is False
    assert state.evimem_repair_attempted == {"old-goal"}
    step(strategy, state)
    assert len(calls) == 3


def test_independent_sessions_own_memory(monkeypatch):
    strategy, first, _ = strategy_fixture(monkeypatch, responses=[PLAN, PLAN, NEXT])
    second = SessionState()
    strategy.process(observation(), second)
    step(strategy, first)
    assert first.evimem is not second.evimem
    assert first.evimem.resets == second.evimem.resets == [1]
    assert first.evimem.checks == [(0, 1)]
    assert second.evimem.checks == []


def test_invalid_observation_invalidates_prior_verification(monkeypatch):
    strategy, state, _ = strategy_fixture(monkeypatch)
    state.evimem.allowed = True
    step(strategy, state, observation(malformed=True))
    assert state.evimem.invalidated == [(1, "invalid online observation")]
    assert state.current_subgoal_idx == 0


def test_stale_observation_fails_closed_with_real_runtime(monkeypatch):
    strategy, state, _ = strategy_fixture(monkeypatch)
    runtime = MemoryRuntime(EviMemConfig(gate="enforce"))
    runtime.reset(state.episode_id)
    runtime.set_plan(PLAN["memory"], PLAN["subgoals"])
    latest = observation(
        **{
            "observation/evimem_objects": {
                "coordinate_frame": "world",
                "units": "m",
                "source": "visual",
                "objects": [{"id": "red", "position": [0, 0, 0], "holding": True}],
            }
        }
    )
    runtime.observe(latest, 4)
    runtime.observe(latest, 5)
    assert runtime.check_completion(0, 5)["allowed"]
    state.evimem = runtime
    state.episode_step = 3
    step(strategy, state)
    assert state.current_subgoal_idx == 0
    assert runtime.check_completion(0, 4)["allowed"] is False
    assert "out-of-order" in runtime.observation_error


def test_same_prompt_new_episode_resets_memory(monkeypatch):
    strategy, state, _ = strategy_fixture(monkeypatch, responses=[PLAN, PLAN])
    state.episode_id = 2
    step(strategy, state)
    assert state.evimem.resets == [1, 2]
    assert state.evimem_blocked_counts == {}


def test_rewritten_prompt_does_not_reset_episode(monkeypatch):
    strategy, state, _ = strategy_fixture(monkeypatch)
    step(strategy, state, observation(prompt=state.rewritten_instruction))
    assert state.evimem.resets == [1]


def test_replan_registers_new_memory_contract(monkeypatch):
    revised = {"subgoals": ["pick up red"], "memory": {"objects": [], "goals": []}}
    strategy, state, calls = strategy_fixture(monkeypatch, responses=[PLAN, revised])
    token = strategy._active_state.set(state)
    try:
        assert strategy._recycle(
            observation(), state, observation()["observation/image"]
        )
    finally:
        strategy._active_state.reset(token)
    assert state.evimem.plans[-1] == (revised["memory"], revised["subgoals"])
    assert '"memory" object' in calls[1][0]


@pytest.mark.parametrize(
    "arguments",
    [
        ["--mode", "next_goal", "--memory-mode", "episodic"],
        ["--memory-mode", "episodic", "--failure-monitor", "gt"],
        ["--memory-mode", "current", "--memory-gate", "enforce"],
        [
            "--memory-mode",
            "episodic",
            "--memory-gate",
            "enforce",
            "--failure-monitor",
            "vlm",
            "--subgoal-check-mode",
            "timer",
        ],
        ["--memory-max-reobservations", "-1"],
        ["--memory-min-confidence", "nan"],
        ["--memory-min-confidence", "1.1"],
        [
            "--memory-mode",
            "episodic",
            "--memory-gate",
            "enforce",
            "--failure-monitor",
            "vlm",
            "--env",
            "libero",
            "--recovery-mode",
            "replan_tools",
        ],
    ],
)
def test_cli_rejects_unsupported_memory_combinations(monkeypatch, arguments):
    monkeypatch.setattr("sys.argv", ["vlm-orchestrator", *arguments])
    with pytest.raises(SystemExit) as raised:
        cli.main()
    assert raised.value.code == 2


@pytest.mark.parametrize("mode", ["off", "current", "episodic"])
def test_cli_selects_independent_strategy(monkeypatch, tmp_path, mode):
    parsed = []

    class StopConstruction(Exception):
        pass

    original = cli._build_strategy

    def capture(args):
        parsed.append(args)
        raise StopConstruction

    monkeypatch.setattr(
        "sys.argv",
        ["vlm-orchestrator", "--memory-mode", mode, "--log-dir", str(tmp_path)],
    )
    monkeypatch.setattr(cli, "_build_strategy", capture)
    with pytest.raises(StopConstruction):
        cli.main()
    monkeypatch.setattr(cli, "_build_vlm", lambda args: PassthroughVLM())
    strategy = original(parsed[0])
    expected = SubgoalStrategy if mode == "off" else evimem_subgoal.EviMemSubgoalStrategy
    assert type(strategy) is expected


def test_real_memory_suspension_release_and_regression_through_strategy(monkeypatch):
    plan = {
        "subgoals": ["place red on blue", "continue next action"],
        "ordered": True,
        "memory": {
            "objects": [
                {"id": "red", "description": "red block"},
                {"id": "blue", "description": "blue block"},
            ],
            "goals": [
                {
                    "subgoal_index": 0,
                    "predicates": [{"name": "on", "args": ["red", "blue"]}],
                }
            ],
        },
    }
    replies = iter(
        [
            plan,
            NEXT,
            NEXT,
            {"done": True, "subgoals": []},
            {"status": "in_progress", "action": "continue"},
        ]
    )
    monkeypatch.setattr(
        SubgoalBaseStrategy,
        "_vlm_call",
        lambda *args, **kwargs: json.dumps(next(replies)),
    )
    strategy = evimem_subgoal.EviMemSubgoalStrategy(
        StrategyContext(vlm=PassthroughVLM(), image_key="observation/image"),
        SubgoalConfig(check_interval=1),
        memory_config=EviMemConfig(gate="enforce"),
        failure_monitor="vlm",
        recovery_mode="replan",
    )

    def scene(frame, *, holding=False, moved=False):
        x = 0.5 if moved else 0.0
        return observation(
            **{
                "observation/evimem_objects": {
                    "coordinate_frame": "world",
                    "units": "m",
                    "source": "rgbd",
                    "objects": [
                        {
                            "id": "red",
                            "position": [x, 0.0, 0.15],
                            "bounds": [
                                [x - 0.025, -0.025, 0.1],
                                [x + 0.025, 0.025, 0.2],
                            ],
                            "frame_id": frame,
                            "holding": holding,
                        },
                        {
                            "id": "blue",
                            "position": [0.0, 0.0, 0.05],
                            "bounds": [[-0.05, -0.05, 0.0], [0.05, 0.05, 0.1]],
                            "frame_id": frame,
                        },
                    ],
                }
            }
        )

    state = SessionState()
    strategy.process(scene("f0", holding=True), state)
    state.infer_count = 1
    step(strategy, state, scene("f1"))
    assert state.current_subgoal_idx == 0
    step(strategy, state, scene("f2"))
    assert state.current_subgoal_idx == 1
    assert state.evimem.broken_goals == ()
    step(strategy, state, scene("f3", moved=True))
    assert state.current_subgoal_idx == 1
    assert state.evimem.broken_goals
    assert state.evimem_hold and state.flush_actions
