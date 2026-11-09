"""CPU integration checks for opt-in REMM strategy behavior."""

import json

import numpy as np
import pytest

from vlm_orchestrator import cli
from vlm_orchestrator.remm_memory.runtime import MemoryRuntime, RemmConfig
from vlm_orchestrator.strategies import remm_subgoal
from vlm_orchestrator.strategies.base import SessionState, StrategyContext
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
        return "REMM evidence for current session"

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

    monkeypatch.setattr(remm_subgoal, "MemoryRuntime", ControlledMemory)
    monkeypatch.setattr(SubgoalBaseStrategy, "_vlm_call", call)
    strategy = remm_subgoal.RemmSubgoalStrategy(
        StrategyContext(vlm=PassthroughVLM(), image_key="observation/image"),
        SubgoalConfig(check_interval=1),
        memory_config=RemmConfig(gate=gate, max_reobservations=retries),
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


def test_plan_and_monitor_receive_memory_without_additional_calls(monkeypatch):
    strategy, state, calls = strategy_fixture(monkeypatch)
    step(strategy, state)
    assert len(calls) == 2
    assert '"memory" object' in calls[0][0]
    assert calls[0][2] >= 2048
    assert calls[1][1][-1]["text"] == "REMM evidence for current session"
    assert state.remm_memory.plans == [(PLAN["memory"], PLAN["subgoals"])]


def test_unknown_gate_blocks_advancement(monkeypatch):
    strategy, state, _ = strategy_fixture(monkeypatch)
    step(strategy, state)
    assert state.current_subgoal_idx == 0
    assert state.remm_memory.marked == []
    assert state.remm_blocked_counts[0] == 1


def test_verified_advance_marks_completion_once(monkeypatch):
    strategy, state, _ = strategy_fixture(monkeypatch)
    state.remm_memory.allowed = True
    step(strategy, state)
    assert state.current_subgoal_idx == 1
    assert state.remm_memory.marked == [0]
    assert state.remm_memory.checks == [(0, 1)]


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
    assert state.remm_blocked_counts == {1: 1}


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
    assert state.remm_hold
    assert state.flush_actions


def test_broken_completion_retries_are_bounded(monkeypatch):
    strategy, state, _ = strategy_fixture(monkeypatch, retries=1)
    state.remm_memory.broken_goals = [0]
    replans = []
    monkeypatch.setattr(
        strategy, "_recycle", lambda *args: replans.append(True) or False
    )
    step(strategy, state)
    step(strategy, state)
    step(strategy, state)
    assert replans == [True, True]
    assert state.remm_hold


def test_new_plan_releases_hold(monkeypatch):
    strategy, state, _ = strategy_fixture(
        monkeypatch, retries=0, responses=[PLAN, NEXT, PLAN]
    )
    step(strategy, state)
    assert len(state.remm_memory.plans) == 2
    assert state.remm_hold is False
    assert state.flush_actions


def test_exhausted_repair_holds_until_current_evidence_verifies(monkeypatch):
    strategy, state, calls = strategy_fixture(monkeypatch, retries=0)
    monkeypatch.setattr(strategy, "_recycle", lambda *args: False)
    step(strategy, state)
    assert state.remm_hold
    step(strategy, state)
    assert state.remm_hold
    assert len(calls) == 2
    state.remm_memory.allowed = True
    step(strategy, state)
    assert state.remm_hold is False
    assert state.current_subgoal_idx == 1
    assert len(calls) == 3


def test_restored_fact_releases_hold_after_failed_repair(monkeypatch):
    strategy, state, _ = strategy_fixture(monkeypatch)
    state.remm_memory.broken_goals = [0]
    monkeypatch.setattr(strategy, "_recycle", lambda *args: False)
    step(strategy, state)
    assert state.remm_hold
    state.remm_memory.broken_goals = []
    step(strategy, state)
    assert state.remm_hold is False


def test_broken_prior_fact_blocks_even_verified_final_goal(monkeypatch):
    strategy, state, _ = strategy_fixture(monkeypatch)
    state.remm_memory.allowed = True
    state.remm_memory.broken_goals = [0]
    state.current_subgoal_idx = 1
    state.rewritten_instruction = state.subgoals[1]
    monkeypatch.setattr(strategy, "_recycle", lambda *args: False)
    step(strategy, state)
    assert state.current_subgoal_idx == 1
    assert state.remm_memory.marked == []


def test_successful_repair_does_not_reschedule_same_pending_obligation(monkeypatch):
    strategy, state, calls = strategy_fixture(monkeypatch, responses=[PLAN, PLAN, NEXT])
    state.remm_memory.broken_goals = ["old-goal"]
    original = state.remm_memory.set_plan

    def set_plan(payload, subgoals):
        original(payload, subgoals)
        state.remm_memory.broken_goals = ["old-goal"]

    monkeypatch.setattr(state.remm_memory, "set_plan", set_plan)
    step(strategy, state)
    assert len(calls) == 2
    assert state.remm_hold is False
    assert state.remm_repair_attempted == {"old-goal"}
    step(strategy, state)
    assert len(calls) == 3


def test_independent_sessions_own_memory(monkeypatch):
    strategy, first, _ = strategy_fixture(monkeypatch, responses=[PLAN, PLAN, NEXT])
    second = SessionState()
    strategy.process(observation(), second)
    step(strategy, first)
    assert first.remm_memory is not second.remm_memory
    assert first.remm_memory.resets == second.remm_memory.resets == [1]
    assert first.remm_memory.checks == [(0, 1)]
    assert second.remm_memory.checks == []


def test_invalid_observation_invalidates_prior_verification(monkeypatch):
    strategy, state, _ = strategy_fixture(monkeypatch)
    state.remm_memory.allowed = True
    step(strategy, state, observation(malformed=True))
    assert state.remm_memory.invalidated == [(1, "invalid online observation")]
    assert state.current_subgoal_idx == 0


def test_stale_observation_fails_closed_with_real_runtime(monkeypatch):
    strategy, state, _ = strategy_fixture(monkeypatch)
    runtime = MemoryRuntime(RemmConfig(gate="enforce"))
    runtime.reset(state.episode_id)
    runtime.set_plan(PLAN["memory"], PLAN["subgoals"])
    latest = observation(
        **{
            "observation/remm_objects": {
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
    state.remm_memory = runtime
    state.episode_step = 3
    step(strategy, state)
    assert state.current_subgoal_idx == 0
    assert runtime.check_completion(0, 4)["allowed"] is False
    assert "out-of-order" in runtime.observation_error


def test_same_prompt_new_episode_resets_memory(monkeypatch):
    strategy, state, _ = strategy_fixture(monkeypatch, responses=[PLAN, PLAN])
    state.episode_id = 2
    step(strategy, state)
    assert state.remm_memory.resets == [1, 2]
    assert state.remm_blocked_counts == {}


def test_rewritten_prompt_does_not_reset_episode(monkeypatch):
    strategy, state, _ = strategy_fixture(monkeypatch)
    step(strategy, state, observation(prompt=state.rewritten_instruction))
    assert state.remm_memory.resets == [1]


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
    assert state.remm_memory.plans[-1] == (revised["memory"], revised["subgoals"])
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
    expected = SubgoalStrategy if mode == "off" else remm_subgoal.RemmSubgoalStrategy
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
    strategy = remm_subgoal.RemmSubgoalStrategy(
        StrategyContext(vlm=PassthroughVLM(), image_key="observation/image"),
        SubgoalConfig(check_interval=1),
        memory_config=RemmConfig(gate="enforce"),
        failure_monitor="vlm",
        recovery_mode="replan",
    )

    def scene(frame, *, holding=False, moved=False):
        x = 0.5 if moved else 0.0
        return observation(
            **{
                "observation/remm_objects": {
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
    assert state.remm_memory.broken_goals == ()
    step(strategy, state, scene("f3", moved=True))
    assert state.current_subgoal_idx == 1
    assert state.remm_memory.broken_goals
    assert state.remm_hold and state.flush_actions
