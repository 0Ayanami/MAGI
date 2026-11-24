import json

import numpy as np
import pytest

from vlm_orchestrator.strategies.base import SessionState, StrategyContext
from vlm_orchestrator.strategies.subgoal import SubgoalConfig, SubgoalStrategy


PLACE = "Place the white mug on the plate"
NEXT = "Open the drawer"
GOAL = PLACE + " and then open the drawer"


class ColorDetector:
    """Test perception backend measures labelled pixels in constructed RGB frames."""

    def detect(self, image, phrase, score_threshold):
        color = {"white mug": (255, 0, 0), "robot gripper": (0, 255, 0),
                 "plate": (0, 0, 255)}[phrase]
        y, x = np.where(np.all(image == color, axis=-1))
        return [{"box": [x.min(), y.min(), x.max(), y.max()]}] if x.size else []


def frame(x, hand, opened, missing=False):
    rgb = np.zeros((100, 100, 3), np.uint8)
    rgb[50:57, 42:72, 2] = 255
    if not missing:
        rgb[40:52, x-2:x+3] = [255, 0, 0]
    rgb[41:49, hand-1:hand+2] = [0, 255, 0]
    q = .04 if opened else .019
    return {"observation/image_raw": rgb, "observation/state": np.array([hand/100, 0, 0, 0, 0, 0, q, -q]),
            "prompt": GOAL}


def replay(mode="enforce", visual_input="history", timeout=9999, monitor="next",
           goals=None, semantic="off"):
    strategy = SubgoalStrategy(StrategyContext(vlm=None), SubgoalConfig(
        semantic_memory=semantic, visual_memory=mode, visual_input=visual_input,
        check_interval=20, subgoal_timeout=timeout), failure_monitor="vlm", recovery_mode="replan")
    strategy._placement_detector = ColorDetector()
    calls = []

    def cloud(system, content, max_tokens=None):
        message = " ".join(row["text"] for row in content if row["type"] == "text")
        calls.append(message)
        if len(calls) == 1 or "Scene instruction" in message:
            return json.dumps({"subgoals": goals or [PLACE, NEXT], "ordered": True})
        if max_tokens:
            return json.dumps({"subgoals": [PLACE, NEXT] if "needs_check" in message else [NEXT]})
        return json.dumps({"status": "complete" if monitor == "next" else "in_progress",
                           "action": monitor})

    strategy._vlm_call = cloud
    state = SessionState()

    def tick(step, *args):
        state.episode_step = step
        obs, _ = strategy.process(frame(*args), state)
        state.infer_count += 1
        return strategy.ctx.get_prompt(obs)

    tick(0, 44, 46, False)
    return strategy, state, calls, tick


CARRY = [(44, 57, False), (47, 60, False), (50, 63, False), (53, 66, False)]
RELEASE = [(52, 53, False), (53, 56, True), (53, 60, True), (53, 66, True)]


@pytest.mark.parametrize("sequence,mode,visual_input,expected,calls", [
    (CARRY, "enforce", "history", PLACE, 1),
    (RELEASE, "enforce", "history", NEXT, 1),
    (CARRY, "enforce", "current", NEXT, 2),
    (CARRY, "shadow", "history", NEXT, 2),
    (CARRY, "off", "history", NEXT, 2),
])
def test_same_last_image_different_dispatch(sequence, mode, visual_input, expected, calls):
    _, state, requests, tick = replay(mode, visual_input, timeout=20)
    for i, values in enumerate(sequence, 1):
        sent = tick(i*5, *values)
    assert sent == expected
    assert len(requests) == calls
    if mode == "enforce":
        assert state.flush_actions == (expected == NEXT)


def test_expiry_occlusion_timeout_and_final_continue():
    _, state, calls, tick = replay(timeout=15, monitor="continue", goals=[PLACE])
    tick(5, *RELEASE[0])
    assert tick(18, *RELEASE[1]) == PLACE
    assert [f.step for f in state.placement_window.frames] == [18]
    assert tick(19, 53, 60, True, True) == PLACE
    assert not state.placement_window.frames
    tick(20, *RELEASE[-1])
    assert len(calls) == 2
    assert tick(40, *RELEASE[-1]) == PLACE
    assert len(calls) == 3
    assert not state.flush_actions


def test_persistent_unknown_replans_and_clears_window():
    _, state, calls, tick = replay(timeout=10, monitor="continue")
    for step in (10, 30):
        tick(step, 50, 52, False, True)
    assert len(calls) == 4
    assert state.rewritten_instruction == NEXT
    assert state.flush_actions and not state.placement_window.frames


@pytest.mark.parametrize("regressed", [False, True])
def test_replan_same_image_uses_episode_evidence(regressed):
    from vlm_orchestrator.failure_handlers.base import HandlerResult

    strategy, state, calls, tick = replay(semantic="episode")
    for i, values in enumerate(RELEASE, 1):
        tick(i*5, *values)
    assert "Episode evidence" not in calls[0]
    assert state.rewritten_instruction == NEXT
    reason = "The white mug fell off the plate." if regressed else "The drawer is stuck."
    strategy._execute_handler_result(frame(*RELEASE[-1]), state, HandlerResult("failure", "replan", reason))
    assert '"status": "observed_complete"' in calls[-1]
    assert ('"status": "needs_check"' in calls[-1]) == regressed
    assert tick(21, *RELEASE[-1]) == (PLACE if regressed else NEXT)
    state.episode_id += 1
    tick(0, *RELEASE[0])
    assert not state.semantic_summary.events
    assert not state.placement_window.frames
