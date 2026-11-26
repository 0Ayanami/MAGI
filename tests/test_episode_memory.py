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

    def __init__(self):
        self.calls = 0

    def detect(self, image, phrase, score_threshold):
        self.calls += 1
        color = {"mug": (220, 220, 220), "bowl": (220, 220, 220), "robot gripper": (0, 255, 0),
                 "plate": (0, 0, 255)}[phrase]
        y, x = np.where(np.all(image == color, axis=-1))
        return [{"box": [x.min(), y.min(), x.max(), y.max()]}] if x.size else []


def frame(x, hand, opened, missing=False, height=0):
    rgb = np.zeros((100, 100, 3), np.uint8)
    rgb[50:57, 42:72, 2] = 255
    if not missing:
        rgb[40:52, x-2:x+3] = [220, 220, 220]
    rgb[41:49, hand-1:hand+2] = [0, 255, 0]
    q = .04 if opened else .019
    return {"observation/image_raw": rgb, "observation/state": np.array([hand/100, 0, height, 0, 0, 0, q, -q]),
            "prompt": GOAL}


def replay(mode="enforce", visual_input="history", timeout=9999, monitor="next",
           goals=None, semantic="off", interval=20):
    strategy = SubgoalStrategy(StrategyContext(vlm=None), SubgoalConfig(
        semantic_memory=semantic, visual_memory=mode, visual_input=visual_input,
        check_interval=interval, subgoal_timeout=timeout), failure_monitor="vlm", recovery_mode="replan")
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
    assert [f.step for f in state.placement_window.observations] == [18]
    assert tick(19, 53, 60, True, True) == PLACE
    assert not state.placement_window.frames
    tick(20, *RELEASE[-1])
    assert len(calls) == 2
    assert tick(40, *RELEASE[-1]) == PLACE
    assert len(calls) == 3
    assert not state.flush_actions


def test_mirrored_camera_and_mug_identity():
    from vlm_orchestrator.strategies.episode_memory import PlacementWindow

    class SceneDetector:
        def detect(self, image, phrase, score_threshold):
            boxes = {"mug": [[15, 35, 25, 52], [70, 35, 80, 52]],
                     "plate": [[5, 50, 35, 60], [60, 50, 90, 60]]}
            return [{"box": box} for box in boxes[phrase]]

    rgb = np.zeros((100, 100, 3), np.uint8)
    rgb[35:53, 15:26] = (222, 202, 55)
    rgb[35:53, 70:81] = 200
    obs = {"observation/state": np.array([-.26, -.06, 1, 0, 0, 0, .02, -.02]),
           "observation/camera_K": np.array([[100, 0, 50], [0, 100, 50], [0, 0, 1]]),
           "observation/camera_extrinsic": np.eye(4)}
    window = PlacementWindow()
    window.observe(obs, rgb, 5, "Put the white mug on the left plate", SceneDetector())
    sample, = window.frames
    assert sample.on_plate
    assert np.allclose(sample.obj, [.75, .435])
    assert np.allclose(sample.hand, [.75, .44])


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


def test_repetition_keeps_distinct_placement_occurrences():
    from vlm_orchestrator.strategies.episode_memory import EpisodeSummary, placement

    skill = "Put the bowl back on the plate"
    summary = EpisodeSummary(skill + " three times")
    assert placement(skill) == ("bowl", "")
    assert placement("Place the grey bowl back on the same plate") == ("bowl", "")
    assert placement(summary.goal) is None
    assert placement("Place the two bowls on the plate") is None
    assert placement("Place the bowl on the plate and open the drawer") is None
    summary.record(skill, 20, "vlm", "model_complete", "vlm@20", occurrence=(0, 0))
    summary.record(skill, 25, "vlm", "model_complete", "vlm@25", occurrence=(0, 0))
    summary.record(skill, 45, "vision", "observed_complete", [35, 40, 45], occurrence=(1, 20))
    cycles = list(summary.placement_cycles.values())
    assert [(e["step"], e["status"]) for e in cycles] == [(20, "model_complete"), (45, "observed_complete")]
    assert '"placement_cycles"' in summary.prompt()


@pytest.mark.parametrize("lifted", [True, False])
def test_bowl_confirmation_runs_detector_only_at_the_check(lifted):
    skill = "Place the bowl back on the plate"
    strategy, state, requests, tick = replay(goals=[skill, NEXT], timeout=20)
    assert tick(5, 48, 49, False, False, .04 if lifted else 0) == skill
    assert tick(10, *RELEASE[1]) == skill
    assert tick(15, *RELEASE[2]) == skill
    assert strategy._placement_detector.calls == 0
    assert tick(20, *RELEASE[-1]) == NEXT
    assert strategy._placement_detector.calls == (11 if lifted else 0)
    assert len(requests) == (1 if lifted else 2) and state.flush_actions


def test_return_to_original_plate_uses_initial_support_evidence():
    from vlm_orchestrator.strategies.episode_memory import PlacementWindow

    class SceneDetector:
        def detect(self, image, phrase, score_threshold):
            boxes = {"bowl": [[47, 40, 53, 52]], "robot gripper": [[54, 40, 56, 49]],
                     "plate": [[40, 50, 60, 60], [70, 50, 89, 61]]}
            return [{"box": box} for box in boxes[phrase]]

    window = PlacementWindow()
    skill = "Place the bowl back on the plate"
    obs = frame(50, 55, False)
    rgb = obs["observation/image_raw"]
    window._measure_frame(obs, rgb, 5, skill, SceneDetector())
    assert not window.frames
    window._measure_frame(obs, rgb, 5, skill, SceneDetector(), initial_image=rgb)
    sample, = window.frames
    assert sample.on_plate and np.allclose(sample.plate, [.5, .55])


def test_release_can_advance_between_cloud_checks():
    _, state, calls, tick = replay(interval=40)
    for i, values in enumerate(RELEASE, 1):
        sent = tick(i * 5, *values)
    assert sent == NEXT and len(calls) == 1 and state.flush_actions
    check, = [e for e in state.log_entries if e["type"] == "placement_check"]
    assert check["verdict"] == "stable" and not check["scheduled"]
    assert check["frames"] == [5, 10, 15, 20]


def test_failed_release_is_not_repeated_and_keeps_cloud_schedule():
    strategy, state, calls, tick = replay(interval=25, monitor="continue")
    tick(5, *RELEASE[0])
    tick(10, *RELEASE[1])
    tick(15, *RELEASE[2])
    assert tick(20, 53, 66, True, True) == PLACE
    attempts = strategy._placement_detector.calls
    assert attempts == 2 and len(calls) == 1
    assert tick(25, 53, 69, True) == PLACE
    assert strategy._placement_detector.calls == attempts and len(calls) == 2
    checks = [e for e in state.log_entries if e["type"] == "placement_check"]
    assert [e["evidence_reason"] for e in checks] == ["object_missing", "release_already_checked"]
    assert [e["scheduled"] for e in checks] == [False, True]
    for i, values in enumerate(RELEASE, 6):
        sent = tick(i * 5, *values)
    assert sent == NEXT and len(calls) == 2


def test_carrying_ignores_unrelated_box_and_does_not_detect_a_plate():
    class CarryDetector(ColorDetector):
        def detect(self, image, phrase, score_threshold):
            assert phrase != "plate"
            hits = super().detect(image, phrase, score_threshold)
            if phrase == "bowl":
                hits.append({"box": [2, 0, 12, 8]})
            return hits

    skill = "Place the bowl back on the plate"
    strategy, state, calls, tick = replay(goals=[skill, NEXT])
    strategy._placement_detector = CarryDetector()
    for i, values in enumerate(CARRY, 1):
        sent = tick(i * 5, *values)
    assert sent == skill and len(calls) == 1
    check, = [e for e in state.log_entries if e["type"] == "placement_check"]
    assert check["verdict"] == "carrying"
    assert strategy._placement_detector.calls == 6


def test_same_region_cannot_establish_object_and_support():
    from vlm_orchestrator.strategies.episode_memory import PlacementWindow

    class ConfusedDetector:
        def detect(self, image, phrase, score_threshold):
            boxes = {"bowl": [101, 125, 134, 156], "plate": [101, 126, 133, 156],
                     "robot gripper": [142, 125, 156, 153]}
            return [{"box": boxes[phrase]}]

    window = PlacementWindow()
    obs = frame(50, 55, True)
    window.observe(obs, np.zeros((224, 224, 3), np.uint8), 5,
                   "Place the bowl on the plate", ConfusedDetector())
    assert window.verdict() == "unknown" and window.evidence_reason == "object_support_overlap"


def test_failed_release_during_timeout_cooldown_cannot_advance():
    _, state, calls, tick = replay(timeout=15, monitor="continue")
    for i, values in enumerate(RELEASE[:-1], 1):
        assert tick(i * 5, *values) == PLACE
    assert len(calls) == 2
    assert tick(20, 53, 66, True, True) == PLACE
    assert len(calls) == 2 and not state.flush_actions
    checks = [e for e in state.log_entries if e["type"] == "placement_check"]
    assert [e["scheduled"] for e in checks] == [True, False]
    assert all(e["verdict"] == "unknown" for e in checks)


def test_previous_grasp_cannot_confirm_new_bowl_release():
    skill = "Place the bowl back on the plate"
    strategy, state, calls, tick = replay(goals=[skill, NEXT], interval=40, monitor="continue")
    tick(5, 48, 49, False, False, .04)
    tick(10, 48, 55, True)
    tick(15, 52, 53, False)
    for step, values in zip((20, 25, 30, 35, 40), RELEASE[1:] + [RELEASE[-1]] * 2):
        sent = tick(step, *values)
    assert sent == skill and len(calls) == 2
    assert strategy._placement_detector.calls == 0
    check, = [e for e in state.log_entries if e["type"] == "placement_check"]
    assert check["evidence_reason"] == "no_lift_or_withdrawal"


@pytest.mark.parametrize("withdrawn", [True, False])
def test_release_retries_once_after_fresh_withdrawal(withdrawn):
    strategy, state, calls, tick = replay(interval=40)
    for i, values in enumerate(RELEASE[:-1], 1):
        tick(i * 5, *values)
    assert tick(20, 53, 62, True) == PLACE
    first_attempts = strategy._placement_detector.calls
    assert tick(22, 53, 62, True) == PLACE
    assert strategy._placement_detector.calls == first_attempts
    assert tick(25, 53, 66 if withdrawn else 64, True) == (NEXT if withdrawn else PLACE)
    attempts = strategy._placement_detector.calls
    if not withdrawn:
        assert tick(30, 53, 68, True) == PLACE
        assert strategy._placement_detector.calls == attempts
    assert len(calls) == 1
    checks = [e for e in state.log_entries if e["type"] == "placement_check"]
    assert [e["evidence_reason"] for e in checks] == [
        "withdrawal_pending", "stable" if withdrawn else "withdrawal_pending",
    ]


def test_final_placement_waits_for_scheduled_check():
    strategy, state, calls, tick = replay(goals=[PLACE], interval=40, monitor="continue")
    for i, values in enumerate(RELEASE, 1):
        assert tick(i * 5, *values) == PLACE
    assert strategy._placement_detector.calls == 0
    assert len(calls) == 1 and not state.flush_actions
