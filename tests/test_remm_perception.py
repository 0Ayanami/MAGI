"""CPU regressions for memory grounding, release and temporal verification."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from vlm_orchestrator.remm_memory.perception import (
    GeometryVerifier,
    ObjectObservation,
    RGBDGrounder,
    observation_from_mask,
    observations_from_payload,
)


def predicate(name, *args):
    return SimpleNamespace(name=name, args=args)


def placement(frame_id="frame-1", *, holding=False):
    return {
        "red": ObjectObservation(
            "red",
            (0.0, 0.0, 0.15),
            ((-0.025, -0.025, 0.1), (0.025, 0.025, 0.2)),
            frame_id=frame_id,
            holding=holding,
        ),
        "blue": ObjectObservation(
            "blue",
            (0.0, 0.0, 0.05),
            ((-0.05, -0.05, 0.0), (0.05, 0.05, 0.1)),
            frame_id=frame_id,
        ),
    }


def advance_frame(objects, frame):
    return {key: replace(obj, frame_id=frame) for key, obj in objects.items()}


def test_released_support_needs_two_distinct_stable_observations():
    verifier = GeometryVerifier()
    relation = predicate("on", "red", "blue")
    objects = placement()
    assert verifier.evaluate(relation, objects, 0).truth == "UNKNOWN"
    assert verifier.evaluate(relation, objects, 0).truth == "UNKNOWN"
    assert verifier.evaluate(relation, objects, 1).truth == "UNKNOWN"
    second = advance_frame(objects, "frame-2")
    result = verifier.evaluate(relation, second, 2)
    assert result.truth == "TRUE"
    assert result.frame_ids == ("frame-1", "frame-2")
    assert verifier.evaluate(relation, second, 2) == result


def test_suspended_object_is_rejected_and_unknown_release_does_not_complete():
    relation = predicate("on", "red", "blue")
    assert (
        GeometryVerifier().evaluate(relation, placement(holding=True), 0).truth
        == "FALSE"
    )
    result = GeometryVerifier().evaluate(relation, placement(holding=None), 0)
    assert result.truth == "UNKNOWN"
    assert result.reason == "release_not_observed"


def test_visual_holding_needs_agreement_while_proprioception_is_direct():
    relation = predicate("holding", "red")
    visual = ObjectObservation("red", frame_id="frame-1", holding=True, source="visual")
    verifier = GeometryVerifier()
    assert verifier.evaluate(relation, {"red": visual}, 0).truth == "UNKNOWN"
    second = replace(visual, frame_id="frame-2", position=(1, 2, 3))
    assert verifier.evaluate(relation, {"red": second}, 1).truth == "TRUE"
    sensor = replace(visual, source="proprioception")
    assert GeometryVerifier().evaluate(relation, {"red": sensor}, 0).truth == "TRUE"


def test_occlusion_clears_temporal_votes_without_false_motion_claim():
    verifier = GeometryVerifier()
    relation = predicate("on", "red", "blue")
    objects = placement()
    verifier.evaluate(relation, objects, 0)
    hidden = dict(objects, red=replace(objects["red"], visible=False))
    assert verifier.evaluate(relation, hidden, 1).truth == "UNKNOWN"
    assert (
        verifier.evaluate(relation, advance_frame(objects, "frame-2"), 2).truth
        == "UNKNOWN"
    )
    assert (
        verifier.evaluate(relation, advance_frame(objects, "frame-3"), 3).truth
        == "TRUE"
    )


def test_instability_and_long_gap_require_fresh_confirmation():
    verifier = GeometryVerifier(max_gap_steps=10)
    relation = predicate("on", "red", "blue")
    objects = placement()
    verifier.evaluate(relation, objects, 0)
    moved = advance_frame(objects, "frame-2")
    moved["red"] = replace(moved["red"], position=(0.02, 0.0, 0.15))
    assert verifier.evaluate(relation, moved, 1).truth == "UNKNOWN"
    assert (
        verifier.evaluate(relation, advance_frame(moved, "frame-3"), 20).truth
        == "UNKNOWN"
    )
    assert (
        verifier.evaluate(relation, advance_frame(moved, "frame-4"), 21).truth == "TRUE"
    )
    assert verifier.evaluate(relation, objects, 0).reason == "out_of_order_observation"


def test_single_frame_ablation_has_no_temporal_votes():
    verifier = GeometryVerifier()
    relation = predicate("on", "red", "blue")
    assert verifier.evaluate(relation, placement(), 0, history=False).truth == "TRUE"
    assert verifier.evaluate(relation, placement(), 0, history=True).truth == "UNKNOWN"


def test_temporal_confidence_retains_the_weaker_observation():
    verifier = GeometryVerifier()
    relation = predicate("on", "red", "blue")
    objects = placement()
    first = dict(objects, red=replace(objects["red"], confidence=0.8))
    verifier.evaluate(relation, first, 0)
    result = verifier.evaluate(relation, advance_frame(objects, "frame-2"), 1)
    assert result.truth == "TRUE"
    assert result.confidence == 0.8


def test_every_object_needs_a_fresh_frame_and_motion_cannot_accumulate():
    verifier = GeometryVerifier(stable_observations=3)
    relation = predicate("on", "red", "blue")
    objects = placement()
    verifier.evaluate(relation, objects, 0)
    partial = dict(objects, red=replace(objects["red"], frame_id="frame-2"))
    assert (
        verifier.evaluate(relation, partial, 1).reason == "repeated_observation_frame"
    )
    second = advance_frame(objects, "frame-2")
    second["red"] = replace(second["red"], position=(0.009, 0.0, 0.15))
    assert verifier.evaluate(relation, second, 2).truth == "UNKNOWN"
    third = advance_frame(second, "frame-3")
    third["red"] = replace(third["red"], position=(0.018, 0.0, 0.15))
    assert verifier.evaluate(relation, third, 3).truth == "UNKNOWN"


def test_containment_requires_explicit_interior_and_full_bounds_inside():
    verifier = GeometryVerifier()
    objects = placement()
    relation = predicate("in", "red", "blue")
    assert (
        verifier.evaluate(relation, objects, 0).reason == "container_interior_unknown"
    )
    objects["blue"] = replace(
        objects["blue"], interior_bounds=((-0.1, -0.1, 0.0), (0.1, 0.1, 0.3))
    )
    assert verifier.evaluate(relation, objects, 1).truth == "UNKNOWN"
    assert (
        verifier.evaluate(relation, advance_frame(objects, "frame-2"), 2).truth
        == "TRUE"
    )
    objects["red"] = replace(objects["red"], bounds=((0.3, 0.3, 0.1), (0.4, 0.4, 0.2)))
    assert verifier.evaluate(relation, objects, 3).truth == "FALSE"


def test_low_confidence_missing_object_and_unsupported_predicates_are_unknown():
    objects = placement()
    objects["red"] = replace(objects["red"], confidence=0.7)
    verifier = GeometryVerifier()
    assert (
        verifier.evaluate(predicate("on", "red", "blue"), objects, 0).truth == "UNKNOWN"
    )
    assert (
        verifier.evaluate(predicate("holding", "missing"), objects, 0).truth
        == "UNKNOWN"
    )
    assert (
        verifier.evaluate(predicate("at", "red", "blue"), objects, 0).truth == "UNKNOWN"
    )


def test_payload_units_provenance_and_duplicate_ids_are_validated():
    payload = {
        "coordinate_frame": "world",
        "units": "m",
        "source": "rgbd",
        "objects": [{"id": "red", "position": [1, 2, 3], "holding": False}],
    }
    result = observations_from_payload(payload, frame_id="f0")
    assert result["red"].position == (1, 2, 3)
    assert result["red"].frame_id == "f0"
    for bad_fields in (
        {"units": "mm"},
        {"coordinate_frame": "camera"},
        {"source": "gt_sim"},
    ):
        with pytest.raises(ValueError):
            observations_from_payload(dict(payload, **bad_fields), frame_id="f0")
    with pytest.raises(ValueError):
        observations_from_payload(
            dict(payload, objects=payload["objects"] * 2), frame_id="f0"
        )


def test_mask_projection_respects_metric_calibration_and_camera_to_world():
    K = np.array([[2.0, 0, 1.5], [0, 2.0, 1.5], [0, 0, 1]])
    transform = np.eye(4)
    transform[:3, 3] = (1, 2, 3)
    depth = np.full((4, 4), 2.0)
    mask = np.ones((4, 4), dtype=bool)
    observation = observation_from_mask(
        "red", depth, mask, K, transform, frame_id="f0", confidence=0.9
    )
    assert observation.position == (1, 2, 5)
    assert np.allclose(observation.bounds, ((-0.5, 0.5, 5), (2.5, 3.5, 5)))
    missing = observation_from_mask(
        "red", np.zeros_like(depth), mask, K, transform, frame_id="f0", confidence=0.9
    )
    assert missing.visible is False
    assert missing.position is None
    with pytest.raises(ValueError):
        observation_from_mask(
            "red", depth, mask[:2], K, transform, frame_id="f0", confidence=0.9
        )
    with pytest.raises(ValueError):
        observation_from_mask(
            "red", depth, mask, -K, transform, frame_id="f0", confidence=0.9
        )
    bad_transform = transform.copy()
    bad_transform[0, 0] = 2
    with pytest.raises(ValueError):
        observation_from_mask(
            "red", depth, mask, K, bad_transform, frame_id="f0", confidence=0.9
        )


class FakePerceptionClient:
    def __init__(self, *, duplicates=False, score=0.9):
        self.duplicates = duplicates
        self.score = score
        self.images = []

    def detect(self, image, description):
        self.images.append(image.copy())
        detection = {"score": 0.95, "box": [0, 0, 4, 4]}
        return [detection] * (2 if self.duplicates else 1)

    def segment(self, image, point_x, point_y, *, box):
        return np.ones(image.shape[:2], dtype=bool), self.score


def camera_observation():
    image = np.arange(48, dtype=np.uint8).reshape(4, 4, 3)
    return {
        "observation/image_raw": image,
        "observation/depth_agentview": np.ones((4, 4)),
        "observation/camera_K": np.array([[2, 0, 1.5], [0, 2, 1.5], [0, 0, 1]]),
        "observation/camera_extrinsic": np.eye(4),
        "observation/gripper_position": np.array([0.0]),
        # A conflicting GT object must have no effect on the result.
        "gt_state": {"red": {"position": (100, 100, 100)}},
        "gt_seg/body_ids": np.ones((4, 4)),
    }


def test_live_grounder_uses_perception_only_and_preserves_policy_image():
    client = FakePerceptionClient()
    obs = camera_observation()
    result = RGBDGrounder(client).observe(
        obs, [{"id": "red", "description": "red block"}], 0
    )
    assert result["red"].visible
    assert result["red"].position == (0, 0, 1)
    assert result["red"].holding is False
    assert np.array_equal(client.images[0], obs["observation/image_raw"])
    obs["observation/gripper_position"] = np.array([1.0])
    assert (
        RGBDGrounder(client)
        .observe(obs, [{"id": "red", "description": "red block"}], 1)["red"]
        .holding
        is None
    )


@pytest.mark.parametrize(
    ("env_mode", "image_for_grasp", "expected_x"),
    [
        ("libero", False, 1.0),
        ("libero", True, -1.0),
        ("robolab", False, -1.0),
        ("vlabench", False, -1.0),
        ("robocasa", False, -1.0),
    ],
)
def test_grounder_preserves_detection_frame_and_projects_registered_mask_depth(
    env_mode, image_for_grasp, expected_x
):
    class LeftHalfClient(FakePerceptionClient):
        def segment(self, image, point_x, point_y, *, box):
            self.images.append(image.copy())
            assert (point_x, point_y) == (0.25, 0.5)
            assert box == (0, 0, 4, 4)
            mask = np.zeros(image.shape[:2], dtype=bool)
            mask[:, :4] = True
            return mask, self.score

    image = np.arange(96, dtype=np.uint8).reshape(4, 8, 3)
    depth = np.ones((4, 8))
    depth[:, 4:] = 2.0
    obs = camera_observation()
    obs["observation/camera_K"] = np.array(
        [[2, 0, 3.5], [0, 2, 1.5], [0, 0, 1]]
    )
    if image_for_grasp:
        obs["observation/image_for_grasp"] = image
        obs["observation/depth_for_grasp"] = depth
    elif env_mode == "libero":
        obs["observation/image_raw"] = image
        obs["observation/depth_agentview"] = depth
    else:
        obs["observation/exterior_image_1_left_raw"] = image
        obs["observation/depth_external"] = depth
    client = LeftHalfClient()
    observed = RGBDGrounder(client, env_mode=env_mode).observe(
        obs, [{"id": "left", "description": "the block on the left"}], 0
    )["left"]

    # Both services see the same image as the description author. The 16 masked
    # left pixels have depth 1; LIBERO calibration places them on the right.
    assert len(client.images) == 2
    assert all(np.array_equal(seen, image) for seen in client.images)
    assert observed.visible
    assert observed.position == (expected_x, 0.0, 1.0)
    assert observed.bounds == (
        (expected_x - 0.75, -0.75, 1.0),
        (expected_x + 0.75, 0.75, 1.0),
    )


def test_live_grounder_refuses_ambiguous_identity_and_missing_calibration():
    obs = camera_observation()
    specs = [{"id": "a", "description": "red block"}]
    assert (
        not RGBDGrounder(FakePerceptionClient(duplicates=True))
        .observe(obs, specs, 0)["a"]
        .visible
    )
    assert (
        not RGBDGrounder(FakePerceptionClient(score=float("nan")))
        .observe(obs, specs, 0)["a"]
        .visible
    )
    del obs["observation/camera_extrinsic"]
    assert not RGBDGrounder(FakePerceptionClient()).observe(obs, specs, 0)["a"].visible


def test_two_labels_bound_to_the_same_mask_remain_unknown():
    specs = [
        {"id": "a", "description": "red block"},
        {"id": "b", "description": "green block"},
    ]
    result = RGBDGrounder(FakePerceptionClient()).observe(
        camera_observation(), specs, 0
    )
    assert all(not observation.visible for observation in result.values())
    specs[1]["aliases"] = ["red block"]
    client = FakePerceptionClient()
    result = RGBDGrounder(client).observe(camera_observation(), specs, 0)
    assert all(not observation.visible for observation in result.values())
    assert client.images == []
