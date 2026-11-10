"""CPU grounding and conservative relation checks for REMM memory.

Positions and bounds use metres in a common world frame with +Z up.  Online
inputs come from visual detections and proprioception.  Simulation GT and GT
segmentation fields are deliberately outside this adapter's input contract.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from vlm_orchestrator.grasp.camera import CameraIntrinsics, depth_to_pointcloud

Vector3 = tuple[float, float, float]
Bounds = tuple[Vector3, Vector3]
_SOURCES = frozenset({"rgbd", "visual", "proprioception"})


def _vector(value) -> Vector3:
    array = np.asarray(value, dtype=float)
    if array.shape != (3,) or not np.isfinite(array).all():
        raise ValueError("Expected three finite coordinates")
    return tuple(float(item) for item in array)


def _bounds(value) -> Bounds:
    array = np.asarray(value, dtype=float)
    if array.shape != (2, 3) or not np.isfinite(array).all():
        raise ValueError("Expected finite bounds [minimum_xyz, maximum_xyz]")
    if np.any(array[0] > array[1]):
        raise ValueError("Bounds minimum exceeds maximum")
    return _vector(array[0]), _vector(array[1])


@dataclass(frozen=True)
class ObjectObservation:
    id: str
    position: Vector3 | None = None
    bounds: Bounds | None = None
    visible: bool = True
    confidence: float = 1.0
    frame_id: str = ""
    holding: bool | None = None
    source: str = "rgbd"
    interior_bounds: Bounds | None = None
    support_z: float | None = None

    def __post_init__(self):
        if not isinstance(self.id, str) or not self.id.strip():
            raise ValueError("Object ID must be a nonempty string")
        if self.source not in _SOURCES:
            raise ValueError(
                "Memory accepts only online RGB-D, visual or proprioceptive sources"
            )
        if not isinstance(self.visible, bool):
            raise TypeError("visible must be boolean")
        if self.holding is not None and not isinstance(self.holding, bool):
            raise ValueError("holding must be boolean or null")
        if not math.isfinite(self.confidence) or not 0 <= self.confidence <= 1:
            raise ValueError("Confidence must be finite and between zero and one")
        if not isinstance(self.frame_id, str):
            raise TypeError("frame_id must be a string")
        if self.position is not None:
            object.__setattr__(self, "position", _vector(self.position))
        if self.bounds is not None:
            object.__setattr__(self, "bounds", _bounds(self.bounds))
        if self.interior_bounds is not None:
            object.__setattr__(self, "interior_bounds", _bounds(self.interior_bounds))
        if self.support_z is not None and not math.isfinite(self.support_z):
            raise ValueError("Support height must be finite")


def observations_from_payload(
    payload, *, frame_id: str
) -> dict[str, ObjectObservation]:
    """Decode an explicit non-GT ``observation/remm_objects`` payload.

    A payload may contain ``objects`` plus shared ``coordinate_frame=world``
    and ``units=m`` fields, or be a list with those fields in each entry.
    Sources and coordinate conventions are mandatory in external payloads.
    Extra observation fields, including GT fields, are never read here.
    """
    defaults = payload if isinstance(payload, dict) else {}
    entries = defaults.get("objects") if isinstance(payload, dict) else payload
    if not isinstance(entries, list):
        raise TypeError("Expected an object list or an objects payload")
    result = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise TypeError("Each observation must be an object")
        if entry.get("coordinate_frame", defaults.get("coordinate_frame")) != "world":
            raise ValueError("Object observations must use the world coordinate frame")
        if entry.get("units", defaults.get("units")) != "m":
            raise ValueError("Object observations must use metres")
        if entry.get("source", defaults.get("source")) not in _SOURCES:
            raise ValueError("An explicit online observation source is required")
        observation = ObjectObservation(
            id=entry["id"],
            position=entry.get("position"),
            bounds=entry.get("bounds"),
            visible=entry.get("visible", True),
            confidence=entry.get("confidence", 1.0),
            frame_id=entry.get("frame_id", frame_id),
            holding=entry.get("holding"),
            source=entry.get("source", defaults.get("source")),
            interior_bounds=entry.get("interior_bounds"),
            support_z=entry.get("support_z"),
        )
        if observation.id in result:
            raise ValueError(f"Duplicate object ID: {observation.id}")
        result[observation.id] = observation
    return result


def observation_from_mask(
    object_id: str,
    depth,
    mask,
    camera_K,
    camera_to_world,
    *,
    frame_id: str,
    confidence: float,
    holding: bool | None = None,
    min_points: int = 16,
    max_depth: float = 10.0,
) -> ObjectObservation:
    """Project a registered visual mask and metric depth into world geometry.

    The RGB/mask, depth and calibration must share the same OpenCV pixel frame.
    No image resizing, camera pose guessing or depth-unit inference is applied.
    Bounds describe the observed surface. Container interiors require separate
    calibrated geometry and are never inferred from the exterior mask.
    """
    depth = np.asarray(depth, dtype=float)
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    mask = np.asarray(mask)
    if depth.ndim != 2 or mask.shape != depth.shape:
        raise ValueError("Mask and depth must have the same HxW shape")
    if not np.isfinite(mask).all() or not np.isin(mask, (0, 1)).all():
        raise ValueError("Segmentation masks must be binary")
    K = np.asarray(camera_K, dtype=float).reshape(3, 3)
    transform = np.asarray(camera_to_world, dtype=float).reshape(4, 4)
    if not np.isfinite(K).all() or K[0, 0] <= 0 or K[1, 1] <= 0:
        raise ValueError("Camera intrinsics must be finite with positive focal lengths")
    if not np.allclose(K[2], (0, 0, 1)) or K[0, 1] != 0 or K[1, 0] != 0:
        raise ValueError("Expected unskewed pinhole camera intrinsics")
    rotation = transform[:3, :3]
    if (
        not np.isfinite(transform).all()
        or not np.allclose(transform[3], (0, 0, 0, 1))
        or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)
        or not np.isclose(np.linalg.det(rotation), 1, atol=1e-5)
    ):
        raise ValueError("Expected a rigid OpenCV camera-to-world transform")
    if min_points < 1 or not math.isfinite(max_depth) or max_depth <= 0:
        raise ValueError("Invalid point-cloud limits")
    height, width = depth.shape
    camera = CameraIntrinsics(K[0, 0], K[1, 1], K[0, 2], K[1, 2], width, height)
    points = depth_to_pointcloud(
        np.where(mask, depth, np.nan), camera, max_depth=max_depth
    )
    if len(points) < min_points:
        return ObjectObservation(
            object_id, visible=False, confidence=0.0, frame_id=frame_id
        )
    world = points @ rotation.T + transform[:3, 3]
    return ObjectObservation(
        object_id,
        position=_vector(np.median(world, axis=0)),
        bounds=(_vector(world.min(axis=0)), _vector(world.max(axis=0))),
        confidence=confidence,
        frame_id=frame_id,
        holding=holding,
    )


@dataclass(frozen=True)
class Verification:
    truth: str
    reason: str
    object_ids: tuple[str, ...]
    frame_ids: tuple[str, ...] = ()
    confidence: float = 0.0


@dataclass(frozen=True)
class _Sample:
    step: int
    frames: tuple[str, ...]
    positions: tuple[Vector3, ...]
    confidence: float


class GeometryVerifier:
    """Check holding, containment and support using released, stable objects.

    Geometric TRUE is a configurable hypothesis from observed geometry. Its
    physical accuracy requires calibration and later closed-loop evaluation.
    FALSE from a clear geometric contradiction is available immediately.
    Missing identity, visibility, release information or calibration is UNKNOWN.
    """

    def __init__(
        self,
        *,
        stable_observations: int = 2,
        min_confidence: float = 0.8,
        max_gap_steps: int = 80,
        placement_tolerance: float = 0.02,
        stability_tolerance: float = 0.01,
    ):
        if stable_observations < 2 or max_gap_steps < 1:
            raise ValueError(
                "Temporal verification needs two observations and a positive step gap"
            )
        if not 0 <= min_confidence <= 1:
            raise ValueError("Invalid confidence threshold")
        if not all(
            math.isfinite(x) and x >= 0
            for x in (placement_tolerance, stability_tolerance)
        ):
            raise ValueError("Geometric tolerances must be finite and nonnegative")
        self.stable_observations = stable_observations
        self.min_confidence = min_confidence
        self.max_gap_steps = max_gap_steps
        self.placement_tolerance = placement_tolerance
        self.stability_tolerance = stability_tolerance
        self._samples: dict[tuple, list[_Sample]] = {}

    def reset(self) -> None:
        self._samples.clear()

    def evaluate(
        self,
        predicate,
        objects: Mapping[str, ObjectObservation],
        step: int,
        *,
        history: bool = True,
    ) -> Verification:
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise ValueError("Observation step must be a nonnegative integer")
        name, ids = predicate.name, tuple(predicate.args)
        key = (name, ids)
        observations = [objects.get(object_id) for object_id in ids]
        frames = tuple(
            dict.fromkeys(
                o.frame_id for o in observations if o is not None and o.frame_id
            )
        )
        confidence = min(
            (o.confidence for o in observations if o is not None), default=0.0
        )
        truth, reason = self._instantaneous(name, observations)
        result = Verification(truth, reason, ids, frames, confidence)
        trusted_holding = (
            name == "holding"
            and observations[0] is not None
            and observations[0].source == "proprioception"
        )
        if not history or trusted_holding:
            return result
        samples = self._samples.setdefault(key, [])
        if samples and step < samples[-1].step:
            return Verification(
                "UNKNOWN", "out_of_order_observation", ids, frames, confidence
            )
        if truth != "TRUE":
            samples.clear()
            return result
        if any(
            not o.frame_id or (name != "holding" and o.position is None)
            for o in observations
        ):
            samples.clear()
            return Verification(
                "UNKNOWN", "missing_temporal_provenance", ids, frames, confidence
            )
        sample = _Sample(
            step,
            tuple(o.frame_id for o in observations),
            tuple(
                o.position if name != "holding" else (0.0, 0.0, 0.0)
                for o in observations
            ),
            confidence,
        )
        verified_reason = (
            "consistent_holding_observations"
            if name == "holding"
            else "released_and_geometrically_stable"
        )
        if samples and step == samples[-1].step:
            if sample == samples[-1] and len(samples) >= self.stable_observations:
                evidence_frames = tuple(
                    dict.fromkeys(
                        frame for previous in samples for frame in previous.frames
                    )
                )
                return Verification(
                    "TRUE",
                    verified_reason,
                    ids,
                    evidence_frames,
                    min(previous.confidence for previous in samples),
                )
            return Verification(
                "UNKNOWN", "repeated_observation_step", ids, frames, confidence
            )
        if any(
            any(
                current == prior
                for current, prior in zip(sample.frames, previous.frames)
            )
            for previous in samples
        ):
            return Verification(
                "UNKNOWN", "repeated_observation_frame", ids, frames, confidence
            )
        if samples and (
            step - samples[-1].step > self.max_gap_steps
            or any(
                np.max(
                    np.linalg.norm(
                        np.asarray(sample.positions) - np.asarray(previous.positions),
                        axis=1,
                    )
                )
                > self.stability_tolerance
                for previous in samples
            )
        ):
            samples.clear()
        samples.append(sample)
        del samples[: -self.stable_observations]
        if len(samples) < self.stable_observations:
            return Verification(
                "UNKNOWN", "awaiting_stable_observations", ids, frames, confidence
            )
        evidence_frames = tuple(
            dict.fromkeys(frame for sample in samples for frame in sample.frames)
        )
        return Verification(
            "TRUE",
            verified_reason,
            ids,
            evidence_frames,
            min(previous.confidence for previous in samples),
        )

    def _instantaneous(self, name, observations):
        arity = 1 if name == "holding" else 2
        if name not in {"holding", "in", "inside", "on"} or len(observations) != arity:
            return "UNKNOWN", "unsupported_predicate"
        if any(o is None or not o.visible for o in observations):
            return "UNKNOWN", "object_not_observed"
        if any(o.confidence < self.min_confidence for o in observations):
            return "UNKNOWN", "low_confidence"
        obj = observations[0]
        if name == "holding":
            if obj.holding is None:
                return "UNKNOWN", "holding_not_observed"
            return ("TRUE" if obj.holding else "FALSE"), "online_holding_signal"
        if observations[0].id == observations[1].id:
            return "UNKNOWN", "self_relation"
        if obj.holding is True:
            return "FALSE", "object_still_held"
        target = observations[1]
        if obj.bounds is None or target.bounds is None:
            return "UNKNOWN", "missing_metric_geometry"
        low, high = np.asarray(obj.bounds)
        target_low, target_high = np.asarray(target.bounds)
        tolerance = self.placement_tolerance
        if name in {"in", "inside"}:
            if target.interior_bounds is None:
                return "UNKNOWN", "container_interior_unknown"
            interior_low, interior_high = np.asarray(target.interior_bounds)
            if np.any(interior_high <= interior_low):
                return "UNKNOWN", "container_interior_degenerate"
            if np.any(high < interior_low - tolerance) or np.any(
                low > interior_high + tolerance
            ):
                return "FALSE", "outside_container_interior"
            fits = np.all(low >= interior_low) and np.all(high <= interior_high)
        else:
            if np.any(high[:2] <= low[:2]) or np.any(target_high[:2] <= target_low[:2]):
                return "UNKNOWN", "support_footprint_degenerate"
            support_z = (
                target.support_z if target.support_z is not None else target_high[2]
            )
            xy_overlap = np.minimum(high[:2], target_high[:2]) - np.maximum(
                low[:2], target_low[:2]
            )
            if np.any(xy_overlap < -tolerance) or abs(low[2] - support_z) > tolerance:
                return "FALSE", "support_geometry_contradicted"
            center_xy = (low[:2] + high[:2]) / 2
            fits = np.all(center_xy >= target_low[:2]) and np.all(
                center_xy <= target_high[:2]
            )
        if not fits:
            return "UNKNOWN", "placement_boundary_uncertain"
        if obj.holding is None:
            return "UNKNOWN", "release_not_observed"
        return "TRUE", "released_and_geometry_matches"


class RGBDGrounder:
    """Ground unique descriptions through existing GDino and SAM2 endpoints.

    The client may be a :class:`GraspClient` or a test double with its detect
    and segment methods. Empty/ambiguous detections, server failures and missing
    calibration produce unobserved objects. No local model is loaded here.
    """

    def __init__(
        self, client, *, env_mode: str = "libero", min_confidence: float = 0.8
    ):
        if env_mode not in {"libero", "robolab", "vlabench", "robocasa"}:
            raise ValueError("Unsupported camera convention")
        if not 0 <= min_confidence <= 1:
            raise ValueError("Invalid confidence threshold")
        self.client = client
        self.env_mode = env_mode
        self.min_confidence = min_confidence

    def observe(
        self, obs: Mapping, objects: list[dict], step: int
    ) -> dict[str, ObjectObservation]:
        frame_id = str(obs.get("observation/frame_id", f"step:{step}"))
        result = {}
        references = []
        for spec in objects:
            object_id = spec["id"]
            if object_id in result:
                raise ValueError("Duplicate plan object ID")
            result[object_id] = ObjectObservation(
                object_id,
                visible=False,
                confidence=0.0,
                frame_id=frame_id,
            )
            names = [spec.get("description", ""), *spec.get("aliases", [])]
            references.append({str(name).strip().casefold() for name in names} - {""})
        try:
            image, depth, K, transform = self._camera_inputs(obs)
        except (KeyError, ValueError, TypeError):
            return result
        # Descriptions refer to the policy image seen by the VLM. LIBERO's
        # camera calibration instead uses the horizontally unmirrored frame.
        unmirror_projection = (
            self.env_mode == "libero" and "observation/image_for_grasp" not in obs
        )
        projection_depth = depth[:, ::-1] if unmirror_projection else depth
        masks = {}
        for index, spec in enumerate(objects):
            description = spec.get("description", "").strip()
            if not description or any(
                references[index] & refs
                for n, refs in enumerate(references)
                if n != index
            ):
                continue
            try:
                candidates = self.client.detect(image, description.rstrip(".") + ".")
                candidates = [
                    d
                    for d in candidates
                    if math.isfinite(d["score"])
                    and self.min_confidence <= d["score"] <= 1
                ]
                if len(candidates) != 1:
                    continue
                detection = candidates[0]
                box = np.asarray(detection["box"], dtype=float)
                height, width = depth.shape
                if (
                    box.shape != (4,)
                    or not np.isfinite(box).all()
                    or not (
                        0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height
                    )
                ):
                    continue
                mask, score = self.client.segment(
                    image,
                    (box[0] + box[2]) / (2 * width),
                    (box[1] + box[3]) / (2 * height),
                    box=tuple(box.astype(int)),
                )
                if not math.isfinite(score) or not 0 <= score <= 1:
                    continue
                confidence = min(float(detection["score"]), float(score))
                if confidence < self.min_confidence:
                    continue
                observed = observation_from_mask(
                    spec["id"],
                    projection_depth,
                    np.fliplr(mask) if unmirror_projection else mask,
                    K,
                    transform,
                    frame_id=frame_id,
                    confidence=confidence,
                    holding=self._released(obs),
                )
                result[spec["id"]] = observed
                masks[spec["id"]] = np.asarray(mask, dtype=bool)
            except (ValueError, TypeError, KeyError, RuntimeError, OSError):
                continue
        ids = tuple(masks)
        for index, left in enumerate(ids):
            for right in ids[index + 1 :]:
                union = np.count_nonzero(masks[left] | masks[right])
                if union and np.count_nonzero(masks[left] & masks[right]) / union > 0.5:
                    result[left] = ObjectObservation(
                        left, visible=False, confidence=0.0, frame_id=frame_id
                    )
                    result[right] = ObjectObservation(
                        right, visible=False, confidence=0.0, frame_id=frame_id
                    )
        return result

    @staticmethod
    def _released(obs):
        value = obs.get("observation/gripper_position")
        if value is None:
            return None
        value = np.asarray(value, dtype=float).reshape(-1)
        if value.size == 1 and np.isfinite(value[0]) and 0 <= value[0] <= 0.1:
            return False
        return None

    def _camera_inputs(self, obs):
        if "observation/image_for_grasp" in obs:
            image = np.asarray(obs["observation/image_for_grasp"])
            depth = np.asarray(obs["observation/depth_for_grasp"])
        elif self.env_mode == "libero":
            image = np.asarray(obs["observation/image_raw"])
            depth = np.asarray(obs["observation/depth_agentview"])
        else:
            image = np.asarray(obs["observation/exterior_image_1_left_raw"])
            depth = np.asarray(obs["observation/depth_external"])
        if depth.ndim == 3 and depth.shape[-1] == 1:
            depth = depth[..., 0]
        if (
            image.dtype != np.uint8
            or image.shape != (*depth.shape, 3)
            or depth.ndim != 2
        ):
            raise ValueError(
                "Registered full-resolution RGB and metric depth are required"
            )
        K = np.asarray(obs["observation/camera_K"], dtype=float).reshape(3, 3)
        transform = np.asarray(
            obs["observation/camera_extrinsic"], dtype=float
        ).reshape(4, 4)
        return np.ascontiguousarray(image), np.ascontiguousarray(depth), K, transform
