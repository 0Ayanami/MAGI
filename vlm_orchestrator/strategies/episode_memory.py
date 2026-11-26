"""Episode-scoped evidence used by the subgoal planner and monitor."""

from collections import OrderedDict, deque
from dataclasses import dataclass
import json
import re
from time import perf_counter

import numpy as np


def placement(instruction):
    """Identify an atomic placement whose object and plate can be associated visually."""
    text = instruction.lower().replace("-", " ")
    match = re.search(
        r"\b((?:yellow and white|white) (?:mug|cup)|bowl|bottle)\b.*?"
        r"\bon(?:to| top of)? (?:the |a )?(?:(left|right|same|original) )?plate\b", text,
    )
    if not match or not re.search(r"\b(?:put|place|set)\b", text):
        return None
    if len(re.findall(r"\b(?:mug|cup|bowl|bottle|plate)\b", text)) != 2:
        return None
    if re.search(r"\b(?:and|then|before|after|while)\b", text[match.end():]):
        return None
    if re.search(r"\b(?:[2-9]|\d{2,}|two|three|four|five|six|seven|eight|nine|ten)\s+times\b", text):
        return None
    side = match[2] if match[2] in {"left", "right"} else ""
    return match[1].replace("cup", "mug"), side


class EpisodeSummary:
    def __init__(self, goal):
        self.goal = goal
        self.events = deque(maxlen=24)
        self.results = OrderedDict()
        self.repeated_placement = bool(re.search(
            r"\b(?:[2-9]|\d{2,}|two|three|four|five|six|seven|eight|nine|ten)\s+times\b", goal, re.I,
        ))
        self.placement_cycles = OrderedDict()

    def record(self, skill, step, source, status, evidence, reason="", occurrence=None):
        entry = dict(skill=skill[:256], step=step, source=source, status=status,
                     evidence=evidence, reason=reason[:240])
        self.events.append(entry)
        if status in {"issued", "attempt_failed"}:
            return
        relation = placement(skill)
        if self.repeated_placement and relation and occurrence is not None:
            if status in {"observed_complete", "model_complete"}:
                previous = self.placement_cycles.get(occurrence)
                if previous is None or status == "observed_complete":
                    self.placement_cycles[occurrence] = dict(entry, occurrence=occurrence)
                    if len(self.placement_cycles) > 16:
                        self.placement_cycles.popitem(last=False)
        key = relation or skill.casefold()
        if (status == "in_progress" and key in self.results
                and self.results[key]["status"] in {"observed_complete", "model_complete"}):
            entry["status"] = "invalidated"
        elif status == "in_progress":
            return
        self.results[key] = entry
        self.results.move_to_end(key)
        if len(self.results) > 16:
            self.results.popitem(last=False)

    def conflict(self, reason, step, evidence):
        # A model report raises a question about its named relation. It
        # cannot turn the relation into an observation-supported failure.
        for entry in list(self.results.values()):
            relation = placement(entry["skill"])
            if relation and entry["status"] in {"model_complete", "observed_complete"}:
                for clause in re.split(r"[.;!?]", reason.casefold()):
                    if (relation[0] in clause and
                            re.search(r"\b(dropped|fell|fallen|no longer|off the plate)\b", clause)):
                        self.record(entry["skill"], step, "vlm", "needs_check", evidence, reason)
                        break

    def prompt(self):
        data = dict(goal=self.goal, results=list(self.results.values()), recent=list(self.events))
        repetition = ""
        if self.repeated_placement:
            data["placement_cycles"] = list(self.placement_cycles.values())
            repetition = ("For repetition, use placement_cycles as the ledger. Count each observed "
                          "lift-and-return once, even if the relation is the same. A model-only "
                          "completion still needs verification and a stationary object does not "
                          "establish another cycle. ")
        return ("\nEpisode evidence (instructions issued are attempts). Preserve supported "
                "results that still hold in NOW, revisit invalid or uncertain prerequisites, "
                "and preserve the original order and repetition requirements.\n" +
                repetition + json.dumps(data, ensure_ascii=False))


@dataclass
class Sample:
    step: int
    obj: np.ndarray
    hand: np.ndarray
    plate: np.ndarray
    on_plate: bool
    width: float
    ee: np.ndarray


@dataclass
class PlacementObservation:
    step: int
    obs: dict
    image: np.ndarray
    width: float
    ee: np.ndarray


def mug_color_matches(image, box, name):
    """Reject yellow/white confusion using pixels inside each detected mug."""
    h, w = image.shape[:2]
    x, y, right, bottom = np.clip(box, 0, [w, h, w, h]).astype(int)
    roi = image[y:bottom+1, x:right+1].astype(float)
    if not roi.size:
        return False
    red, green, blue = roi.transpose(2, 0, 1)
    yellow = np.mean((red > blue*1.17+6) & (green > blue*1.14+5) & (green > red*.86))
    if name.startswith("yellow"):
        return yellow > .09
    neutral = np.mean((np.ptp(roi, axis=2) < 18) & (roi.mean(axis=2) > 45))
    return neutral > .25 and yellow < .04


class PlacementWindow:
    """Coordinates in image-width/height units, finger opening in metres."""

    def __init__(self, current_only=False):
        limit = 1 if current_only else 8
        self.frames = deque(maxlen=limit)
        self.observations = deque(maxlen=limit)
        self.plate_anchors = {}
        self.capture_step = None
        self.skill = None
        self.last_step = None
        self.last_check = None
        self.unknown_checks = 0

    def reset(self, skill=None):
        self.frames.clear()
        self.observations.clear()
        self.capture_step = None
        self.skill = skill
        self.last_step = self.last_check = None
        self.unknown_checks = 0

    def capture(self, obs, image, step, skill):
        """Save cheap observation history without running the detector on the control path."""
        if self.skill != skill:
            self.reset(skill)
        if self.capture_step == step:
            return
        if self.capture_step is not None and not 0 < step - self.capture_step <= 10:
            self.observations.clear()
        self.capture_step = step
        joints = np.asarray(obs.get("observation/state", [])).ravel()
        if image is None or joints.size != 8 or not np.isfinite(joints).all():
            self.observations.clear()
            return
        fields = ("observation/camera_K", "observation/camera_extrinsic")
        inputs = {key: np.array(obs[key], copy=True) for key in fields if key in obs}
        inputs["observation/state"] = joints.copy()
        self.observations.append(PlacementObservation(
            step, inputs, image.copy(), float(np.mean(np.abs(joints[-2:]))), joints[:3].copy(),
        ))

    def measure(self, detector, initial_image=None):
        """Detect at most four frames per check after rejecting impossible kinematics."""
        self.frames.clear()
        observations = list(self.observations)
        recent = observations[-3:]
        selected = []
        if len(recent) == 3 and all(item.width < .032 for item in recent):
            if np.linalg.norm(recent[-1].ee - recent[0].ee) > .015:
                selected = recent
        elif len(recent) == 3 and all(item.width > .033 for item in recent):
            relation = placement(self.skill or "")
            require_lift = relation is not None and relation[0] in {"bowl", "bottle"}
            held = next((item for item in reversed(observations[:-3])
                         if item.width < .032 and
                         (not require_lift or item.ee[2] - recent[0].ee[2] > .015)), None)
            if held is not None and np.linalg.norm(recent[-1].ee - held.ee) > .015:
                selected = [held, *recent]
        started = perf_counter()
        for item in selected:
            self._measure_frame(item.obs, item.image, item.step, self.skill, detector, initial_image)
            if not self.frames:
                break
        return len(selected), perf_counter() - started

    def observe(self, obs, image, step, skill, detector):
        if self.skill != skill:
            self.reset(skill)
        if self.last_step == step:
            return
        if self.last_step is not None and not 0 < step - self.last_step <= 10:
            self.frames.clear()
        self.last_step = step
        self._measure_frame(obs, image, step, skill, detector)

    def _origin_plate(self, initial_image, label, detector):
        """Find the unique plate directly supporting the object at episode start."""
        if label not in self.plate_anchors and initial_image is not None:
            h, w = initial_image.shape[:2]
            objects = detector.detect(initial_image, label, score_threshold=.40)
            anchor = None
            if len(objects) == 1:
                left, _, right, bottom = objects[0]["box"]
                foot = np.array([(left + right) / 2, bottom])
                plates = [np.asarray(hit["box"]) for hit in
                          detector.detect(initial_image, "plate", score_threshold=.40)]
                supports = [box for box in plates if np.all(box[:2] <= foot) and np.all(foot <= box[2:])]
                if len(supports) == 1:
                    box = supports[0]
                    anchor = (box[:2] + box[2:]) / [2*w, 2*h]
            self.plate_anchors[label] = anchor
        return self.plate_anchors.get(label)

    def _measure_frame(self, obs, image, step, skill, detector, initial_image=None):
        relation = placement(skill)
        joints = np.asarray(obs.get("observation/state", [])).ravel()
        if image is None or not relation or joints.size != 8 or not np.isfinite(joints).all():
            self.frames.clear()
            return
        h, w = image.shape[:2]
        boxes = []
        label = "mug" if relation[0].endswith("mug") else relation[0]
        for phrase in (label, "robot gripper", "plate"):
            if phrase == "robot gripper" and "observation/camera_K" in obs and "observation/camera_extrinsic" in obs:
                K = np.asarray(obs["observation/camera_K"]).reshape(3, 3)
                pose = np.asarray(obs["observation/camera_extrinsic"]).reshape(4, 4)
                camera_point = pose[:3, :3].T @ (joints[:3] - pose[:3, 3])
                uv = K @ camera_point
                if uv[2] <= 0 or not np.isfinite(uv).all():
                    self.frames.clear()
                    return
                # LIBERO's policy RGB has an extra horizontal mirror.
                hand = (np.array([w-1-uv[0]/uv[2], uv[1]/uv[2]]) / [w, h])
                boxes.append(np.tile(hand, 2))
                continue
            hits = detector.detect(image, phrase, score_threshold=0.40)
            if phrase == "mug":
                hits = [hit for hit in hits if mug_color_matches(image, hit["box"], relation[0])]
            candidates = sorted((np.asarray(hit["box"]) / [w, h, w, h] for hit in hits),
                                key=lambda box: box[0])
            if phrase == "plate" and relation[1] and len(candidates) == 2:
                if candidates[0][2] < candidates[1][0]:
                    candidates = [candidates[int(relation[1] == "left")]]
            elif phrase == "plate" and relation[1]:
                candidates = []  # one visible plate cannot establish left/right
            elif phrase == "plate" and label in {"bowl", "bottle"} and initial_image is not None:
                anchor = self._origin_plate(initial_image, label, detector)
                candidates = [box for box in candidates if anchor is not None
                              and np.linalg.norm((box[:2] + box[2:])/2 - anchor) < .06]
            if len(candidates) != 1:
                self.frames.clear()
                return
            boxes.append(candidates[0])
        obj, hand, plate = boxes
        centers = [(box[:2] + box[2:]) / 2 for box in boxes]
        foot = np.array([centers[0][0], obj[3]])
        on_plate = bool(np.all(plate[:2] <= foot) and np.all(foot <= plate[2:]))
        self.frames.append(Sample(step, *centers, on_plate,
                                  float(np.mean(np.abs(joints[-2:]))), joints[:3].copy()))

    def verdict(self):
        frames = list(self.frames)
        norm = np.linalg.norm
        if len(frames) >= 3:
            recent = frames[-3:]
            a, b = recent[0], recent[-1]
            match = placement(self.skill or "")
            require_lift = match is not None and match[0] in {"bowl", "bottle"}
            if (all(f.width < .032 and norm(f.hand - f.obj) < .14 for f in recent)
                    and norm(b.ee - a.ee) > .015 and norm(b.obj - a.obj) > .025
                    and all(norm((f.obj-a.obj) - (f.hand-a.hand)) < .025 for f in recent)):
                return "carrying"
            if (all(f.width > .033 and f.on_plate for f in recent)
                    and all(norm((f.obj-f.plate) - (a.obj-a.plate)) < .018 for f in recent)
                    and any(f.width < .032 and norm(f.hand-f.obj) < .14
                            and norm(f.obj-a.obj) < .09
                            and norm(b.hand-b.obj) - norm(f.hand-f.obj) > .045
                            and norm(b.ee-f.ee) > .015
                            and (not require_lift or (
                                f.ee[2] - a.ee[2] > .015
                                and norm((f.obj-f.plate) - (a.obj-a.plate)) > .025
                            )) for f in frames[:-3])
                    and norm(b.hand-b.obj) > .12):
                return "stable"
        return "unknown"
