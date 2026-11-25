"""Episode-scoped evidence used by the subgoal planner and monitor."""

from collections import OrderedDict, deque
from dataclasses import dataclass
import json
import re

import numpy as np


def placement(instruction):
    """Identify the supported atomic LIBERO mug-on-plate relation."""
    text = instruction.lower().replace("-", " ")
    match = re.search(r"\b((?:yellow and white|white) (?:mug|cup))\b.*?\bon(?:to| top of)? (?:the |a )?(?:(left|right) )?plate\b", text)
    if not match or not re.search(r"\b(put|place|set)\b", text):
        return None
    if len(re.findall(r"\b(?:mug|cup)\b", text)) != 1:
        return None
    return match[1].replace("cup", "mug"), match[2] or ""


class EpisodeSummary:
    def __init__(self, goal):
        self.goal = goal
        self.events = deque(maxlen=24)
        self.results = OrderedDict()

    def record(self, skill, step, source, status, evidence, reason=""):
        entry = dict(skill=skill[:256], step=step, source=source, status=status,
                     evidence=evidence, reason=reason[:240])
        self.events.append(entry)
        if status in {"issued", "attempt_failed"}:
            return
        relation = placement(skill)
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
        return ("\nEpisode evidence (instructions issued are attempts). Preserve supported "
                "results that still hold in NOW, revisit invalid or uncertain prerequisites, "
                "and preserve the original order and repetition requirements.\n" +
                json.dumps(data, ensure_ascii=False))


@dataclass
class Sample:
    step: int
    obj: np.ndarray
    hand: np.ndarray
    plate: np.ndarray
    on_plate: bool
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
        self.frames = deque(maxlen=1 if current_only else 8)
        self.skill = None
        self.last_step = None
        self.last_check = None
        self.unknown_checks = 0

    def reset(self, skill=None):
        self.frames.clear()
        self.skill = skill
        self.last_step = self.last_check = None
        self.unknown_checks = 0

    def observe(self, obs, image, step, skill, detector):
        if self.skill != skill:
            self.reset(skill)
        if self.last_step == step:
            return
        if self.last_step is not None and not 0 < step - self.last_step <= 10:
            self.frames.clear()
        self.last_step = step
        relation = placement(skill)
        joints = np.asarray(obs.get("observation/state", [])).ravel()
        if image is None or not relation or joints.size != 8 or not np.isfinite(joints).all():
            self.frames.clear()
            return
        h, w = image.shape[:2]
        boxes = []
        for phrase in ("mug", "robot gripper", "plate"):
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
            if (all(f.width < .032 and norm(f.hand - f.obj) < .14 for f in recent)
                    and norm(b.ee - a.ee) > .015 and norm(b.obj - a.obj) > .025
                    and all(norm((f.obj-a.obj) - (f.hand-a.hand)) < .025 for f in recent)):
                return "carrying"
            if (all(f.width > .033 and f.on_plate for f in recent)
                    and all(norm((f.obj-f.plate) - (a.obj-a.plate)) < .018 for f in recent)
                    and any(f.width < .032 and norm(f.hand-f.obj) < .14
                            and norm(f.obj-a.obj) < .09
                            and norm(b.hand-b.obj) - norm(f.hand-f.obj) > .045
                            and norm(b.ee-f.ee) > .015 for f in frames[:-3])
                    and norm(b.hand-b.obj) > .12):
                return "stable"
        return "unknown"
