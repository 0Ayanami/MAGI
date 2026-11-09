"""Episode-local REMM adaptation and experiment configuration.

Only the explicitly documented online object stream or the configured RGB-D
grounder supplies observations. Simulator ground truth is never consumed here.
"""

from __future__ import annotations

import json
import math
from copy import deepcopy
from dataclasses import asdict, dataclass

from .perception import (
    GeometryVerifier,
    ObjectObservation,
    RGBDGrounder,
    observations_from_payload,
)
from .store import MemoryStore
from .types import Evidence, Predicate, Truth


@dataclass(frozen=True)
class RemmConfig:
    mode: str = "episodic"
    gate: str = "shadow"
    max_reobservations: int = 3
    perception_url: str | None = None
    env_mode: str = "libero"
    min_confidence: float = 0.8
    stable_observations: int = 2
    max_gap_steps: int = 80
    placement_tolerance: float = 0.02

    def __post_init__(self):
        if self.mode not in {"current", "episodic"}:
            raise ValueError("memory mode must be current or episodic")
        if self.gate not in {"shadow", "enforce"}:
            raise ValueError("memory gate must be shadow or enforce")
        if (
            type(self.max_reobservations) is not int
            or self.max_reobservations < 0
            or type(self.stable_observations) is not int
            or self.stable_observations < 2
        ):
            raise ValueError(
                "nonnegative retry budget and at least two stable observations required"
            )
        if not math.isfinite(self.min_confidence) or not 0 <= self.min_confidence <= 1:
            raise ValueError("confidence must be finite and between zero and one")
        if (
            type(self.max_gap_steps) is not int
            or self.max_gap_steps < 1
            or not math.isfinite(self.placement_tolerance)
            or self.placement_tolerance <= 0
        ):
            raise ValueError(
                "positive observation gap and placement tolerance required"
            )


class MemoryRuntime:
    """Coordinate identity references, evidence, current verification and history."""

    def __init__(self, config: RemmConfig):
        self.config = config
        self.grounder = None
        if config.perception_url:
            from urllib.parse import urlparse

            from vlm_orchestrator.grasp.client import GraspClient

            url = urlparse(config.perception_url)
            if url.scheme != "http" or not url.hostname or url.username or url.password:
                raise ValueError(
                    "perception URL must be an HTTP endpoint without credentials"
                )
            if url.path not in {"", "/"} or url.query or url.fragment:
                raise ValueError("perception URL must contain only host and port")
            self.grounder = RGBDGrounder(
                GraspClient(config.perception_url, timeout=10),
                env_mode=config.env_mode,
                min_confidence=config.min_confidence,
            )
        self.reset(0)

    def reset(self, episode_id):
        self.store = MemoryStore(episode_id)
        self.verifier = GeometryVerifier(
            stable_observations=self.config.stable_observations,
            min_confidence=self.config.min_confidence,
            max_gap_steps=self.config.max_gap_steps,
            placement_tolerance=self.config.placement_tolerance,
        )
        self.objects = []
        self.contracts = {}
        self.plan_version = 0
        self.plan_error = None
        self.step = -1
        self.results = {}
        self.observation_error = None
        self._goal_ids = {}
        self._frame_evidence = {}
        self._new_events = []
        self._object_specs = {}
        self._completed_contracts = {}
        self._last_index = -1

    def set_plan(self, payload, subgoals):
        """Install a plan contract, failing uncovered goals closed.

        The VLM proposes symbolic references only. Numeric geometry comes from
        the online perception input. Failed or partial contracts remain visible
        as coverage diagnostics and can never borrow another plan's completion.
        """
        self.plan_version += 1
        self.contracts = {}
        self._goal_ids = {}
        self.results = {}
        self.plan_error = None
        self._last_index = len(subgoals) - 1
        self.verifier.reset()
        try:
            if not isinstance(payload, dict):
                raise ValueError("planner must return a memory object")
            objects = payload.get("objects", [])
            goals = payload.get("goals", [])
            if not isinstance(objects, list) or not isinstance(goals, list):
                raise ValueError("objects and goals must be lists")
            specs = []
            ids = set()
            for item in objects:
                if not isinstance(item, dict):
                    raise ValueError("object reference must be an object")
                object_id, description = item.get("id"), item.get("description")
                if (
                    not isinstance(object_id, str)
                    or not object_id.strip()
                    or object_id in ids
                ):
                    raise ValueError("unique nonempty object IDs required")
                if not isinstance(description, str) or not description.strip():
                    raise ValueError("object description required")
                previous = self._object_specs.get(object_id)
                if previous and previous["description"] != description:
                    raise ValueError("object ID rebound to a different description")
                ids.add(object_id)
                specs.append({"id": object_id, "description": description})
            contracts = {}
            for goal in goals:
                index = goal.get("subgoal_index")
                if (
                    type(index) is not int
                    or not 0 <= index < len(subgoals)
                    or index in contracts
                ):
                    raise ValueError("unique in-range subgoal_index required")
                predicates = goal.get("predicates")
                if not isinstance(predicates, list) or not predicates:
                    raise ValueError("at least one completion predicate required")
                parsed = []
                for predicate in predicates:
                    name, args = predicate.get("name"), predicate.get("args")
                    arity = {"holding": 1, "on": 2, "in": 2}.get(name)
                    if not isinstance(args, list) or len(args) != arity:
                        raise ValueError(
                            "supported predicates are holding(object), on(object,support), in(object,container)"
                        )
                    if any(not isinstance(arg, str) or arg not in ids for arg in args):
                        raise ValueError(
                            "predicate arguments must reference declared object IDs"
                        )
                    if len(set(args)) != len(args):
                        raise ValueError("predicate arguments must be distinct")
                    parsed.append(Predicate(name, tuple(args)))
                contracts[index] = tuple(parsed)
            self._object_specs.update({item["id"]: item for item in specs})
            self.objects = list(self._object_specs.values())
            self.contracts = contracts
            self._goal_ids = {
                index: f"plan{self.plan_version}:subgoal{index}" for index in contracts
            }
        except (AttributeError, TypeError, ValueError) as error:
            self.objects = list(self._object_specs.values())
            self.plan_error = str(error)

    def invalidate_current(self, step, reason):
        self.results = {}
        self.observation_error = reason
        self._new_events = []
        if type(step) is not int or step < 0:
            raise ValueError("nonnegative integer simulation step required")
        if step < self.step:
            raise ValueError("out-of-order observation")
        self.step = step

    def observe(self, obs, step):
        self.invalidate_current(step, "observation not yet verified")
        if self.config.mode == "current":
            self.store = MemoryStore(self.store.episode_id)
            self._frame_evidence = {}
        frame_id = f"step:{step}"
        store = self.store.fork()
        verifier = deepcopy(self.verifier)
        frame_evidence = dict(self._frame_evidence)
        try:
            if "observation/remm_objects" in obs:
                objects = observations_from_payload(
                    obs["observation/remm_objects"],
                    frame_id=frame_id,
                )
            elif self.grounder is not None:
                # Explicit whitelist makes GT and any future unrelated client
                # fields unavailable even to an external perception adapter.
                allowed = (
                    "observation/image_raw",
                    "observation/image",
                    "observation/depth_agentview",
                    "observation/depth_external",
                    "observation/camera_K",
                    "observation/camera_extrinsic",
                    "observation/gripper_position",
                    "observation/frame_id",
                    "observation/exterior_image_1_left_raw",
                    "observation/image_for_grasp",
                    "observation/depth_for_grasp",
                )
                online = {key: obs[key] for key in allowed if key in obs}
                objects = self.grounder.observe(online, self.objects, step)
            else:
                objects = {}
            declared = {item["id"]: item["description"] for item in self.objects}
            objects = {key: value for key, value in objects.items() if key in declared}
            for object_id in declared:
                objects.setdefault(
                    object_id,
                    ObjectObservation(
                        object_id,
                        visible=False,
                        confidence=0.0,
                        frame_id=frame_id,
                    ),
                )
            for object_id, item in objects.items():
                event = Evidence(
                    id=f"plan{self.plan_version}:step{step}:object:{object_id}",
                    step=step,
                    source=item.source,
                    object_ids=(object_id,),
                    frame_id=item.frame_id,
                    payload=asdict(item),
                )
                store.add_evidence(event)
                self._new_events.append(event.to_dict())
                frame_evidence[(object_id, item.frame_id)] = event.id
                store.observe_object(
                    object_id,
                    step=step,
                    label=declared[object_id],
                    position=item.position,
                    visible=item.visible,
                    identity_confident=item.confidence >= self.config.min_confidence,
                    evidence_ids=(event.id,),
                )
            results = {}
            predicates = {p for group in self.contracts.values() for p in group}
            if self.config.mode == "episodic":
                predicates.update(
                    p for group in self._completed_contracts.values() for p in group
                )
            for predicate in sorted(predicates, key=lambda p: (p.name, p.args)):
                result = verifier.evaluate(
                    predicate,
                    objects,
                    step,
                    history=self.config.mode == "episodic",
                )
                truth = Truth(result.truth)
                evidence_ids = tuple(
                    dict.fromkeys(
                        frame_evidence[(obj, frame)]
                        for frame in result.frame_ids
                        for obj in result.object_ids
                        if (obj, frame) in frame_evidence
                    )
                )
                results[predicate] = {
                    "truth": truth.value,
                    "reason": result.reason,
                    "evidence_ids": list(evidence_ids),
                    "confidence": result.confidence,
                }
                # Occlusion or missing detections preserve earlier support.
                # Current completion still requires a fresh TRUE result below.
                if truth != Truth.UNKNOWN and evidence_ids:
                    store.assert_fact(
                        predicate,
                        truth,
                        evidence_ids=evidence_ids,
                        confidence=result.confidence,
                    )
            for goal in store.broken_goals:
                conditions = self._completed_contracts.get(goal, ())
                if conditions and all(
                    results.get(p, {}).get("truth") == "TRUE" for p in conditions
                ):
                    event = Evidence(
                        id=f"plan{self.plan_version}:step{step}:restore:{goal}",
                        step=step,
                        source="verification",
                        payload={
                            "goal_id": goal,
                            "reason": "physical postconditions reverified",
                        },
                    )
                    store.add_evidence(event)
                    self._new_events.append(event.to_dict())
                    store.complete_goal(goal, conditions, evidence_ids=(event.id,))
            self.results = results
            self.store = store
            self.verifier = verifier
            self._frame_evidence = frame_evidence
            self.observation_error = None
        except Exception as error:
            self.results = {}
            self._new_events = []
            self.observation_error = str(error)
            if isinstance(error, (ValueError, TypeError, KeyError, AttributeError)):
                raise ValueError(str(error)) from error
            raise

    def check_completion(self, index, step):
        predicates = self.contracts.get(index)
        if not predicates:
            return {
                "allowed": False,
                "status": "UNKNOWN",
                "reason": "uncovered subgoal",
                "evidence_ids": [],
            }
        if step != self.step or self.observation_error:
            return {
                "allowed": False,
                "status": "UNKNOWN",
                "reason": self.observation_error or "stale observation",
                "evidence_ids": [],
            }
        rows = [
            self.results.get(p, {"truth": "UNKNOWN", "evidence_ids": []})
            for p in predicates
        ]
        allowed = all(row["truth"] == "TRUE" for row in rows)
        if index == self._last_index and self.broken_goals:
            return {
                "allowed": False,
                "status": "FALSE",
                "reason": "previously completed physical conditions remain invalidated",
                "evidence_ids": [],
                "broken_goals": list(self.broken_goals),
            }
        status = (
            "TRUE"
            if allowed
            else "FALSE"
            if any(row["truth"] == "FALSE" for row in rows)
            else "UNKNOWN"
        )
        return {
            "allowed": allowed,
            "status": status,
            "reason": "verified physical postconditions"
            if allowed
            else "postconditions need verification",
            "evidence_ids": list(
                dict.fromkeys(e for row in rows for e in row["evidence_ids"])
            ),
        }

    def mark_complete(self, index):
        decision = self.check_completion(index, self.step)
        if not decision["allowed"]:
            raise ValueError("completion requires fresh verified physical support")
        event = Evidence(
            id=f"plan{self.plan_version}:step{self.step}:complete:{index}",
            step=self.step,
            source="verification",
            payload=decision,
        )
        self.store.add_evidence(event)
        self._new_events.append(event.to_dict())
        completed = self.store.complete_goal(
            self._goal_ids[index],
            self.contracts[index],
            evidence_ids=(event.id,),
        )
        self._completed_contracts[self._goal_ids[index]] = self.contracts[index]
        return completed

    @property
    def broken_goals(self):
        return self.store.broken_goals

    def snapshot(self):
        return {
            "config": asdict(self.config),
            "plan_version": self.plan_version,
            "step": self.step,
            "plan_error": self.plan_error,
            "observation_error": self.observation_error,
            "covered_subgoals": sorted(self.contracts),
            "broken_goals": list(self.broken_goals),
            "verification": [
                {"predicate": {"name": p.name, "args": list(p.args)}, **row}
                for p, row in self.results.items()
            ],
            "observation_events": self._new_events,
            "memory": self.store.snapshot(include_history=False),
        }

    def context(self):
        snapshot = self.snapshot()
        snapshot.pop("observation_events")
        if self.config.mode == "episodic":
            records = list(self.store.assertions.values())
            revisions = [r for r in records if r.status != "active"][-8:]
            events = {event.id: event for event in self.store.events}
            references = [
                evidence_id
                for row in self.results.values()
                for evidence_id in row["evidence_ids"]
            ]
            references.extend(
                evidence_id
                for record in records[-16:]
                for evidence_id in record.evidence_ids
            )
            references.extend(event.id for event in self.store.events[-12:])
            selected = list(dict.fromkeys(references))[:32]
            snapshot["memory"]["evidence_history"] = [
                events[evidence_id].to_dict() for evidence_id in selected
            ]
            snapshot["memory"]["assertion_revisions"] = [r.to_dict() for r in revisions]
            snapshot["memory"]["history_window"] = {
                "total_events": len(events),
                "included_events": len(selected),
                "total_assertions": len(records),
                "included_revisions": len(revisions),
            }
        return json.dumps(snapshot, ensure_ascii=False, allow_nan=False)
