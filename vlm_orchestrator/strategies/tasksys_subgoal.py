"""Execute optional TaskSys task graphs over EviMem and the LIBERO VLA."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field

from vlm_orchestrator.tasksys.graph import GOAL, TaskGraph, conflicts
from vlm_orchestrator.evimem.types import Predicate
from vlm_orchestrator.vlm import parse_json

from .evimem_subgoal import EviMemSubgoalStrategy
from .subgoal_base import MAX_RECYCLES

_NODE_SCHEMA = """
Each node contains id, skill, object_ids, preconditions and expected_effects.
Skills are pick, place_on and place_in. The executor renders its own VLA
instruction from the skill and object descriptions. Each node declares one expected effect, using
holding, on and in respectively. Every condition is {"name": ..., "args": [...]}
and references only declared IDs also listed in that node's object_ids.
Use [] for no preconditions. Do not invent required holding or containment
conditions if online observations cannot verify them. A place instruction may
include acquisition and release in the same VLA skill.
Combine grasp and placement into one place node when the task only requires a
final placement. Do not introduce a separate pick node in that case.
"""
_PLAN_PROMPT = """
Return one JSON object with objects (a list of {"id": ..., "description": ...}),
nodes (in reference execution order) and final_conditions (nonempty list of
physical conditions). Use short noun phrases containing only the object's
appearance and type, such as "silver bowl" or "white plate with a red rim".
Never mention neighboring objects, scene locations, or actions in descriptions.
These phrases are reused for detection after the objects move. Express
only real dependencies, and include all terminal task requirements using the
available predicates. Provide effects that establish downstream prerequisites.
The plan requests verification; it never certifies observed facts.
""" + _NODE_SCHEMA
_REPAIR_PROMPT = """
Return one JSON object with basis (copy it exactly from the request) and
repair_nodes (a nonempty ordered list). Use only existing object IDs and new
node IDs. These steps will be inserted immediately before the retained
continuation. Restore the required conditions from the current scene, preserve
protected effects, and keep the retained continuation and final conditions
achievable. Do not replace existing nodes, provide observed truth values,
or reissue unrelated completed steps.
""" + _NODE_SCHEMA

_MONITOR_PROMPT = """
Judge the current TaskSys step using the current camera views and EviMem object
observations. Return one JSON object with action (next, continue, wait, or replan)
and reason (a short explanation). Include confirmed_conditions copied from
candidate_conditions and evidence_refs copied from scene_ref or EviMem refs.
Confirm only conditions supported by the current scene. Choose next when all
expected effects and conditions_to_review are satisfied, citing scene_ref.
For place_on and place_in, verify the object is released and supported at the
destination. A stale position or open gripper alone cannot show placement.
In dispatch, continue requires all preconditions. In execute, continue keeps
the skill running if the conditions_to_review still hold. Use wait when a
condition needs another observation and the robot should hold still. Use
replan if the node or a prerequisite needs repair. In final, next confirms
the final conditions. On the last node, also confirm any final conditions
visible now so the executor can reuse this observation. Give a reason_code
of in_progress, insufficient_evidence, or condition_failed when applicable.
"""


@dataclass(frozen=True)
class EffectConfirmation:
    step: int
    refs: tuple[str, ...]
    positions: dict


@dataclass
class GraphExecution:
    graph: TaskGraph | None = None
    cursor: int = 0
    active: bool = False
    started_at: int = 0
    planning_attempts: int = 0
    last_planning_step: int | None = None
    repair_attempts: int = 0
    last_repair_step: int | None = None
    final_verified: bool = False
    last_monitor_step: int | None = None
    next_monitor_step: int | None = None
    effects: dict[Predicate, EffectConfirmation] = field(default_factory=dict)


class TaskSysSubgoalStrategy(EviMemSubgoalStrategy):
    """Select verified graph steps and request bounded local repair fragments."""

    def __init__(self, ctx, config, *, memory_config, **kwargs):
        if (
            memory_config.mode != "episodic" or memory_config.gate != "enforce"
            or memory_config.verification != "vlm"
            or kwargs.get("hitl_state") is not None
            or kwargs.get("initial_strategy") is not None
        ):
            raise ValueError("TaskSys requires episodic EviMem, VLM verification and autonomous graph planning")
        super().__init__(ctx, config, memory_config=memory_config, **kwargs)

    def _on_new_episode(self, obs, state, prompt):
        state.tasksys = GraphExecution()
        obs, state = super()._on_new_episode(obs, state, prompt)
        return self._tick_graph(obs, state)

    def _decompose_and_setup(self, obs, state, instruction, image):
        execution = state.tasksys
        execution.planning_attempts += 1
        execution.last_planning_step = state.episode_step
        content = self._build_image_message(
            f"Task: {instruction}", image, self.ctx.get_vlm_extra_images(obs),
        )
        payload = self._request_graph(state, "plan", _PLAN_PROMPT, content)
        try:
            graph = TaskGraph.parse(payload)
        except ValueError as error:
            self._log(state, "plan_rejected", reason=str(error))
            return
        runtime = self._runtime(state)
        runtime.set_plan({"objects": graph.to_dict()["objects"]}, ())
        if runtime.plan_error:
            self._log(state, "plan_rejected", reason=runtime.plan_error)
            return
        self._install_graph(state, graph)
        self._log(state, "plan_registered", graph=graph.to_dict())

    def _install_graph(self, state, graph):
        state.tasksys.graph = graph
        state.subgoals = [node.instruction for node in graph.nodes]
        state.subgoals_ordered = True
        self._select_instruction(state)

    def _select_instruction(self, state):
        state.current_subgoal_idx = min(state.tasksys.cursor, len(state.subgoals) - 1)
        state.rewritten_instruction = state.subgoals[state.current_subgoal_idx]
        state.flush_actions = True

    def _on_step(self, obs, state):
        execution = state.tasksys
        if execution.graph is None and execution.planning_attempts < MAX_RECYCLES:
            last = execution.last_planning_step
            if last is None or state.episode_step - last >= self.config.check_interval:
                image = self.ctx.get_vlm_image(obs)
                if image is not None:
                    self._decompose_and_setup(obs, state, state.original_instruction, image)
        self._observe(obs, state, object_ids=self._object_ids(execution))
        return self._tick_graph(obs, state)

    @staticmethod
    def _object_ids(execution):
        graph = execution.graph
        if graph is None:
            return set()
        cursor = execution.cursor
        objects = set(graph.nodes[cursor].object_ids) if cursor < len(graph.nodes) else set()
        if cursor >= len(graph.nodes) - 1:
            objects.update(obj for p in graph.final_conditions for obj in p.args)
        return objects

    def _tick_graph(self, obs, state):
        execution = state.tasksys
        if execution.graph is None:
            self._hold(state, "awaiting graph")
            return obs, state

        self._tick_model(obs, state)
        return self.ctx.set_prompt(obs, state.rewritten_instruction), state

    def _memory_context(self, state):
        # Graph requests supply node-scoped evidence explicitly.
        return None

    def _tick_model(self, obs, state):
        execution = state.tasksys
        graph = execution.graph
        step = state.episode_step
        if execution.final_verified:
            return
        final = execution.cursor == len(graph.nodes)
        node = None if final else graph.nodes[execution.cursor]
        if node is not None and not execution.active and not node.preconditions:
            self._dispatch_model_node(obs, state, node)
            return
        if final or not execution.active:
            self._hold(state, "awaiting VLM decision")
        runtime = self._runtime(state)
        interval = self.config.check_interval
        prerequisites = node.preconditions if node else ()
        review = prerequisites if not execution.active else tuple(
            p for p in prerequisites
            if not any(conflicts(p, effect) for effect in node.expected_effects)
            and self._needs_review(execution, runtime, p, step, interval)
        )
        if execution.active and review:
            self._hold(state, "awaiting dependency review")
        effects = graph.final_conditions if final else node.expected_effects
        final_conditions = graph.final_conditions if final or execution.cursor == len(graph.nodes) - 1 else ()
        object_ids = self._object_ids(execution)
        last = execution.last_monitor_step
        due = execution.next_monitor_step
        periodic = due is None or step >= due
        opened = runtime.last_open_step is not None and last is not None and runtime.last_open_step > last
        reacquired = last is not None and any(runtime.reacquired_steps.get(obj, -1) > last for obj in object_ids)
        early = (execution.active and not periodic and step >= due - interval // 2
                 and (opened or reacquired or bool(review)))
        if not periodic and not early:
            return
        image = self.ctx.get_vlm_image(obs)
        if image is None:
            return

        execution.last_monitor_step = step
        execution.next_monitor_step = (step if periodic else due) + interval
        node_id = GOAL if final else node.id
        phase = "final" if final else "execute" if execution.active else "dispatch"
        candidates = sorted(graph.predicates, key=str)
        scene_ref = f"scene:{step}"
        evidence = runtime.evidence(object_ids, since_step=execution.started_at if execution.active else step)
        refs = {scene_ref, *(sample["ref"] for item in evidence["objects"] for sample in item["trusted_samples"])}
        if evidence["gripper"]:
            refs.add(evidence["gripper"]["ref"])
        request = {
            "task": state.original_instruction, "step": step, "phase": phase,
            "node": node.to_dict() if node else None,
            "expected_effects": [p.to_dict() for p in effects],
            "conditions_to_review": [p.to_dict() for p in review],
            "prior_confirmations": [
                {"condition": p.to_dict(), "step": execution.effects[p].step,
                 "evidence_refs": execution.effects[p].refs}
                for p in review if p in execution.effects
            ],
            "final_conditions": [p.to_dict() for p in final_conditions],
            "completed_nodes": [n.id for n in graph.nodes[:execution.cursor]],
            "remaining_nodes": [n.to_dict() for n in graph.nodes[execution.cursor + 1:]],
            "candidate_conditions": [p.to_dict() for p in candidates],
            "steps_on_node": step - execution.started_at if execution.active else None,
            "node_timed_out": execution.active and step - execution.started_at >= self.config.subgoal_timeout,
            "scene_ref": scene_ref, "evidence": evidence,
        }
        content = self._build_image_message(
            json.dumps(request, ensure_ascii=False), image, self.ctx.get_vlm_extra_images(obs),
        )
        payload = self._request_graph(state, "monitor", _MONITOR_PROMPT, content)
        action = payload.get("action") if isinstance(payload, dict) else None
        reason = payload.get("reason") if isinstance(payload, dict) else None
        if action not in ("next", "continue", "wait", "replan") or not isinstance(reason, str):
            self._log(state, "monitor_rejected", node_id=node_id, phase=phase)
            return
        reported = payload.get("confirmed_conditions", [])
        cited = payload.get("evidence_refs", [])
        if (not isinstance(reported, list) or any(item not in request["candidate_conditions"] for item in reported)
                or not isinstance(cited, list) or any(not isinstance(ref, str) or ref not in refs for ref in cited)):
            self._log(state, "monitor_rejected", node_id=node_id, reason="invalid condition or evidence reference")
            return
        facts = {p for p in candidates if p.to_dict() in reported} if scene_ref in cited else set()
        self._log(state, "monitor", node_id=node_id, phase=phase, action=action, reason=reason,
                  reason_code=payload.get("reason_code"), trigger="periodic" if periodic else "event",
                  evidence_refs=cited, confirmed_conditions=[p.to_dict() for p in candidates if p in facts])

        if action in ("next", "continue") and (set(review) - facts or action == "next" and set(effects) - facts):
            self._log(state, "monitor_unconfirmed", node_id=node_id, action=action)
            return
        if action in ("next", "continue"):
            accepted = facts.intersection((*review, *(effects if action == "next" else ())))
            for condition in accepted:
                records = runtime.store.objects
                execution.effects[condition] = EffectConfirmation(step, tuple(cited), {
                    obj: records[obj].last_trusted_position for obj in condition.args if obj in records
                })

        if action == "next":
            if final:
                execution.final_verified = True
                self._log(state, "final_verification", verified=True, source="vlm", reason=reason)
                self._hold(state, "final conditions confirmed by VLM")
                return
            self._log(state, "node_verified", node_id=node_id, dispatched=execution.active,
                      source="vlm", reason=reason, evidence_refs=cited)
            execution.cursor += 1
            execution.active = False
            self._select_instruction(state)
            if final_conditions and set(final_conditions) <= facts:
                execution.final_verified = True
                self._log(state, "final_verification", verified=True, source="vlm", reason=reason,
                          evidence_refs=cited)
                self._hold(state, "final conditions confirmed by VLM")
                return
            if execution.cursor == len(graph.nodes) or graph.nodes[execution.cursor].preconditions:
                execution.last_monitor_step = None
                execution.next_monitor_step = None
                self._hold(state, "awaiting VLM decision")
            else:
                self._dispatch_model_node(obs, state, graph.nodes[execution.cursor])
        elif action == "continue":
            if phase == "dispatch":
                self._dispatch_model_node(obs, state, node)
            elif phase == "execute":
                self._release(state)
        elif action == "wait":
            self._hold(state, "VLM requested another observation")
        elif action == "replan":
            if scene_ref not in cited:
                self._log(state, "monitor_unconfirmed", node_id=node_id, action=action)
                return
            required = review or effects
            for condition in set(review) - facts:
                execution.effects.pop(condition, None)
            self._hold(state, "VLM requested repair")
            self._repair_graph(obs, state, node_id, tuple(p for p in required if p not in facts),
                               reason, model_facts=facts)

    @staticmethod
    def _needs_review(execution, runtime, condition, step, interval):
        confirmation = execution.effects.get(condition)
        if confirmation is None or step - confirmation.step >= 2 * interval:
            return True
        for object_id in condition.args:
            record = runtime.store.objects.get(object_id)
            if record is None or record.trusted_step is None or record.trusted_step <= confirmation.step:
                continue
            previous = confirmation.positions.get(object_id)
            if previous is None or math.dist(previous, record.last_trusted_position) > 0.03:
                return True
        return False

    def _dispatch_model_node(self, obs, state, node):
        execution = state.tasksys
        execution.active = True
        execution.started_at = state.episode_step
        execution.last_monitor_step = state.episode_step
        execution.next_monitor_step = state.episode_step + self.config.check_interval
        self._release(state)
        self._reset_detection_state(obs, state)
        self._log(state, "node_dispatched", node_id=node.id, source="vlm")

    def _repair_graph(self, obs, state, node_id, required, reason, *, model_facts):
        execution = state.tasksys
        graph = execution.graph
        runtime = self._runtime(state)
        step = state.episode_step
        if (
            execution.repair_attempts >= MAX_RECYCLES
            or (execution.last_repair_step is not None
                and step - execution.last_repair_step < self.config.check_interval)
        ):
            return
        image = self.ctx.get_vlm_image(obs)
        if image is None:
            return
        basis = {"episode_id": state.episode_id, "plan_version": graph.version, "step": step}
        continuation = graph.nodes[execution.cursor:]
        affected = graph.affected(set(required), {GOAL, *(node.id for node in continuation)})
        affected.add(node_id)
        facts = model_facts
        protected = {
            p for node in graph.nodes[:execution.cursor] for p in node.expected_effects
            if p in facts and not any(conflicts(p, target) for target in required)
        }
        request = {
            "task": state.original_instruction,
            "basis": basis, "reason": reason, "blocked_node": node_id,
            "required": [p.to_dict() for p in required],
            "affected_nodes": sorted(affected),
            "related_objects": sorted({obj for p in required for obj in p.args}),
            "protected": [p.to_dict() for p in sorted(protected, key=str)],
            "continuation": [node.to_dict() for node in continuation],
            "final_conditions": [p.to_dict() for p in graph.final_conditions],
        }
        request["confirmed_conditions"] = [p.to_dict() for p in sorted(facts, key=str)]
        request["evidence"] = runtime.evidence({obj for p in (*required, *protected) for obj in p.args},
                                               since_step=execution.started_at)
        execution.repair_attempts += 1
        execution.last_repair_step = step
        self._log(state, "repair_requested", **request)
        content = self._build_image_message(
            json.dumps(request, ensure_ascii=False), image, self.ctx.get_vlm_extra_images(obs),
        )
        payload = self._request_graph(state, "repair", _REPAIR_PROMPT, content)
        try:
            if (
                not isinstance(payload, dict)
                or json.dumps(payload.get("basis"), sort_keys=True) != json.dumps(basis, sort_keys=True)
                or state.tasksys is not execution or execution.graph is not graph
                or state.episode_id != basis["episode_id"]
                or state.episode_step != step
            ):
                raise ValueError("repair does not match the current episode, plan version and observation")
            patched, _ = graph.insert_repair(
                payload.get("repair_nodes"), execution.cursor, facts, required, protected,
            )
        except ValueError as error:
            self._log(state, "repair_rejected", reason=str(error))
            return
        self._apply_repair(state, patched)
        self._hold(state, "awaiting repair node")

    def _apply_repair(self, state, graph):
        execution = state.tasksys
        retained = execution.graph.nodes[execution.cursor:]
        self._install_graph(state, graph)
        execution.active = False
        self._log(state, "repair_applied", graph=graph.to_dict(),
                  retained_node_ids=[node.id for node in retained])

    def _request_graph(self, state, kind, prompt, content):
        started = time.monotonic()
        try:
            raw = self._vlm_call(prompt, content, max_tokens=max(3072, self.config.vlm_max_tokens))
            payload = parse_json(raw)
        except Exception as error:  # noqa: BLE001 -- transport and parse failures must hold the policy
            self._log(state, "model_error", request_kind=kind, reason=str(error),
                      latency_s=time.monotonic() - started)
            return None
        self._log(state, "model_response", request_kind=kind, latency_s=time.monotonic() - started)
        return payload

    @staticmethod
    def _hold(state, reason):
        if not state.evimem_hold:
            state.flush_actions = True
        state.evimem_hold = True
        state.evimem_hold_reason = reason

    @staticmethod
    def _release(state):
        if state.evimem_hold:
            state.flush_actions = True
        state.evimem_hold = False
        state.evimem_hold_reason = ""

    @staticmethod
    def _log(state, event, **data):
        graph = state.tasksys.graph
        state.log({"type": "tasksys_event", "event": event, "step": state.episode_step,
                   "episode_id": state.episode_id, "plan_version": graph.version if graph else None,
                   **data})
