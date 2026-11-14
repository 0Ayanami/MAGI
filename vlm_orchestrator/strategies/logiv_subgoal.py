"""Execute optional LOGIV task graphs over REMM and the LIBERO VLA."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass

from vlm_orchestrator.failure_handlers.base import ACTION_REPLAN
from vlm_orchestrator.logiv.graph import GOAL, TaskGraph, conflicts
from vlm_orchestrator.remm_memory.types import Predicate
from vlm_orchestrator.vlm import parse_json

from .remm_subgoal import RemmSubgoalStrategy
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


@dataclass
class PendingRepair:
    graph: TaskGraph
    repair_count: int
    proposed_at: int
    required: tuple[Predicate, ...]
    protected: set[Predicate]


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
    pending_repair: PendingRepair | None = None
    fault: str | None = None
    blocked_key: tuple | None = None
    blocked_step: int = -1
    blocked_observations: int = 0
    final_verified: bool = False


class LogivSubgoalStrategy(RemmSubgoalStrategy):
    """Select verified graph steps and request bounded local repair fragments."""

    def __init__(self, ctx, config, *, memory_config, **kwargs):
        if (
            memory_config.mode != "episodic" or memory_config.gate != "enforce"
            or kwargs.get("hitl_state") is not None
            or kwargs.get("initial_strategy") is not None
        ):
            raise ValueError("LOGIV requires episodic/enforce REMM and autonomous graph planning")
        super().__init__(ctx, config, memory_config=memory_config, **kwargs)

    def _on_new_episode(self, obs, state, prompt):
        state.logiv = GraphExecution()
        obs, state = super()._on_new_episode(obs, state, prompt)
        return self._tick_graph(obs, state)

    def _decompose_and_setup(self, obs, state, instruction, image):
        execution = state.logiv
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
        state.logiv.graph = graph
        self._runtime(state).watched_predicates = graph.predicates
        state.subgoals = [node.instruction for node in graph.nodes]
        state.subgoals_ordered = True
        self._select_instruction(state)

    def _select_instruction(self, state):
        state.current_subgoal_idx = min(state.logiv.cursor, len(state.subgoals) - 1)
        state.rewritten_instruction = state.subgoals[state.current_subgoal_idx]
        state.flush_actions = True

    def _on_step(self, obs, state):
        execution = state.logiv
        if execution.graph is None and execution.planning_attempts < MAX_RECYCLES:
            last = execution.last_planning_step
            if last is None or state.episode_step - last >= self.config.check_interval:
                image = self.ctx.get_vlm_image(obs)
                if image is not None:
                    self._decompose_and_setup(obs, state, state.original_instruction, image)
        self._observe(obs, state)
        return self._tick_graph(obs, state)

    def _tick_graph(self, obs, state):
        execution = state.logiv
        step = state.episode_step
        if execution.graph is None:
            self._hold(state, "awaiting graph")
            return obs, state

        runtime = self._runtime(state)
        if runtime.step != step or runtime.observation_error:
            self._hold(state, "awaiting valid REMM observation")
            return obs, state
        if execution.pending_repair is not None:
            if step <= execution.pending_repair.proposed_at:
                return obs, state
            self._resolve_repair(state)
        graph = execution.graph
        while execution.cursor < len(graph.nodes):
            node = graph.nodes[execution.cursor]
            effects = self._conditions(runtime, node.expected_effects, step)
            if self._verified(effects):
                receipt = runtime.record_completion(
                    f"logiv:{node.id}", node.expected_effects, step,
                )
                self._log(state, "node_verified", node_id=node.id,
                          dispatched=execution.active, assertion=receipt.to_dict(), effects=effects)
                execution.cursor += 1
                execution.active = False
                execution.fault = None
                self._select_instruction(state)
                continue

            # A placement can consume its own holding precondition during execution.
            preconditions = tuple(
                p for p in node.preconditions
                if not execution.active or not any(conflicts(p, e) for e in node.expected_effects)
            )
            checks = self._conditions(runtime, preconditions, step)
            if not self._verified(checks):
                execution.active = False
                self._blocked(obs, state, node.id, preconditions, checks, "preconditions")
            elif execution.fault:
                self._blocked(obs, state, node.id, node.expected_effects, effects, execution.fault)
            elif not execution.active:
                execution.active = True
                execution.started_at = step
                execution.blocked_key = None
                self._release(state)
                self._reset_detection_state(obs, state)
                self._log(state, "node_dispatched", node_id=node.id, preconditions=checks)
            else:
                self._monitor_node(obs, state, node, effects)
            return self.ctx.set_prompt(obs, state.rewritten_instruction), state

        checks = self._conditions(runtime, graph.final_conditions, step)
        verified = self._verified(checks)
        if verified != execution.final_verified:
            execution.final_verified = verified
            self._log(state, "final_verification", verified=verified, conditions=checks)
        if verified:
            execution.blocked_key = None
            self._hold(state, "final conditions verified")
        else:
            self._blocked(obs, state, GOAL, graph.final_conditions, checks, "final conditions")
        return self.ctx.set_prompt(obs, state.rewritten_instruction), state

    def _monitor_node(self, obs, state, node, effects):
        execution = state.logiv
        if state.episode_step - execution.started_at >= self.config.subgoal_timeout:
            execution.fault = "subgoal timeout"
        elif self._failure_handler is not None:
            decision = self._failure_handler.step(obs, state)
            if decision is not None:
                self._log(state, "monitor", node_id=node.id, action=decision.action,
                          reason=decision.reason, latency_s=decision.extra.get("vlm_latency_s"))
                if decision.action == ACTION_REPLAN:
                    execution.fault = "monitor requested repair"
        if execution.fault:
            self._blocked(obs, state, node.id, node.expected_effects, effects, execution.fault)

    def _blocked(self, obs, state, node_id, predicates, checks, reason):
        execution = state.logiv
        required = tuple(p for p, row in zip(predicates, checks) if not self._verified([row]))
        step = state.episode_step
        key = (node_id, required, reason)
        if key != execution.blocked_key:
            execution.blocked_key = key
            execution.blocked_observations = 0
            execution.blocked_step = -1
        if step > execution.blocked_step:
            execution.blocked_step = step
            execution.blocked_observations += 1
        self._hold(state, reason)
        self._log(state, "node_blocked", node_id=node_id, reason=reason, conditions=checks)
        if (
            execution.fault or any(row["truth"] == "FALSE" for row in checks)
            or execution.blocked_observations > self.memory_config.max_reobservations
        ):
            self._repair_graph(obs, state, node_id, required, reason)

    def _repair_graph(self, obs, state, node_id, required, reason):
        execution = state.logiv
        graph = execution.graph
        runtime = self._runtime(state)
        step = state.episode_step
        if (
            execution.repair_attempts >= MAX_RECYCLES
            or (execution.last_repair_step is not None
                and step - execution.last_repair_step < self.config.check_interval)
            or runtime.step != step or runtime.observation_error
        ):
            return
        image = self.ctx.get_vlm_image(obs)
        if image is None:
            return
        basis = {"episode_id": state.episode_id, "plan_version": graph.version, "step": step}
        continuation = graph.nodes[execution.cursor:]
        affected = graph.affected(set(required), {GOAL, *(node.id for node in continuation)})
        affected.add(node_id)
        facts = self._verified_facts(runtime, graph.predicates, step)
        protected = {
            p for node in graph.nodes[:execution.cursor] for p in node.expected_effects
            if p in facts and not any(conflicts(p, target) for target in required)
        }
        request = {
            "task": state.original_instruction,
            "basis": basis, "reason": reason, "blocked_node": node_id,
            "required": self._conditions(runtime, required, step),
            "affected_nodes": sorted(affected),
            "related_objects": sorted({obj for p in required for obj in p.args}),
            "invalidated_completions": list(runtime.broken_goals),
            "protected": [p.to_dict() for p in sorted(protected, key=str)],
            "continuation": [node.to_dict() for node in continuation],
            "final_conditions": [p.to_dict() for p in graph.final_conditions],
        }
        execution.repair_attempts += 1
        execution.last_repair_step = step
        self._log(state, "repair_requested", **request)
        content = self._build_image_message(json.dumps(request, ensure_ascii=False), image, None)
        payload = self._request_graph(state, "repair", _REPAIR_PROMPT, content)
        try:
            if (
                not isinstance(payload, dict)
                or json.dumps(payload.get("basis"), sort_keys=True) != json.dumps(basis, sort_keys=True)
                or state.logiv is not execution or execution.graph is not graph
                or state.episode_id != basis["episode_id"]
                or state.episode_step != step or runtime.step != step or runtime.observation_error
            ):
                raise ValueError("repair does not match the current episode, plan version and observation")
            patched, repair_count = graph.insert_repair(
                payload.get("repair_nodes"), execution.cursor, facts, required, protected,
            )
        except ValueError as error:
            self._log(state, "repair_rejected", reason=str(error))
            return
        execution.pending_repair = PendingRepair(patched, repair_count, step, required, protected)
        runtime.watched_predicates = patched.predicates
        self._log(state, "repair_staged", proposed_version=patched.version)

    def _resolve_repair(self, state):
        execution = state.logiv
        proposal = execution.pending_repair
        runtime = self._runtime(state)
        execution.pending_repair = None
        runtime.watched_predicates = execution.graph.predicates
        facts = self._verified_facts(runtime, proposal.graph.predicates, state.episode_step)
        if set(proposal.required) <= facts:
            self._log(state, "repair_cancelled", reason="required conditions recovered")
            return
        try:
            proposal.graph.validate_repair(
                execution.cursor, proposal.repair_count, facts, proposal.required, proposal.protected,
            )
        except ValueError as error:
            self._log(state, "repair_rejected", reason=str(error), phase="fresh_observation")
            return
        retained = execution.graph.nodes[execution.cursor:]
        self._install_graph(state, proposal.graph)
        execution.active = False
        execution.fault = None
        execution.blocked_key = None
        self._log(state, "repair_applied", graph=proposal.graph.to_dict(),
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
    def _conditions(runtime, predicates, step):
        return [{"predicate": p.to_dict(), **runtime.condition(p, step)} for p in predicates]

    @staticmethod
    def _verified(rows):
        return all(row["truth"] == "TRUE" and row["assertion_id"] for row in rows)

    def _verified_facts(self, runtime, predicates, step):
        return {p for p in predicates if self._verified([runtime.condition(p, step)])}

    @staticmethod
    def _hold(state, reason):
        if not state.remm_hold:
            state.flush_actions = True
        state.remm_hold = True
        state.remm_hold_reason = reason

    @staticmethod
    def _release(state):
        if state.remm_hold:
            state.flush_actions = True
        state.remm_hold = False
        state.remm_hold_reason = ""

    @staticmethod
    def _log(state, event, **data):
        graph = state.logiv.graph
        state.log({"type": "logiv_event", "event": event, "step": state.episode_step,
                   "episode_id": state.episode_id, "plan_version": graph.version if graph else None,
                   **data})
