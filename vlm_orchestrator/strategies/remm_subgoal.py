"""Opt-in evidence memory for the subgoal strategy.

The original strategy remains the baseline. All additional state belongs to
the current session, and VLM calls keep their original scheduling.
"""

from __future__ import annotations

from contextvars import ContextVar

from vlm_orchestrator.failure_handlers.base import ACTION_NEXT
from vlm_orchestrator.remm_memory.runtime import MemoryRuntime, RemmConfig
from vlm_orchestrator.vlm import parse_json

from .subgoal import SubgoalStrategy
from .subgoal_base import MAX_RECYCLES

_PLAN_MEMORY_PROMPT = """
Include an additional "memory" object in the existing JSON response. Identify
only task-relevant physical objects visible in the supplied images. Give each
object a stable id and a description suitable for visual grounding. For each
existing instance, reuse its ID and description from memory across replans.
For each
subgoal, describe its verifiable completion conditions using registered
predicates "on", "in", or "holding" and object ids. Use a zero-based
subgoal_index into the subgoals returned in this response. Example structure:
"memory": {
  "objects": [{"id": "red_1", "description": "red block"},
              {"id": "blue_1", "description": "blue block"}],
  "goals": [{"subgoal_index": 0,
             "predicates": [{"name": "on", "args": ["red_1", "blue_1"]}]}]
}
For a condition outside this vocabulary, omit its goal entry. Descriptions and
predicates are requests for observation, and do not certify completion.
"""


class RemmSubgoalStrategy(SubgoalStrategy):
    """Subgoal execution with independently configurable evidence memory."""

    def __init__(self, ctx, config, *, memory_config: RemmConfig, **kwargs):
        if kwargs.get("failure_monitor") in {"gt", "gt_hitl", "gt_vlm"}:
            raise ValueError(
                "REMM memory requires online observations without GT monitors"
            )
        if memory_config.gate == "enforce" and (
            kwargs.get("failure_monitor") != "vlm"
            or config.check_mode != "vlm"
            or memory_config.env_mode != "libero"
            or kwargs.get("recovery_mode", "replan") != "replan"
        ):
            raise ValueError(
                "Enforced memory requires LIBERO, VLM checks and replan recovery"
            )
        super().__init__(ctx, config, **kwargs)
        self.memory_config = memory_config
        self._active_state = ContextVar("remm_session", default=None)
        self._planning = ContextVar("remm_planning", default=False)
        self._plan_memory = ContextVar("remm_plan_memory", default=None)
        self._approved_index = ContextVar("remm_approved_index", default=None)

    def _runtime(self, state):
        if not hasattr(state, "remm_memory"):
            state.remm_memory = MemoryRuntime(self.memory_config)
            state.remm_blocked_counts = {}
            state.remm_repair_attempted = set()
            state.remm_repair_failures = {}
        return state.remm_memory

    def process(self, obs, state):
        token = self._active_state.set(state)
        try:
            return super().process(obs, state)
        finally:
            self._active_state.reset(token)

    def _is_new_episode(self, obs, state):
        current = self.ctx.get_prompt(obs)
        last_episode = getattr(state, "remm_last_episode_id", 0)
        prompt_changed = current not in (
            state.original_instruction,
            state.rewritten_instruction,
        )
        is_new = (
            state.infer_count == 0
            or prompt_changed
            or (state.episode_id > 0 and state.episode_id != last_episode)
        )
        if is_new:
            if state.episode_id == 0:
                state.episode_id = 1
            elif state.infer_count > 0 and state.episode_id == last_episode:
                state.episode_id += 1
            state.remm_last_episode_id = state.episode_id
        return is_new

    def _on_new_episode(self, obs, state, prompt):
        self._runtime(state).reset(state.episode_id)
        state.remm_blocked_counts = {}
        state.remm_repair_attempted = set()
        state.remm_repair_failures = {}
        state.remm_hold = False
        state.remm_hold_reason = ""
        state.remm_gripper_command = None
        state.remm_hold_env = self.memory_config.env_mode
        result = super()._on_new_episode(obs, state, prompt)
        self._observe(obs, state)
        return result

    def _observe(self, obs, state):
        runtime = self._runtime(state)
        try:
            runtime.observe(obs, state.episode_step)
        except ValueError as error:
            runtime.invalidate_current(
                max(getattr(runtime, "step", 0), state.episode_step, 0),
                str(error),
            )
            state.log(
                {
                    "type": "remm_event",
                    "event": "observation_rejected",
                    "step": state.episode_step,
                    "reason": str(error),
                }
            )
        state.log(
            {
                "type": "remm_snapshot",
                "step": state.episode_step,
                "memory_mode": self.memory_config.mode,
                "memory_gate": self.memory_config.gate,
                "snapshot": runtime.snapshot(),
            }
        )

    def _on_step(self, obs, state):
        self._observe(obs, state)
        runtime = self._runtime(state)
        broken = set(runtime.broken_goals)
        unreported = broken - state.remm_repair_attempted
        if getattr(state, "remm_hold", False) and not broken:
            restored = state.remm_hold_reason == "completed conditions invalidated"
            verified = runtime.check_completion(
                state.current_subgoal_idx,
                state.episode_step,
            )["allowed"]
            if restored or verified:
                state.remm_hold = False
                state.remm_hold_reason = ""
                state.flush_actions = True
        if unreported and self.memory_config.gate == "enforce":
            state.remm_repair_attempted.update(unreported)
            if self._repair(obs, state, "completed conditions invalidated"):
                return self.ctx.set_prompt(obs, state.rewritten_instruction), state
        if getattr(state, "remm_hold", False):
            signature = self._repair_signature(state)
            failures = state.remm_repair_failures.get(signature, 0)
            if failures < self.memory_config.max_reobservations + 1 and (
                state.episode_step - state.remm_step_at_repair
                >= self.config.check_interval
            ):
                if self._repair(obs, state, state.remm_hold_reason):
                    obs = self.ctx.set_prompt(obs, state.rewritten_instruction)
            return obs, state
        return super()._on_step(obs, state)

    def _vlm_call(self, system_prompt, user_content, max_tokens=None):
        state = self._active_state.get()
        content = list(user_content)
        if state is not None:
            content.append({"type": "text", "text": self._runtime(state).context()})
        if self._planning.get():
            system_prompt += "\n" + _PLAN_MEMORY_PROMPT
            max_tokens = max(2048, max_tokens or self.config.vlm_max_tokens)
        raw = super()._vlm_call(system_prompt, content, max_tokens=max_tokens)
        if self._planning.get():
            try:
                payload = parse_json(raw)
                self._plan_memory.set(payload.get("memory", {}))
            except (ValueError, AttributeError):
                self._plan_memory.set({})
        return raw

    def _decompose_and_setup(self, obs, state, instruction, image):
        stage = self._planning.set(True)
        payload = self._plan_memory.set(None)
        try:
            super()._decompose_and_setup(obs, state, instruction, image)
            self._set_memory_plan(state, self._plan_memory.get() or {})
        finally:
            self._planning.reset(stage)
            self._plan_memory.reset(payload)

    def _set_memory_plan(self, state, payload):
        self._runtime(state).set_plan(payload, state.subgoals)
        state.remm_blocked_counts = {}
        if getattr(state, "remm_hold", False):
            state.flush_actions = True
        state.remm_hold = False
        state.remm_hold_reason = ""
        state.log(
            {
                "type": "remm_event",
                "event": "plan_registered",
                "step": state.episode_step,
                "snapshot": self._runtime(state).snapshot(),
            }
        )

    def _recycle(self, obs, state, current_image, regression_context=None):
        token = self._planning.set(True)
        try:
            return super()._recycle(obs, state, current_image, regression_context)
        finally:
            self._planning.reset(token)

    def _apply_recycle_subgoals(self, obs, state, data):
        applied = super()._apply_recycle_subgoals(obs, state, data)
        if applied:
            self._set_memory_plan(state, data.get("memory", {}))
        return applied

    def _apply_new_subgoals(self, obs, state, subgoals):
        super()._apply_new_subgoals(obs, state, subgoals)
        self._set_memory_plan(state, {})

    def _repair(self, obs, state, reason):
        state.remm_hold = True
        state.remm_hold_reason = reason
        state.flush_actions = True
        signature = self._repair_signature(state)
        failures = state.remm_repair_failures.get(signature, 0)
        state.remm_step_at_repair = state.episode_step
        image = self.ctx.get_vlm_image(obs)
        available = (
            image is not None
            and self._recycle_count < MAX_RECYCLES
            and failures < self.memory_config.max_reobservations + 1
        )
        state.log(
            {
                "type": "remm_event",
                "event": "repair_requested",
                "step": state.episode_step,
                "reason": reason,
                "available": available,
            }
        )
        repaired = bool(available and self._recycle(obs, state, image))
        if not repaired:
            state.remm_repair_failures[signature] = failures + 1
        return repaired

    def _repair_signature(self, state):
        runtime = self._runtime(state)
        broken = tuple(sorted(runtime.broken_goals))
        if broken:
            return broken
        return (
            f"plan:{getattr(runtime, 'plan_version', 0)}:goal:{state.current_subgoal_idx}",
        )

    def _allow_completion(self, obs, state):
        runtime = self._runtime(state)
        index = state.current_subgoal_idx
        result = runtime.check_completion(index, state.episode_step)
        if runtime.broken_goals and index == len(state.subgoals) - 1:
            result = {
                **result,
                "allowed": False,
                "reason": "previous completion conditions invalidated",
            }
        allowed = bool(result["allowed"])
        state.log(
            {
                "type": "remm_event",
                "event": "completion_check",
                "step": state.episode_step,
                "subgoal_idx": index,
                "memory_gate": self.memory_config.gate,
                "decision": result,
            }
        )
        if allowed:
            runtime.mark_complete(index)
            state.remm_blocked_counts.pop(index, None)
            state.log(
                {
                    "type": "remm_snapshot",
                    "step": state.episode_step,
                    "memory_mode": self.memory_config.mode,
                    "memory_gate": self.memory_config.gate,
                    "snapshot": runtime.snapshot(),
                }
            )
        if self.memory_config.gate == "shadow" or allowed:
            return True
        attempts = state.remm_blocked_counts.get(index, 0) + 1
        state.remm_blocked_counts[index] = attempts
        if attempts == self.memory_config.max_reobservations + 1:
            self._repair(obs, state, "completion evidence budget exhausted")
        return False

    def _execute_handler_result(self, obs, state, result):
        if result.action != ACTION_NEXT:
            return super()._execute_handler_result(obs, state, result)
        if not self._allow_completion(obs, state):
            return obs, state
        token = self._approved_index.set(state.current_subgoal_idx)
        try:
            return super()._execute_handler_result(obs, state, result)
        finally:
            self._approved_index.reset(token)

    def _advance_subgoal(self, obs, state):
        if self._approved_index.get() != state.current_subgoal_idx:
            if not self._allow_completion(obs, state):
                return
        super()._advance_subgoal(obs, state)
