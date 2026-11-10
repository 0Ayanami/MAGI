"""Opt-in post-success observations for LIBERO execution audits."""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np


def capture_terminal_observations(
    env,
    client,
    observation: dict,
    *,
    start_step: int,
    count: int,
    step_interval: int,
    build_observation: Callable[[dict], dict],
    record_frame: Callable[[dict, int, dict], None] | None = None,
    action_mode: str = "hold",
    policy_replan_steps: int = 5,
) -> dict:
    """Query fresh post-success observations during a bounded execution audit.

    The caller captures the primary episode result before this audit. The
    default holds the measured gripper and discards returned policy actions.
    Policy mode consumes returned actions with the ordinary client replan
    cadence, so release/retraction can occur without a forced gripper command.
    """
    if type(count) is not int or count < 0:
        raise ValueError("terminal observation count must be a nonnegative integer")
    if type(step_interval) is not int or step_interval < 1:
        raise ValueError("terminal observation interval must be a positive integer")
    if type(start_step) is not int or start_step < 0:
        raise ValueError("terminal start step must be a nonnegative integer")
    if action_mode not in {"hold", "policy"}:
        raise ValueError("terminal action mode must be hold or policy")
    if type(policy_replan_steps) is not int or policy_replan_steps < 1:
        raise ValueError("terminal policy replan steps must be a positive integer")
    if count == 0:
        return {}

    started = time.monotonic()
    result = {
        "is_post_success_audit": True,
        "action_mode": action_mode,
        "policy_replan_steps": policy_replan_steps,
        "requested_observations": count,
        "step_interval": step_interval,
        "start_step": start_step,
        "held_steps": 0,
        "policy_actions_executed": 0,
        "memory_hold_steps": 0,
        "executed_steps": 0,
        "query_count": 0,
        "queries": [],
        "observations": [],
        "stability_success": True,
        "all_held_steps_success": True,
        "all_executed_steps_success": True,
    }
    step = start_step
    success = True
    last_response = {}
    actions = []

    def query(kind):
        nonlocal last_response, actions
        wire["__step"] = step
        wire["__terminal_observation"] = True
        # Success remains passive scoring; it is never a policy input.
        wire.pop("ground_truth_done", None)
        request = {"step": step, "kind": kind, "response_received": False}
        result["queries"].append(request)
        result["query_count"] += 1
        response = client.infer(wire)
        if not isinstance(response, dict):
            raise TypeError("terminal observation response must be a mapping")
        request["response_received"] = True
        last_response = response
        if action_mode == "policy":
            chunk = np.asarray(response.get("actions"), dtype=float)
            if (
                chunk.ndim != 2
                or chunk.shape[0] == 0
                or chunk.shape[1] != 7
                or not np.isfinite(chunk).all()
            ):
                raise ValueError(
                    "terminal policy actions must be a finite nonempty Nx7 array"
                )
            actions = list(chunk[:policy_replan_steps].copy())
            request["action_source"] = (
                "memory_hold" if response.get("orchestrator_memory_hold") else "policy"
            )
        return response

    try:
        wire = build_observation(observation)
        if action_mode == "hold":
            gripper = np.asarray(
                wire.get("observation/gripper_position", []),
                dtype=float,
            ).reshape(-1)
            if (
                gripper.size != 1
                or not np.isfinite(gripper[0])
                or not 0 <= gripper[0] <= 1
            ):
                raise ValueError(
                    "terminal hold requires a valid normalized gripper observation"
                )
            hold = [0.0] * 6 + [1.0 if gripper[0] > 0.5 else -1.0]
            result["gripper_command"] = hold[-1]
        for index in range(count):
            if index:
                for offset in range(step_interval):
                    if action_mode == "policy":
                        if not actions:
                            query("refill")
                        action = actions.pop(0)
                        is_hold = bool(last_response.get("orchestrator_memory_hold"))
                    else:
                        action, is_hold = hold, True
                    observation, _, success, _ = env.step(action)
                    success = bool(success)
                    step += 1
                    result["executed_steps"] += 1
                    result["held_steps" if is_hold else "policy_actions_executed"] += 1
                    if action_mode == "policy" and is_hold:
                        result["memory_hold_steps"] += 1
                    result["stability_success"] = success
                    result["all_executed_steps_success"] &= success
                    if is_hold:
                        result["all_held_steps_success"] &= success
                    wire = build_observation(observation)
                    if record_frame is not None and offset < step_interval - 1:
                        record_frame(wire, step, last_response)
            response = query("sample")
            result["observations"].append(
                {
                    "index": index,
                    "step": step,
                    "environment_success": bool(success),
                    "policy_actions_executed": result["policy_actions_executed"],
                    "held_steps": result["held_steps"],
                    "orchestrator": {
                        key: value
                        for key, value in response.items()
                        if key
                        in {
                            "orchestrator_instruction",
                            "orchestrator_subgoal_idx",
                            "orchestrator_subgoals",
                            "orchestrator_flush_actions",
                            "orchestrator_failure",
                            "orchestrator_memory_hold",
                        }
                    },
                }
            )
            if record_frame is not None:
                record_frame(wire, step, response)
    except Exception as error:  # noqa: BLE001 -- preserve the captured primary result
        result["error"] = f"{type(error).__name__}: {error}"
    result["complete"] = len(result["observations"]) == count and "error" not in result
    result["end_step"] = step
    result["duration_s"] = time.monotonic() - started
    return result
