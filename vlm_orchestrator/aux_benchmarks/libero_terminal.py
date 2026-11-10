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
) -> dict:
    """Query fresh post-success observations while executing measured holds.

    The caller captures the primary episode result before this audit. Policy
    responses are recorded as metadata and their actions are discarded. Only
    zero end-effector displacement with the initial measured gripper command
    advances the simulation between audit samples.
    """
    if type(count) is not int or count < 0:
        raise ValueError("terminal observation count must be a nonnegative integer")
    if type(step_interval) is not int or step_interval < 1:
        raise ValueError("terminal observation interval must be a positive integer")
    if type(start_step) is not int or start_step < 0:
        raise ValueError("terminal start step must be a nonnegative integer")
    if count == 0:
        return {}

    started = time.monotonic()
    result = {
        "is_post_success_audit": True,
        "requested_observations": count,
        "step_interval": step_interval,
        "start_step": start_step,
        "held_steps": 0,
        "observations": [],
        "stability_success": True,
        "all_held_steps_success": True,
    }
    step = start_step
    success = True
    last_response = {}
    try:
        wire = build_observation(observation)
        gripper = np.asarray(
            wire.get("observation/gripper_position", []),
            dtype=float,
        ).reshape(-1)
        if gripper.size != 1 or not np.isfinite(gripper[0]) or not 0 <= gripper[0] <= 1:
            raise ValueError(
                "terminal hold requires a valid normalized gripper observation"
            )
        hold = [0.0] * 6 + [1.0 if gripper[0] > 0.5 else -1.0]
        result["gripper_command"] = hold[-1]
        for index in range(count):
            if index:
                for offset in range(step_interval):
                    observation, _, success, _ = env.step(hold)
                    success = bool(success)
                    step += 1
                    result["held_steps"] += 1
                    result["stability_success"] = success
                    result["all_held_steps_success"] &= success
                    wire = build_observation(observation)
                    if record_frame is not None and offset < step_interval - 1:
                        record_frame(wire, step, last_response)
            wire["__step"] = step
            wire["__terminal_observation"] = True
            # Success sampling is known to the client. The model receives only
            # its ordinary online inputs through the proxy's existing filters.
            wire.pop("ground_truth_done", None)
            response = client.infer(wire)
            if not isinstance(response, dict):
                raise TypeError("terminal observation response must be a mapping")
            last_response = response
            result["observations"].append(
                {
                    "index": index,
                    "step": step,
                    "environment_success": bool(success),
                    "policy_actions_executed": 0,
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
