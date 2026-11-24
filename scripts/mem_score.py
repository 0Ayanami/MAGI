"""Score actual LIBERO dispatches against client-only placement measurements."""

import argparse
import json
import math
from pathlib import Path

from vlm_orchestrator.strategies.episode_memory import placement


def relation_index(instruction, goals):
    relation = placement(instruction or "")
    if not relation:
        return None
    obj = {"white mug": "porcelain_mug_1", "yellow and white mug": "white_yellow_mug_1"}[relation[0]]
    support = {"left": "plate_1", "right": "plate_2"}.get(relation[1])
    return next((i for i, (_, o, plate) in enumerate(goals)
                 if o == obj and (support is None or support == plate)), None)


def score(metadata, events):
    rows = metadata["executed_steps"]
    goals = rows[0]["goals"]
    stable = {}
    for index, row in enumerate(rows):
        recent = rows[max(0, index-2):index+1]
        stable[row["step"]] = [
            len(recent) == 3 and all(r["satisfied"][j] and not r["held"][j] and
                                     math.dist(r["relative"][j], recent[0]["relative"][j]) <= .005
                                     for r in recent) for j in range(len(goals))]
    dispatches = []
    last = None
    for row in rows:
        instruction = row["instruction"]
        if instruction != last:
            dispatches.append((row["step"], instruction, relation_index(instruction, goals)))
            last = instruction
    handoffs = []
    for (start, _, j), (end, new, k) in zip(dispatches, dispatches[1:]):
        next_object = ("white_yellow_mug_1" if "yellow" in (new or "").lower()
                       else "porcelain_mug_1" if "white mug" in (new or "").lower() else None)
        if j is None or k == j or next_object is None or goals[j][1] == next_object:
            continue
        handoffs.append({"step": end, "next": new, "premature": not stable[end][j]})
    delays, missed = [], 0
    for i, (start, _, j) in enumerate(dispatches):
        if j is None:
            continue
        end = dispatches[i+1][0] if i+1 < len(dispatches) else rows[-1]["step"]
        onset = next((t for t in stable if start <= t <= end and stable[t][j]), None)
        if onset is None:
            continue
        confirmation = next((e["sim_step"] for e in events if onset <= e["sim_step"] <= end
                             and stable.get(e["sim_step"], [False]*len(goals))[j] and
                             ((e["type"] == "vlm_detect" and e.get("action") == "next")
                              or (e["type"] == "placement_check" and e.get("verdict") == "stable"))), None)
        if confirmation is None:
            missed += 1
        else:
            delays.append(confirmation-onset)
    replans = [e for e in events if e["type"] == "recycle"]
    repeated_plans = 0
    repeated_executions = 0
    for i, event in enumerate(replans):
        start = event["sim_step"]
        until = replans[i+1]["sim_step"] if i+1 < len(replans) else rows[-1]["step"]+1
        for skill in event["new_subgoals"]:
            j = relation_index(skill, goals)
            if j is not None and stable.get(start, [False]*len(goals))[j]:
                repeated_plans += 1
                repeated_executions += any(start <= r["step"] < until and r["instruction"] == skill
                                           for r in rows)
    calls = [e for e in events if e["type"] == "cloud_request"]
    usage = [e.get("tokens") for e in events if e["type"] == "cloud_response"]
    errors = sum(h["premature"] for h in handoffs)
    return dict(task=metadata["task_description"], success=metadata["success"],
                duration_s=metadata["duration_s"], executed_dispatches=dispatches, handoffs=handoffs,
                premature=errors, denominator=len(handoffs), rate=errors/len(handoffs) if handoffs else None,
                missed=missed, confirmation_delays_steps=delays, replans=len(replans),
                repeated_planned=repeated_plans, repeated_executed=repeated_executions, calls=len(calls),
                tokens=sum(usage) if len(usage)==len(calls) and None not in usage else None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("group", type=Path)
    args = parser.parse_args()
    episodes = []
    for meta in sorted(args.group.rglob("eval_metadata.json")):
        log = meta.with_name("rewrites.jsonl")
        episodes.append(score(json.loads(meta.read_text()), [json.loads(s) for s in log.read_text().splitlines()]))
    if not episodes:
        parser.error("No scored episodes in group directory")
    print(json.dumps({"episodes": episodes, "eval_wall_s": int((args.group / "eval_wall_s").read_text()),
                      "total_episode_s": sum(r["duration_s"] for r in episodes),
                      "success_s": [r["duration_s"] for r in episodes if r["success"]],
                      "failure_s": [r["duration_s"] for r in episodes if not r["success"]]}, indent=2))
