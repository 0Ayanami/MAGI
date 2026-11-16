"""Summarize paired VoLoAgent and TaskSys LIBERO runs."""

import argparse
import csv
import json
from pathlib import Path

from summarize_libero_run import summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch", type=Path)
    parser.add_argument("--quota", type=Path, help="Optional billing snapshots from this batch")
    args = parser.parse_args()
    batch = args.batch
    enhanced = "logiv" if (batch / "logiv" / "config.txt").is_file() and not (batch / "tasksys").exists() else "tasksys"
    arms = ("baseline", enhanced)
    quota = json.loads(args.quota.read_text(encoding="utf-8-sig")) if args.quota else None
    runs = {arm: summarize(batch / arm, batch / f"{arm}-calls", quota) for arm in arms}
    baseline, tasksys = (runs[arm]["config"] for arm in arms)
    for key in ("suite", "task_ids", "seed", "initial_state_index", "trials_per_task",
                "vlm_model", "monitor_interval", "terminal_observations", "terminal_action_mode"):
        if baseline.get(key) != tasksys.get(key):
            raise ValueError(f"experiment arms differ on {key}")
    expected = [int(task) for task in baseline.get("task_ids", "0 1 2 3 4 5 6 7 8 9").split()]
    paired = {task: {"task_id": task} for task in expected}
    for arm, run in runs.items():
        completed = set()
        for episode in run["episodes"]:
            task = episode["task_id"]
            if task not in paired or task in completed:
                raise ValueError(f"unexpected or duplicate task ID {task} for {arm}")
            completed.add(task)
            row = paired[task]
            if row.setdefault("task", episode["task_description"]) != episode["task_description"]:
                raise ValueError(f"task {task} differs across experiment arms")
            metrics = {
                "success": episode["success"], "num_steps": episode["num_steps"],
                "duration_s": episode["duration_s"], "terminal_success": episode.get("terminal_stability_success"),
                "calls": len(episode["qwen_request_ids"]),
                "tokens": episode["qwen_token_totals_reported"]["totalTokens"],
            }
            row.update({f"{arm}_{name}": value for name, value in metrics.items()})
        run["completed_tasks"] = len(completed)
        run["successes"] = sum(episode["success"] for episode in run["episodes"])
        run["episode_wall_s"] = sum(episode["duration_s"] for episode in run["episodes"])
        (batch / arm / "summary.json").write_text(json.dumps(run, ensure_ascii=False, indent=2) + "\n",
                                                  encoding="utf-8")

    columns = ["task_id", "task", *(
        f"{arm}_{name}" for arm in arms
        for name in ("success", "num_steps", "duration_s", "terminal_success", "calls", "tokens")
    )]
    with (batch / "paired.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(paired.values())
    summary = {
        "complete": all(run["completed_tasks"] == len(expected) and run["qwen"]["requests"]
                        for run in runs.values()),
        "task_ids": expected,
        "runs": {arm: {"completed_tasks": run["completed_tasks"], "successes": run["successes"],
                       "episode_wall_s": run["episode_wall_s"], "calls": len(run["qwen"]["requests"]),
                       "total_tokens": run["qwen"]["tokens"]["totalTokens"]["reported_sum"],
                       "bridge_total_s": run["qwen"]["bridge_total_s"], "tasksys": run["tasksys"]["counts"]}
                 for arm, run in runs.items()},
    }
    (batch / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                                        encoding="utf-8")


if __name__ == "__main__":
    main()
