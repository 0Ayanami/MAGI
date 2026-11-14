"""Write a paired LIBERO-Goal scorecard from full-suite run artifacts."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from summarize_logiv_smoke import read_events, read_json, summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch", type=Path)
    parser.add_argument("--quota", type=Path, help="Optional observed billing/balance snapshots")
    args = parser.parse_args()
    batch = args.batch
    quota = (
        json.loads(args.quota.read_text(encoding="utf-8-sig"))
        if args.quota else {"status": "no balance snapshots scoped to this batch"}
    )
    quota["codex_subscription"] = "not_applicable"
    runs = {}
    for arm in ("baseline", "logiv"):
        root = batch / arm
        if (root / "config.txt").is_file():
            runs[arm] = summarize(root, batch / f"{arm}-calls", quota)
            run = runs[arm]
            expected = {"suite": "libero_goal", "seed": "0", "initial_state_index": "0",
                        "trials_per_task": "1", "vlm_model": "qwen3.8-flash"}
            if any(run["config"].get(k) != v for k, v in expected.items()):
                raise ValueError(f"unexpected experiment configuration for {arm}")
            tokens = run["qwen"]["tokens"]
            i, c, o = (tokens[k]["reported_sum"] for k in
                       ("inputTokens", "cachedInputTokens", "outputTokens"))
            run["qwen"]["estimated_cny"] = (
                round(((i - c) * 0.8 + c * 0.1 + o * 2.7) / 1_000_000, 8)
                if None not in (i, c, o) else None
            )
            run["qwen"]["cost_basis"] = "Beijing list price with cache discount, excluding taxes or credits"
            run["qwen"]["cost_source"] = "https://help.aliyun.com/zh/model-studio/qwen3-8-flash"
            if arm == "logiv":
                for path in root.rglob("eval_metadata.json"):
                    task_id = read_json(path)["task_id"]
                    episode = next(row for row in run["episodes"] if row["task_id"] == task_id)
                    episode_events = list(read_events(path.parent.glob("rewrites.jsonl")))
                    episode["logiv_events"] = dict(Counter(
                        event["event"] for event in episode_events if event["type"] == "logiv_event"
                    ))
                    episode["remm_verdicts"] = dict(Counter(
                        item["truth"] for event in episode_events if event["type"] == "remm_snapshot"
                        for item in event["snapshot"]["verification"]
                    ))
            (root / "summary.json").write_text(json.dumps(run, ensure_ascii=False, indent=2) + "\n",
                                               encoding="utf-8")
    pair = [{"task_id": i} for i in range(10)]
    for arm, report in runs.items():
        completed = {}
        for row in report["episodes"]:
            task_id = row["task_id"]
            if not 0 <= task_id < 10 or task_id in completed:
                raise ValueError(f"{arm} has invalid or duplicate task ID {task_id}")
            completed[task_id] = row
            item = pair[task_id]
            if item.setdefault("task", row["task_description"]) != row["task_description"]:
                raise ValueError(f"task {task_id} differs across experiment arms")
            item.update({f"{arm}_{key}": row[key] for key in ("success", "duration_s", "num_steps")})
            item[f"{arm}_tokens"] = row["qwen_token_totals_reported"]["totalTokens"]
            item[f"{arm}_calls"] = len(row["qwen_request_ids"])
            if arm == "logiv":
                item["logiv_plan_rejected"] = row["logiv_events"].get("plan_rejected", 0)
                item["logiv_nodes_verified"] = row["logiv_events"].get("node_verified", 0)
                item["logiv_repairs_applied"] = row["logiv_events"].get("repair_applied", 0)
                item["remm_true"] = row["remm_verdicts"].get("TRUE", 0)
        report["completed_tasks"] = len(completed)
        report["successes"] = sum(row["success"] for row in completed.values())
    summary = {
        "complete": len(runs) == 2 and all(run["completed_tasks"] == 10 for run in runs.values()),
        "runs": {arm: {"completed_tasks": run["completed_tasks"], "successes": run["successes"],
                       "episodes_wall_s": sum(e["duration_s"] for e in run["episodes"]),
                       "qwen": {k: v for k, v in run["qwen"].items() if k != "requests"},
                       "memory": run["memory"], "logiv": run["logiv"]["counts"]}
                 for arm, run in runs.items()},
        "quota": quota,
    }
    columns = ["task_id", "task", *(
        f"{arm}_{k}" for arm in ("baseline", "logiv")
        for k in ("success", "duration_s", "num_steps", "tokens", "calls")
    ), "logiv_plan_rejected", "logiv_nodes_verified", "logiv_repairs_applied", "remm_true"]
    with (batch / "paired.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(pair)
    (batch / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                                        encoding="utf-8")


if __name__ == "__main__":
    main()
