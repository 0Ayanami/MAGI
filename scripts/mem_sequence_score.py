"""Aggregate paired LIBERO-Mem results using the benchmark's sequence tracker."""

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
from statistics import mean


def read_cases(root):
    for marker in sorted(root.glob("*/*/*/complete")):
        folder = marker.parent
        arm = folder.parent.parent.name
        [metadata_path] = folder.rglob("eval_metadata.json")
        metadata = json.loads(metadata_path.read_text())
        events = [json.loads(line) for line in
                  metadata_path.with_name("rewrites.jsonl").read_text().splitlines()]
        checks = [event for event in events if event["type"] == "placement_check"]
        verdicts = Counter(event["verdict"] for event in checks)
        responses = [event.get("tokens") for event in events if event["type"] == "cloud_response"]
        calls = sum(event["type"] == "cloud_request" for event in events)
        dispatches = []
        completions = []
        previous = None
        count = metadata.get("completed_before_execution", 0)
        for row in metadata["executed_steps"]:
            dispatch = row["subgoal_idx"], row["instruction"]
            if dispatch != previous:
                dispatches.append((row["step"], *dispatch))
                previous = dispatch
            if row["completed_after"] != count:
                count = row["completed_after"]
                completions.append((row["step"], count))
        yield {
            "arm": arm, "task_id": metadata["task_id"], "language": metadata["task_language"],
            "init": metadata["init_state_idx"], "success": metadata["success"],
            "completed": metadata["n_subgoals_completed"], "available": metadata["n_subgoals_total"],
            "scr": metadata["tiered_success"], "episode_s": round(metadata["duration_s"], 3),
            "eval_wall_s": int((folder / "eval_wall_s").read_text()),
            "cloud_calls": calls, "cloud_responses": len(responses),
            "tokens": sum(responses) if len(responses) == calls and None not in responses else None,
            "replans": sum(e["type"] == "recycle" for e in events),
            "vlm_replan_requests": sum(e["type"] == "vlm_detect" and e.get("action") == "replan" for e in events),
            "visual_carrying": verdicts["carrying"], "visual_stable": verdicts["stable"],
            "visual_unknown": verdicts["unknown"],
            "visual_detection_s": round(sum(e.get("detection_s", 0) for e in checks), 3),
            "executed_dispatches": len(dispatches),
            "dispatch_trace": json.dumps(dispatches, ensure_ascii=False),
            "completion_trace": json.dumps(completions),
            "raw_dir": metadata_path.parent.relative_to(root).as_posix(),
        }


def aggregate(rows):
    groups = {}
    for row in rows:
        key = (row["task_id"], row["init"])
        arms = groups.setdefault(key, {})
        assert row["arm"] not in arms, (key, row["arm"])
        arms[row["arm"]] = row
    pairs = [p for p in groups.values() if set(p) == {"baseline", "dual"}]
    report = {"runs": len(rows), "pairs": len(pairs), "paired_outcomes": [
        {key: p["baseline"][key] for key in ("task_id", "init")}
        | {arm: {field: p[arm][field] for field in ("success", "scr", "episode_s")} for arm in p}
        for p in pairs
    ]}
    for arm in ("baseline", "dual"):
        records = [p[arm] for p in pairs]
        totals = {key: sum(r[key] for r in records)
                  for key in ("success", "completed", "available", "cloud_calls", "cloud_responses",
                              "eval_wall_s", "replans", "vlm_replan_requests", "visual_carrying",
                              "visual_stable", "visual_unknown", "visual_detection_s", "executed_dispatches")}
        totals["mean_scr"] = mean(r["scr"] for r in records) if records else None
        totals["tokens"] = (sum(r["tokens"] for r in records)
                            if all(r["tokens"] is not None for r in records) else None)
        totals["success_s"] = [r["episode_s"] for r in records if r["success"]]
        totals["failure_s"] = [r["episode_s"] for r in records if not r["success"]]
        report[arm] = totals
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    rows = list(read_cases(args.directory))
    if not rows:
        parser.error("No completed episodes in this directory")
    with (args.directory / "episodes.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    report = json.dumps(aggregate(rows), indent=2)
    (args.directory / "summary.json").write_text(report + "\n")
    print(report)


if __name__ == "__main__":
    main()
