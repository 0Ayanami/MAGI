"""Summarize one LIBERO run and its dedicated Codex bridge call directory."""

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import median


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_events(paths):
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                yield json.loads(line)


def summarize(run, calls, quota):
    usage_fields = (
        "inputTokens", "cachedInputTokens", "cacheWriteInputTokens",
        "outputTokens", "reasoningOutputTokens", "totalTokens",
    )
    records = []
    for directory in sorted(calls.glob("chatcmpl-*")):
        path = directory / "result.json"
        result = read_json(path) if path.is_file() else {"error": "no recorded result"}
        usage = (result.get("usage") or {}).get("total") or {}
        usage_source = "result" if usage else None
        rpc_log = directory / "app-server-events.jsonl"
        if not usage and rpc_log.is_file():
            for event in read_events([rpc_log]):
                if event.get("method") == "thread/tokenUsage/updated":
                    usage = event.get("params", {}).get("tokenUsage", {}).get("total") or usage
            if usage:
                usage_source = "app_server_events"
        decision = json.loads(result.get("text") or "{}")
        kind = (
            "repair" if "repair_nodes" in decision
            else "monitor" if "action" in decision
            else "replan" if "remaining" in decision or "done" in decision
            else "plan" if "nodes" in decision or "subgoals" in decision
            else "unclassified"
        )
        request_path = directory / "request.json"
        records.append({
            "id": directory.name, "kind": kind, "model": result.get("model"),
            "provider": result.get("model_provider"), "effort": result.get("effort"),
            "request_recorded_at": request_path.stat().st_mtime if request_path.is_file() else None,
            "error": result.get("error"), "usage": usage, "usage_source": usage_source,
            **{k: result.get(k) for k in (
                "latency_s", "turn_latency_s", "queue_wait_s", "service_latency_s",
                "thread_id", "turn_id",
            )},
        })
    totals = {}
    for field in usage_fields:
        values = [row["usage"][field] for row in records if row["usage"].get(field) is not None]
        totals[field] = {"reported_sum": sum(values) if values else None, "calls_reported": len(values)}
    latencies = [r["latency_s"] for r in records if r["latency_s"] is not None]
    events = list(read_events(run.rglob("rewrites.jsonl")))
    graph = [event for event in events if event.get("type") == "logiv_event"]
    memory = [event for event in events if event.get("type") == "remm_snapshot"]
    observations = [o for row in memory for o in row["snapshot"]["observation_events"]]
    episodes = sorted(
        (read_json(path) for path in run.rglob("eval_metadata.json")),
        key=lambda row: (row["task_id"], row["episode_idx"]),
    )
    for episode in episodes:
        end = episode["end_time"] + (episode.get("terminal_verification") or {}).get("duration_s", 0)
        episode_calls = [
            row for row in records if row["request_recorded_at"] is not None
            and episode["start_time"] <= row["request_recorded_at"] <= end
        ]
        episode["qwen_request_ids"] = [row["id"] for row in episode_calls]
        episode["qwen_token_totals_reported"] = {
            field: sum(values) if (values := [
                row["usage"][field] for row in episode_calls if row["usage"].get(field) is not None
            ]) else None
            for field in usage_fields
        }
    perception = list(read_events(run.rglob("requests.jsonl")))
    return {
        "config": dict(
            line.split("=", 1) for line in (run / "config.txt").read_text().splitlines()
            if "=" in line
        ),
        "episodes": episodes,
        "qwen": {
            "requests": records, "by_kind": dict(Counter(r["kind"] for r in records)),
            "failed": sum(bool(r["error"]) for r in records), "tokens": totals,
            "bridge_total_s": sum(latencies) if latencies else None,
            "bridge_median_s": median(latencies) if latencies else None,
        },
        "memory": {
            "snapshots": len(memory),
            "observation_errors": sum(bool(row["snapshot"]["observation_error"]) for row in memory),
            "visible_object_observations": sum(o["payload"]["visible"] for o in observations),
            "verification_counts": dict(Counter(
                item["truth"] for row in memory for item in row["snapshot"]["verification"]
            )),
        },
        "logiv": {
            "counts": dict(Counter(event["event"] for event in graph)), "events": graph,
        },
        "perception": {
            "requests": dict(Counter(event["endpoint"] for event in perception)),
            "total_s": sum(event.get("duration_s", 0) for event in perception) if perception else None,
        },
        "quota": quota,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("calls", type=Path)
    parser.add_argument("--quota", type=Path, help="Optional provider console snapshots JSON")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    quota = read_json(args.quota) if args.quota else {
        "codex_subscription": "not_applicable",
        "bailian_account": "unavailable",
        "remaining_tokens_before": None, "remaining_tokens_after": None, "delta_tokens": None,
        "source": "Bailian console free-quota or billing ledger",
    }
    result = summarize(args.run, args.calls, quota)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
