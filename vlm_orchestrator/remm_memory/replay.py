"""CPU-only replay of online perception observations and completion queries."""

import argparse
import json
from pathlib import Path

from .runtime import MemoryRuntime, RemmConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("current", "episodic"), default="episodic")
    args = parser.parse_args()
    if args.output.resolve() in {args.input.resolve(), args.plan.resolve()}:
        parser.error("output must differ from input and plan")
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    runtime = MemoryRuntime(RemmConfig(mode=args.mode))
    runtime.reset(plan.get("episode_id", 1))
    runtime.set_plan(plan.get("memory"), plan["subgoals"])
    if runtime.plan_error:
        parser.error(runtime.plan_error)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with (
        args.input.open(encoding="utf-8") as observations,
        args.output.open("w", encoding="utf-8") as output,
    ):
        for line in observations:
            row = json.loads(line)
            runtime.observe(row.get("observation", {}), row["step"])
            decision = None
            if "check_subgoal" in row:
                index = row["check_subgoal"]
                decision = runtime.check_completion(index, row["step"])
                if row.get("commit_completion") and decision["allowed"]:
                    runtime.mark_complete(index)
            output.write(
                json.dumps({"snapshot": runtime.snapshot(), "decision": decision})
                + "\n"
            )
        output.write(
            json.dumps({"final_memory": runtime.store.snapshot(include_history=True)})
            + "\n"
        )


if __name__ == "__main__":
    main()
