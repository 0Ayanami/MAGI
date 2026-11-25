"""Shadow replay of episode 0 of HuggingFaceVLA/libero, fixed SCENE5 agentview."""

import io
import argparse
import json
import os

import numpy as np
import pyarrow.parquet as pq
from PIL import Image
from libero.libero import benchmark

from examples.libero.libero_eval_client import get_libero_env
from vlm_orchestrator.perception.gdino import GroundingDINODetector
from vlm_orchestrator.strategies.base import SessionState, StrategyContext
from vlm_orchestrator.strategies.subgoal import SubgoalConfig, SubgoalStrategy


parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["shadow", "enforce"], default="shadow")
args = parser.parse_args()
suite = benchmark.get_benchmark_dict()["libero_10"]()
task = suite.get_task(4)
env, goal, K, extrinsic, _, _ = get_libero_env(task, 256, 7, camera_calibration=True)
env.close()
assert K is not None and extrinsic is not None
strategy = SubgoalStrategy(StrategyContext(vlm=None), SubgoalConfig(visual_memory=args.mode, check_interval=5),
                           failure_monitor="vlm", recovery_mode="replan")
strategy._placement_detector = GroundingDINODetector(os.environ["VOLO_GDINO_MODEL"], device="cpu")
skills = ["Put the white mug on the left plate", "Put the yellow and white mug on the right plate"]


def mock_cloud(system, content, **kwargs):
    if any("Scene instruction" in entry.get("text", "") for entry in content):
        return json.dumps({"subgoals": skills, "ordered": True})
    return json.dumps({"action": "continue", "status": "in_progress"})


strategy._vlm_call = mock_cloud
state = SessionState()
columns = ["episode_index", "frame_index", "observation.images.image", "observation.state"]
for row in pq.read_table("data/mem/replay/file-000.parquet", columns=columns).to_pylist():
    t = row["frame_index"]
    if row["episode_index"] != 0 or t % 5 or not 40 <= t <= 120:
        continue
    rgb = Image.open(io.BytesIO(row["observation.images.image"]["bytes"])).convert("RGB")
    obs = {"prompt": goal, "observation/image_raw": np.asarray(rgb),
           "observation/state": np.asarray(row["observation.state"]),
           "observation/camera_K": K, "observation/camera_extrinsic": extrinsic}
    state.episode_step = t
    sent, _ = strategy.process(obs, state)
    for event in state.log_entries:
        if event["type"] == "placement_check":
            print(json.dumps(dict(event, mode=args.mode, dispatched_prompt=sent["prompt"],
                                  flush_actions=state.flush_actions)), flush=True)
    state.flush_actions = False
    state.log_entries.clear()
    state.infer_count += 1
