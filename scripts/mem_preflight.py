"""Validate installed assets and task registration without constructing a simulator."""

import json
import os
from pathlib import Path

import mujoco
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from openpi_client import websocket_client_policy
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor


stage = Path(os.environ["LIBERO_CONFIG_PATH"]).parent
print("MuJoCo", mujoco.__version__, OffScreenRenderEnv.__name__, websocket_client_policy.__name__)
name = "LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate"
for suite, task in [("libero_90", name), ("libero_10", name + "_and_put_the_yellow_and_white_mug_on_the_right_plate")]:
    tasks = benchmark.get_benchmark_dict()[suite]()
    idx, = [i for i in range(tasks.n_tasks) if tasks.get_task(i).name == task]
    spec = tasks.get_task(idx)
    assert (Path(get_libero_path("bddl_files")) / spec.problem_folder / spec.bddl_file).is_file()
    initial = tasks.get_task_init_states(idx)
    assert len(initial) and len(initial[0])
    print(suite, idx, task, initial[0].shape)
model = stage / "gdino"
AutoProcessor.from_pretrained(model, local_files_only=True)
AutoModelForZeroShotObjectDetection.from_pretrained(model, local_files_only=True)
print("GroundingDINO loaded on CPU")
assert (stage / "cache/openpi/big_vision/paligemma_tokenizer.model").stat().st_size > 0
manifest = json.loads((stage / "checkpoints/manifest.json").read_text())
assert manifest
for item in manifest:
    assert (stage / item["name"]).stat().st_size == int(item["size"])
print("Verified checkpoint manifest", len(manifest), "objects")
