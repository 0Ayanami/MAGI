"""Controlled replan on fixed recorded images. No policy actions are executed."""

import io
import json
import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

from vlm_orchestrator.strategies.base import SessionState, StrategyContext
from vlm_orchestrator.strategies.episode_memory import EpisodeSummary
from vlm_orchestrator.strategies.subgoal import SubgoalConfig, SubgoalStrategy


skills = ["Put the white mug on the left plate", "Put the yellow and white mug on the right plate"]
goal = " and ".join(skills)
source = Path("data/mem/replay/file-000.parquet")
rows = pq.read_table(source, columns=["episode_index", "frame_index", "observation.images.image"]).to_pylist()
frames = {r["frame_index"]: np.asarray(Image.open(io.BytesIO(r["observation.images.image"]["bytes"])).convert("RGB"))
          for r in rows if r["episode_index"] == 0 and r["frame_index"] in (0, 110)}
for condition in ("baseline", "valid", "regressed"):
    config = SubgoalConfig(vlm_model=os.environ["VLM_MODEL"], vlm_base_url=os.environ["VLM_BASE_URL"])
    strategy = SubgoalStrategy(StrategyContext(vlm=None), config, failure_monitor="vlm", recovery_mode="replan")
    state = SessionState(original_instruction=goal, subgoals=skills.copy(), current_subgoal_idx=1,
                         episode_step=110, initial_image=frames[0], rewritten_instruction=skills[1])
    strategy._original_subgoals = skills.copy()
    strategy._call_state = state
    if condition != "baseline":
        state.semantic_summary = EpisodeSummary(goal)
        state.semantic_summary.record(skills[0], 100, "controlled_input", "observed_complete",
                                      "injected:placed-and-released")
        if condition == "regressed":
            state.semantic_summary.record(skills[0], 109, "controlled_input", "invalidated",
                                          "injected:lifted-off-left-plate")
    accepted = strategy._recycle({}, state, frames[110])
    print(json.dumps(dict(condition=condition, controlled=True, episode=0, frame=110,
                          plan=state.subgoals if accepted else None, events=state.log_entries)), flush=True)
