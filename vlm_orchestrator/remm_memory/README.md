# Revisable embodied memory

This independent module adapts the evidence ledger and supported assertion graph
in REMM section III to VoLoAgent. Each episode owns its memory. The frozen VLA
receives its native observation and the selected instruction. The existing
strategy history remains available through the default configuration.

The implementation stores immutable evidence events, persistent object references,
initial and last trusted positions, and assertions with TRUE, FALSE or UNKNOWN
values. Each derived completion records its required parent assertion IDs. A
contradicted physical fact supersedes that assertion and invalidates its dependent
completions. The evidence and earlier assertions remain available for inspection.
Previously completed physical conditions continue to be checked across replans.
A replacement plan must reuse an object's ID and description. Fresh accepted
evidence can restore an invalidated completion.

## Experiment configurations

Use the same model, observation sequence, seeds, monitor interval and perception
settings when comparing these configurations.

| CLI flags | Input available to the high-level model |
| --- | --- |
| `--memory-mode off` | Existing strategy context |
| `--memory-mode current --memory-gate shadow` | Current-frame structured geometry |
| `--memory-mode episodic --memory-gate shadow` | Current verification, retained facts, bounded evidence history and revisions |
| `--memory-mode episodic --memory-gate enforce` | Same memory plus verified completion and recovery gates |

Shadow mode records each proposed completion gate and permits the original
control decision. Enforce supports the standard LIBERO `subgoal + vlm + replan`
path. TRUE requires a released object, matching geometry and two distinct
agreeing observations with limited position variation. Current-frame mode uses
instantaneous geometry for its ablation. Missing detections retain the last
trusted history while the current verification becomes UNKNOWN. Held objects,
geometric contradictions and invalidated earlier obligations block completion.
Uncovered subgoals are reported as UNKNOWN.

After the configured additional observation budget, enforcement requests ordinary
replanning. Failed recovery returns zero-displacement LIBERO actions and preserves
the last valid measured gripper command. Repair retries are bounded. This takes
effect when the client receives the next response and flushes its action chunk.
Physical stopping latency and closed-loop recovery remain GPU experiment gates.

```bash
vlm-orchestrator --env libero --mode subgoal \
  --failure-monitor vlm --recovery-mode replan \
  --memory-mode episodic --memory-gate shadow \
  --memory-perception-url http://127.0.0.1:8003 \
  --vlm-base-url "$VLM_BASE_URL" --vlm-model "$VLM_MODEL" \
  --enable-gt-metrics --gt-metric-types perception,scene_qa,gt_state \
  --log-dir results/libero/remm_shadow
```

The optional perception URL uses the existing GDino `/detect` and SAM2
`/segment` endpoints. RGB, metric depth, camera intrinsics and camera-to-world
extrinsics must be registered. LIBERO's policy-image horizontal mirror is
undone for geometry. Multiple matching detections or overlapping instance masks
produce UNKNOWN. The default confidence gate is 0.8. Use
`--memory-min-confidence` for a separately recorded calibration experiment.
The normalized LIBERO gripper scalar is 0 when open and 1 when closed. Opening
provides a release signal. Closed fingers alone provide no positive holding
confirmation.

For another online tracker, supply the proxy-only field below. Coordinates use
metres in a shared +Z-up world frame. The producer must supply actual online
estimates and stable frame identifiers.

```json
{
  "observation/remm_objects": {
    "coordinate_frame": "world",
    "units": "m",
    "source": "rgbd",
    "objects": [
      {
        "id": "red",
        "position": [0.0, 0.0, 0.15],
        "bounds": [[-0.025, -0.025, 0.1], [0.025, 0.025, 0.2]],
        "holding": false,
        "confidence": 0.95,
        "frame_id": "camera0:step10"
      }
    ]
  }
}
```

The planner adds a `memory` object to its existing plan response. Its symbolic
contracts request verification. The runtime obtains geometry from perception.

```json
{
  "subgoals": ["place red block on blue block"],
  "ordered": true,
  "memory": {
    "objects": [
      {"id": "red", "description": "red block"},
      {"id": "blue", "description": "blue block"}
    ],
    "goals": [
      {"subgoal_index": 0, "predicates": [{"name": "on", "args": ["red", "blue"]}]}
    ]
  }
}
```

Registered relations are `holding(object)`, `on(object,support)` and
`in(object,container)`. Containment requires independently estimated
`interior_bounds` for the container. A visible exterior surface alone leaves
containment UNKNOWN. Support uses observed surface bounds or an explicit
`support_z`. Partial visibility and imperfect masks can bias these estimates,
so their physical accuracy requires later calibration and LIBERO evaluation.
The live adapter resolves unique object descriptions. Tracking visually
indistinguishable instances through occlusion requires an external instance
tracker.

## CPU verification and replay

```bash
python -m pytest tests/test_remm_store.py tests/test_remm_perception.py \
  tests/test_remm_runtime.py tests/test_remm_strategy.py tests/test_remm_proxy.py

python -m vlm_orchestrator.remm_memory.replay \
  --plan examples/remm/plan.json \
  --input examples/remm/observations.jsonl \
  --output results/remm/replay.jsonl --mode episodic
```

The example is synthetic online perception. It covers a held object, release,
stable placement, occlusion and a later displacement. Its second stable released
observation supports completion. Occlusion preserves the historical conclusion.
Displacement invalidates it. Replay output includes each verification and the
complete final evidence ledger and assertion history.

Normal experiments write `remm_snapshot` and `remm_event` records to the existing
episode `rewrites.jsonl`, including mode, gate, plan version, simulation step,
coverage, evidence references, decisions and repairs. Episodic VLM context includes
up to 32 referenced/recent evidence records and eight recent assertion revisions,
with truncation counts. The in-process ledger remains complete for the episode.
Planning and replanning reserve at least 2048 output tokens for the extra contract
fields. Record this budget when comparing against the baseline.

GT observations and GT segmentation are excluded by explicit online-input
whitelists. The proxy strips `observation/remm_*` from VLA inputs. GT may still
be enabled for the existing passive scoring infrastructure. Learned perception,
GPU inference, task success rates and error-rate improvements require the later
matched closed-loop experiments in the project roadmap.

The design follows [REMM section III](https://arxiv.org/html/2610.12710).
