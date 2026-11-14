# Real LIBERO and Codex smoke configuration

The prepared topology uses server GPU 2 for the local `pi05_libero` policy,
GPU 0 for GDino/SAM2 perception, and the logged-in local Codex App Server for
multimodal planning and monitoring. The server checkout is
`/home2/wanquan/VoLoAgent`. Experiment files and caches belong to that checkout.

| Component | Listener |
| --- | --- |
| Local Codex HTTP bridge | `127.0.0.1:18080` |
| SSH reverse forward on server | `127.0.0.1:18080` |
| Local VLA on server GPU 2 | `127.0.0.1:18002` |
| GDino/SAM2 service on server GPU 0 | `127.0.0.1:18003` |
| Memory orchestrator | `127.0.0.1:18001` |

Start the local bridge from the local checkout with its existing ChatGPT login.

```powershell
.\.venv\Scripts\python.exe -m vlm_orchestrator.utils.codex_vlm_server --port 18080 --model gpt-6.1-sol --effort low --timeout 300 --log-dir results/remm_codex/local
```

Keep a reverse SSH forward open from the local computer.

```powershell
ssh -N -o BatchMode=yes -o ExitOnForwardFailure=yes -R 127.0.0.1:18080:127.0.0.1:18080 wanquan@222.20.97.104
```

The bridge accepts the project's nonstreaming OpenAI chat-completions request
format and passes text and base64 images to a real Codex turn. Each request uses
an ephemeral thread. The App Server process persists across requests. Exact input,
output, thread/turn IDs, latency and available token usage are saved under the
specified log directory. Caller token limits are recorded as unapplied because
this App Server path controls its own generation budget. Model or transport
errors return HTTP 503 and remain visible in the experiment logs.

### Qwen inference through the same App Server

For the Qwen experiment, add `DASHSCOPE_API_KEY=your_key` to the local,
Git-ignored `.env` file and start the bridge from the repository root.
Use `--env-file` for a different local path. An existing environment variable
takes precedence. The bridge reads only the configured API key from this file.
Use the Responses endpoint for your Bailian workspace.
For example, with a Beijing workspace and `qwen3.8-max` enabled:

```powershell
.\.venv\Scripts\python.exe -m vlm_orchestrator.utils.codex_vlm_server `
  --port 18080 --model qwen3.8-max --effort low `
  --responses-base-url 'https://<WorkspaceId>.cn-beijing.maas.aliyuncs.com/compatible-mode/v1' `
  --log-dir results/remm_qwen/local
```

Use `VLM_MODEL=qwen3.8-max` and the SSH-forwarded bridge URL as `VLM_BASE_URL`.
The bridge starts a custom provider in its App Server subprocess. App Server
handles the turn and calls the provider over HTTP Responses SSE. See the
[Bailian Responses reference](https://help.aliyun.com/zh/model-studio/qwen-api-via-openai-responses)
and [Codex gateway requirements](https://learn.chatgpt.com/docs/enterprise/gateway-compatibility).
If your Codex version needs model metadata for the selected ID, pass a
compatible Codex model catalog using `--model-catalog`.

For this backend, verify an image-bearing plan and monitor request, including
the actual model/provider, valid JSON, and reported token usage. Use Bailian
usage and billing for API spend. The bridge marks ChatGPT quota snapshots as
not applicable to external inference.

## Required accounting for the next experiment

For the ChatGPT-backed baseline, use `gpt-6.1-sol` and reasoning effort `low`.
Record the actual returned model and requested effort for each backend.
Keep each experiment in a separate log directory, including warmup requests.

For every plan/monitor/replan request retain `result.json`, exact request,
thread/turn IDs and raw App Server events. Record end-to-end `latency_s`,
`queue_wait_s`, `service_latency_s` and `turn_latency_s` (from turn/start reply
to completion, excluding quota sampling). Report count, total, median and p95
by request type; report episode wall time and simulated steps separately.
Keep VLA, memory and perception timing from their respective runtime logs,
and keep warmup/model-loading and post-success audit timing separate.

Token accounting uses `usage.total` once per ephemeral thread: input, cached
input, output, reasoning output and total. Cached input is part of input;
reasoning output is part of output. Never add them a second time. Missing
usage means unavailable, not zero. Failed requests and retries remain in raw
events; usage may be incomplete if the server never reports it.

The bridge now samples `account/rateLimits/read` before and after each turn,
persisting `quota-before.json` and `quota-after.json`, including timestamps,
window IDs/reset times and any returned credit balance. Unsupported or timed
out sampling is explicitly marked unavailable and does not invent a charge.
Only compare percentage points within the same bucket and reset window;
resets, rounding and concurrent account use prevent attributing all changes
to this experiment. Report actual credit balances/deltas only when returned.
Token totals are not a conversion formula for subscription quota or currency.
See [official App Server accounting](https://learn.chatgpt.com/docs/app-server).

Once the actual policy and online perception services are ready on the server,
run the prepared script from that checkout.

```bash
export VLM_BASE_URL=http://127.0.0.1:18080/v1
export VLM_MODEL=gpt-6.1-sol
export MEMORY_GATE=shadow
bash scripts/run_remm_libero_smoke.sh
```

The default selection is standard `libero_spatial` task 2, selected by
`from_table_center`, seed 7 and initial state index 0. Its task is to put the
black bowl on the plate, which uses the registered `on` relation. The script
starts and cleans up its own orchestrator. VLA, perception and SSH-forward
process lifetimes are managed by the experiment owner. Results include exact
configuration, health replies, orchestrator logs, episode metadata and videos.
Set `MEMORY_GATE=enforce` for a separately recorded gate experiment.

For the 2026-11-10 live smoke, use `SMOKE_TASK_SUITE=libero_goal` and
`SMOKE_TASK_FILTER=put_the_bowl_on_the_plate` (task 8, seed 7, initial state 0).
The default spatial task contains several similar bowls; goal task 8 has one
black bowl and one white plate with a red rim. RGB-only prewarming measured
detector scores 0.610 and 0.729 with these distinguishing descriptions, so this
run explicitly sets `MEMORY_MIN_CONFIDENCE=0.4`. Actual descriptions and
completion contracts must still come from the real Codex plan. Do not replace
them with a fixed plan or ground-truth object bindings.

CUDA allocation was verified again on GPU 2 on 2026-11-10. The local pi05 VLA
is hosted on GPU 2 and real GDino/SAM2 perception on GPU 0. Detection uses the
same image orientation as Codex; masks and depth are converted together for
metric projection. Panda's signed finger positions are normalized using their
magnitudes so an open gripper correctly supplies release evidence.

The confidence default is 0.8. Inspect actual detector scores before choosing
and recording a different `MEMORY_MIN_CONFIDENCE`. Prewarm the real perception
endpoints so lazy model loading finishes before the runtime's ten-second request
timeout. The live grounder produces a release signal from the normalized gripper
observation, while a positive holding relation needs additional online evidence.
Containment needs independently estimated container interior geometry.

For a complete smoke result, inspect successful Codex plan and monitor calls,
valid memory contracts, multiple accepted online object observations with world
geometry, and actual VLA action responses. Report LIBERO task outcome and memory
verification separately. The prepared script requests two optional post-success
observations, separated by 40 real simulation steps of measured zero-displacement
hold. Returned policy actions are discarded during this audit. The first task
outcome and primary rollout length are recorded before the audit. Terminal
stability, simulation steps and observations are recorded separately. This
success-triggered audit supplies final placement evidence without adding the
success flag to the model inputs.

If the first success frame still has a closed gripper, a hold audit cannot test
the policy's later release. For a separately labelled bounded continuation,
set `SMOKE_TERMINAL_MODE=policy`. The policy then executes up to 40 additional
real actions, refreshing at the normal five-action replanning interval. No
release command is injected. The audit records policy actions, query steps and
stability separately; it does not change the captured primary outcome. This
mode may reveal later placement failure as well as successful release.

## Preparation result on 2026-11-09

The server showed free memory on GPU 0 and GPU 2, but both failed a minimal CUDA
tensor allocation, including UUID-based isolation. Directly loading the system
`libcuda.so.1` gave `cuInit(0) = 999 / CUDA_ERROR_UNKNOWN`. The policy failed
before accepting an episode. The user deferred driver repair.

The CUDA diagnostic is saved in
`results/remm_codex_readiness/cuda-diagnostic.json` locally and
`data/remm-smoke/cuda-diagnostic.json` on the server. LIBERO EGL rendering
successfully captured the selected task's initial RGB frame. That frame supports
a real Codex connectivity check. It does not establish a completed VLA/memory
rollout.

SAM2 and the small service dependencies are installed in the project environment.
SAM2 source is pinned to `2b90b9f5ceec907a1c18123530e92e794ad901a4`.
Perception model weights and the live perception service still need preparation
after CUDA initialization is restored. Host driver changes remain an
administrator operation.
