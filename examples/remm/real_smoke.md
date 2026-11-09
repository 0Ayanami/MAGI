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

Once the actual policy and online perception services are ready on the server,
run the prepared script from that checkout.

```bash
export VLM_BASE_URL=http://127.0.0.1:18080/v1
export VLM_MODEL=gpt-6.1-sol
export MEMORY_GATE=shadow
bash scripts/run_remm_libero_smoke.sh
```

The fixed selection is standard `libero_spatial` task 2, selected by
`from_table_center`, seed 7 and initial state index 0. Its task is to put the
black bowl on the plate, which uses the registered `on` relation. The script
starts and cleans up its own orchestrator. VLA, perception and SSH-forward
process lifetimes are managed by the experiment owner. Results include exact
configuration, health replies, orchestrator logs, episode metadata and videos.
Set `MEMORY_GATE=enforce` for a separately recorded gate experiment.

The confidence default is 0.8. Inspect actual detector scores before choosing
and recording a different `MEMORY_MIN_CONFIDENCE`. Prewarm the real perception
endpoints so lazy model loading finishes before the runtime's ten-second request
timeout. The live grounder produces a release signal from the normalized gripper
observation, while a positive holding relation needs additional online evidence.
Containment needs independently estimated container interior geometry.

For a complete smoke result, inspect successful Codex plan and monitor calls,
valid memory contracts, multiple accepted online object observations with world
geometry, and actual VLA action responses. Report LIBERO task outcome and memory
verification separately. The current eval client ends immediately after success,
so an extra terminal observation or a terminal-frame replay is needed to verify
the last placement after the environment declares success.

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
