#!/usr/bin/env bash
# Run one real Codex/VLA/RGB-D memory episode after the three services are ready.
set -euo pipefail

root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
cd -- "$root"
stage="$root/data/stage1"
gate=${MEMORY_GATE:-shadow}
case "$gate" in shadow|enforce) ;; *) echo "MEMORY_GATE must be shadow or enforce" >&2; exit 2 ;; esac
: "${VLM_BASE_URL:?Set the SSH-forwarded Codex bridge URL}"
: "${VLM_MODEL:?Set the Codex model identifier}"
export VLM_API_KEY="${VLM_API_KEY:-local-codex-session}"
export TMPDIR="$stage/tmp" LIBERO_CONFIG_PATH="$stage/config/libero"
export XDG_CACHE_HOME="$stage/cache" MPLCONFIGDIR="$stage/cache/matplotlib"
export NUMBA_CACHE_DIR="$stage/cache/numba" CUDA_CACHE_PATH="$stage/cache/cuda"
export HF_HOME="$stage/cache/huggingface" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
mkdir -p "$root/results/libero"
run=$(mktemp -d "$root/results/libero/remm_codex_${gate}_XXXXXXXX")
policy_port=${SMOKE_POLICY_PORT:-18002}
proxy_port=${SMOKE_PROXY_PORT:-18001}
perception_url=${MEMORY_PERCEPTION_URL:-http://127.0.0.1:18003}
confidence=${MEMORY_MIN_CONFIDENCE:-0.8}
task_suite=${SMOKE_TASK_SUITE:-libero_spatial}
task_filter=${SMOKE_TASK_FILTER:-from_table_center}

printf 'run=%s\n' "$run"
{
    printf 'commit=%s\n' "$(git rev-parse HEAD)"
    printf 'branch=%s\n' "$(git branch --show-current)"
    printf 'suite=%s\ntask_filter=%s\nseed=7\ninitial_state_index=0\n' "$task_suite" "$task_filter"
    printf 'memory_mode=episodic\nmemory_gate=%s\nmemory_min_confidence=%s\n' "$gate" "$confidence"
    printf 'vlm_base_url=%s\nvlm_model=%s\nperception_url=%s\n' "$VLM_BASE_URL" "$VLM_MODEL" "$perception_url"
    printf 'policy_port=%s\nproxy_port=%s\nmonitor_interval=40\nterminal_observations=2\nterminal_observation_interval=40\n' "$policy_port" "$proxy_port"
} >"$run/config.txt"

curl -fsS --max-time 5 "http://127.0.0.1:$policy_port/healthz" >/dev/null
curl -fsS --max-time 5 "$perception_url/healthz" >"$run/perception-health.json"
curl -fsS --max-time 5 "${VLM_BASE_URL%/v1}/healthz" >"$run/codex-health.json"
"$root/.venv/bin/python" - "$proxy_port" <<'PY'
import socket
import sys
with socket.socket() as listener:
    listener.bind(("127.0.0.1", int(sys.argv[1])))
PY

proxy_pid=
cleanup() {
    if [[ -n "$proxy_pid" ]]; then
        kill "$proxy_pid" 2>/dev/null || true
        wait "$proxy_pid" 2>/dev/null || true
    fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

"$root/.venv/bin/python" -B -m vlm_orchestrator.cli \
    --host 127.0.0.1 --port "$proxy_port" --vla-host 127.0.0.1 --vla-port "$policy_port" \
    --env libero --mode subgoal --failure-monitor vlm --recovery-mode replan \
    --memory-mode episodic --memory-gate "$gate" --memory-perception-url "$perception_url" \
    --memory-min-confidence "$confidence" \
    --vlm-base-url "$VLM_BASE_URL" --vlm-model "$VLM_MODEL" \
    --enable-gt-metrics --gt-metric-types perception,scene_qa,gt_state \
    --log-dir "$run/orchestrator" >"$run/orchestrator.log" 2>&1 &
proxy_pid=$!
for ((i=0; i<60; i++)); do
    kill -0 "$proxy_pid" || { tail -40 "$run/orchestrator.log"; exit 1; }
    if (echo >"/dev/tcp/127.0.0.1/$proxy_port") 2>/dev/null; then break; fi
    sleep 1
done
(echo >"/dev/tcp/127.0.0.1/$proxy_port") 2>/dev/null

PYTHONPATH="$root:$stage/libero-source" MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
    MUJOCO_EGL_DEVICE_ID="${SMOKE_EGL_GPU:-2}" \
    "$stage/.libero311-venv/bin/python" -B examples/libero/run_eval.py \
    --host 127.0.0.1 --port "$proxy_port" --task-suite-name "$task_suite" \
    --task-filter "$task_filter" --max-tasks 1 --num-trials-per-task 1 --seed 7 \
    --terminal-observations 2 --terminal-observation-interval 40 \
    --enable-depth --enable-gt-state --log-dir "$run/eval" --video-out-path "$run/videos" \
    2>&1 | tee "$run/eval.log"

if grep -E 'Exception during episode|GT state export failed|Failed to init GT state exporter' "$run/eval.log"; then
    exit 1
fi
echo "Episode output saved in $run"
