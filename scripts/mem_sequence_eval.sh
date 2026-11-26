#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
root=$(pwd -P)
stage="$root/data/mem"
suite="${1:?smoke or full}"
batch="${2:?batch name}"
gpu="${3:?physical GPU index}"
[[ "$batch" =~ ^[A-Za-z0-9_-]+$ && "$gpu" =~ ^[0-9]+$ ]]
case "$suite" in
    smoke) tasks=(0 2); states=(0) ;;
    full) tasks=(0 1 2 3 4 5); states=(1 2 3) ;;
    *) echo "Expected smoke or full" >&2; exit 2 ;;
esac
IFS= read -r credential
out="$root/results/mem/$batch"
mkdir -p "$out"
exec 9>"$stage/tmp/libero-mem.lock"
flock -n 9
export PYTHONPATH="$stage/libero-mem:$stage/libero-mem/thirdparty/robosuite:$root:$stage/src/openpi/src:$stage/src/openpi/packages/openpi-client/src"
export LIBERO_CONFIG_PATH="$stage/libero-mem-config" TMPDIR="$stage/tmp" XDG_CACHE_HOME="$stage/cache"
export HF_HOME="$stage/cache/hf" OPENPI_DATA_HOME="$stage/cache/openpi" HF_HUB_OFFLINE=1
export MPLCONFIGDIR="$stage/cache/matplotlib" NUMBA_CACHE_DIR="$stage/cache/numba"
export CUDA_CACHE_PATH="$stage/cache/cuda" TRITON_CACHE_DIR="$stage/cache/triton"
export MUJOCO_GL=egl PYTHONDONTWRITEBYTECODE=1 VOLO_GDINO_MODEL="$stage/gdino"
export CUDA_VISIBLE_DEVICES="$gpu" MUJOCO_EGL_DEVICE_ID="$gpu"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${MEM_GPU_FRACTION:-.36}"
export XLA_FLAGS="--xla_gpu_cuda_data_dir=$stage/openpi-env/lib/python3.11/site-packages/nvidia/cuda_nvcc"
export OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=3 PYTHONUNBUFFERED=1
export VLM_MODEL=qwen3.5-plus-2026-04-20 VOLO_QWEN_NONTHINKING=1
export VLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
py="$stage/libero-env/bin/python"
commit=$(git rev-parse HEAD)
policy=
proxy=
cleanup() {
    for pid in "$proxy" "$policy"; do
        if [[ -n "$pid" ]]; then
            kill "$pid" 2>/dev/null || true
            wait "$pid" 2>/dev/null || true
        fi
    done
    proxy=
    policy=
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

ready() {
    for ((attempt=0; attempt<150; attempt++)); do
        kill -0 "$1" 2>/dev/null || return 1
        if ss -Hltn "sport = :$2" | grep -q .; then return 0; fi
        sleep 2
    done
    return 1
}

run_case() {
    local arm="$1" task="$2" init="$3"
    local dir="$out/$arm/task-$task/init-$init"
    if [[ -f "$dir/complete" ]]; then
        test "$(head -1 "$dir/config.txt")" = "$commit"
        echo "$arm task=$task init=$init already complete"
        return
    fi
    if ss -Hltn 'sport = :18001' | grep -q . || ss -Hltn 'sport = :18002' | grep -q .; then
        echo 'A required service port is in use' >&2
        return 1
    fi
    mkdir -p "$(dirname "$dir")"
    mkdir "$dir"
    local semantic=off visual=off
    if [[ "$arm" == dual ]]; then semantic=episode; visual=enforce; fi
    printf '%s\n' "$commit" "$gpu" "$task" "$init" "$VLM_MODEL" "$VLM_BASE_URL" "$semantic" "$visual" \
        "$XLA_PYTHON_CLIENT_MEM_FRACTION" > "$dir/config.txt"
    echo "$arm task=$task init=$init started $(date -Is)"
    "$stage/openpi-env/bin/python" "$stage/src/openpi/scripts/serve_policy.py" \
        --port 18002 policy:checkpoint --policy.config=pi05_libero \
        --policy.dir="$stage/checkpoints/pi05_libero" > "$dir/policy.log" 2>&1 &
    policy=$!
    ready "$policy" 18002
    VLM_API_KEY="$credential" "$stage/libero-env/bin/vlm-orchestrator" \
        --env libero --mode subgoal --failure-monitor vlm --recovery-mode replan \
        --semantic-memory "$semantic" --visual-memory "$visual" --check-interval 40 \
        --subgoal-timeout 9999 --vlm-model "$VLM_MODEL" --vlm-base-url "$VLM_BASE_URL" \
        --vla-port 18002 --host 127.0.0.1 --port 18001 --log-dir "$dir" > "$dir/proxy.log" 2>&1 &
    proxy=$!
    ready "$proxy" 18001
    local start=$SECONDS
    "$py" examples/libero/run_mem_eval.py --host 127.0.0.1 --port 18001 \
        --task-ids "$task" --initial-state-offset "$init" --num-trials-per-task 1 --seed 7 \
        --replan-steps 5 --camera-calibration --score-progress > "$dir/eval.log" 2>&1
    echo "$(( SECONDS - start ))" > "$dir/eval_wall_s"
    test "$(find "$dir" -name eval_metadata.json | wc -l)" -eq 1
    "$py" - "$dir" "$task" "$init" <<'PY'
import json
import sys
from pathlib import Path

directory = Path(sys.argv[1])
metadata = json.loads(next(directory.rglob("eval_metadata.json")).read_text())
assert metadata["task_id"] == int(sys.argv[2])
assert metadata["init_state_idx"] == int(sys.argv[3])
assert metadata["executed_steps"]
PY
    cleanup
    touch "$dir/complete"
    echo "$arm task=$task init=$init complete $(date -Is)"
}

pair=0
for init in "${states[@]}"; do
    for task in "${tasks[@]}"; do
        if ((pair % 2 == 0)); then order=(baseline dual); else order=(dual baseline); fi
        for arm in "${order[@]}"; do run_case "$arm" "$task" "$init"; done
        pair=$((pair + 1))
    done
done
