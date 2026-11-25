#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
root=$(pwd -P)
stage="$root/data/mem"
export PYTHONPATH="$stage/src/LIBERO-f78abd68ee283de9f9be3c8f7e2a9ad60246e95c:$root${PYTHONPATH:+:$PYTHONPATH}"
export LIBERO_CONFIG_PATH="$stage/config" TMPDIR="$stage/tmp" XDG_CACHE_HOME="$stage/cache"
export HF_HOME="$stage/cache/hf" OPENPI_DATA_HOME="$stage/cache/openpi" HF_HUB_OFFLINE=1
export MUJOCO_GL=egl PYTHONDONTWRITEBYTECODE=1 VOLO_GDINO_MODEL="$stage/gdino"
export MPLCONFIGDIR="$stage/cache/matplotlib" NUMBA_CACHE_DIR="$stage/cache/numba"
export CUDA_CACHE_PATH="$stage/cache/cuda" TRITON_CACHE_DIR="$stage/cache/triton"
export VLM_MODEL=qwen3.5-plus-2026-04-20
export VOLO_QWEN_NONTHINKING=1
export VLM_BASE_URL="${DASHSCOPE_BASE_URL:-https://dashscope.aliyuncs.com/compatible-mode/v1}"
py="$stage/libero-env/bin/python"
run=${1:?choose preflight, policy, proxy, probe or eval}
arm=${2:-baseline}
case "$arm" in
    baseline) semantic=off; visual=off; input=history ;;
    semantic) semantic=episode; visual=off; input=history ;;
    visual) semantic=off; visual=enforce; input=history ;;
    dual) semantic=episode; visual=enforce; input=history ;;
    current) semantic=off; visual=enforce; input=current ;;
    *) exit 2 ;;
esac
case "$run" in
    preflight)
        "$py" scripts/mem_preflight.py
        "$stage/openpi-env/bin/python" -c 'from openpi.policies import policy_config; from openpi.training.config import get_config; print(get_config("pi05_libero").name)'
        ;;
    policy)
        exec "$stage/openpi-env/bin/python" "$stage/src/openpi/scripts/serve_policy.py" --port 18002 policy:checkpoint --policy.config=pi05_libero --policy.dir="$stage/checkpoints/pi05_libero"
        ;;
    recorded)
        exec "$py" scripts/mem_recorded_vision.py
        ;;
    proxy|probe|replay)
        IFS= read -r VLM_API_KEY
        export VLM_API_KEY
        if [[ "$run" == probe ]]; then
            exec "$root/.venv/bin/python" scripts/mem_probe.py
        fi
        if [[ "$run" == replay ]]; then
            exec "$py" scripts/mem_recorded_replan.py
        fi
        batch=${3:?supply a fresh batch ID}
        [[ "$batch" =~ ^[[:alnum:]_-]+$ ]]
        out="$root/results/mem/$batch/$arm"
        mkdir -p "$(dirname "$out")"
        mkdir "$out"
        git rev-parse HEAD > "$out/commit"
        printf '%s\n' "$VLM_MODEL" "$VLM_BASE_URL" "$semantic" "$visual" "$input" > "$out/config.txt"
        exec "$stage/libero-env/bin/vlm-orchestrator" --env libero --mode subgoal --failure-monitor vlm \
            --recovery-mode replan --semantic-memory "$semantic" --visual-memory "$visual" --visual-input "$input" \
            --check-interval 40 --subgoal-timeout 9999 --vlm-model "$VLM_MODEL" --vlm-base-url "$VLM_BASE_URL" \
            --vla-port 18002 --host 127.0.0.1 --port 18001 --log-dir "$out"
        ;;
    eval)
        batch=${3:?supply proxy batch ID}
        [[ "$batch" =~ ^[[:alnum:]_-]+$ ]]
        out="$root/results/mem/$batch/$arm"
        test -f "$out/config.txt"
        start=$SECONDS
        for suite in libero_90 libero_10; do
            match=LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate
            if [[ "$suite" == libero_10 ]]; then match+=_and_put_the_yellow_and_white_mug_on_the_right_plate; fi
            "$py" examples/libero/libero_eval_client.py --host 127.0.0.1 --port 18001 --task-suite-name "$suite" \
                --task-filter "$match" --max-tasks 1 --seed 7 --num-trials-per-task 1 --replan-steps 5 --score-placements --camera-calibration
        done
        echo "$(( SECONDS - start ))" > "$out/eval_wall_s"
        ;;
    *) exit 2 ;;
esac
