#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 5 ]; then
    echo "Usage: $0 <config> <result_path> <gpu_list(comma)> <data_id_start> <data_id_end> [start_idx(default:0)]"
    echo "Environment variables: TARGET_ENV (default: targetdiff), AMBER_ENV (default: openmm-env)"
    exit 1
fi

CONFIG_FILE=$1
RESULT_PATH=$2
GPU_LIST_RAW=$3
DATA_START=$4
DATA_END=$5
START_IDX=${6:-0}
export CHEM_EVAL_SPEED=${CHEM_EVAL_SPEED:-strict}
export CHEM_EVAL_MIN_EVERY=${CHEM_EVAL_MIN_EVERY:-1}

IFS=',' read -r -a GPU_ARRAY <<< "$GPU_LIST_RAW"
NODE_ALL=${#GPU_ARRAY[@]}
if [ $NODE_ALL -eq 0 ]; then
    echo "No GPU specified"
    exit 1
fi

TARGET_ENVIRONMENT=${TARGET_ENV:-targetdiff}
AMBER_ENVIRONMENT=${AMBER_ENV:-openmm-env}

echo "Launching sampling across GPUs: ${GPU_ARRAY[*]} (nodes=$NODE_ALL)"
pids=()
CURRENT_AMBER_PID=""
cleanup() {
    status=$?
    trap - EXIT INT TERM
    if [ ${#pids[@]} -gt 0 ]; then
        echo "Cleaning up ${#pids[@]} background sampling jobs..."
        for pid in "${pids[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                kill "$pid" 2>/dev/null || true
            fi
        done
    fi
    if [ -n "${CURRENT_AMBER_PID}" ] && kill -0 "${CURRENT_AMBER_PID}" 2>/dev/null; then
        echo "Terminating running AMBER job (pid=${CURRENT_AMBER_PID})"
        kill "${CURRENT_AMBER_PID}" 2>/dev/null || true
    fi
    exit $status
}
trap cleanup EXIT INT TERM
for idx in "${!GPU_ARRAY[@]}"; do
    gpu_id=${GPU_ARRAY[$idx]}
    echo "[Sampling] Start worker idx=${idx}, GPU=${gpu_id}"
    env CUDA_VISIBLE_DEVICES=${gpu_id} conda run -n "${TARGET_ENVIRONMENT}" bash scripts/batch_sample_diffusion.sh \
        "${CONFIG_FILE}" "${RESULT_PATH}" "${NODE_ALL}" "${idx}" "${START_IDX}" &
    pids+=($!)
done

for pid in "${pids[@]}"; do
    wait $pid
done
pids=()
echo "Sampling finished for GPUs: ${GPU_ARRAY[*]}"

echo "Recomputing chem-eval+AMBER for data IDs ${DATA_START}-${DATA_END}"
total_ids=$((DATA_END - DATA_START + 1))
completed=0
for ((data_id=DATA_START; data_id<=DATA_END; data_id++)); do
    result_file="${RESULT_PATH}/result_${data_id}.pt"
    if [ ! -f "$result_file" ]; then
        echo "[AMBER] Skip data_id=${data_id} (missing $result_file)"
        continue
    fi
    echo "[AMBER] data_id=${data_id} via env ${AMBER_ENVIRONMENT}"
    conda run -n "${AMBER_ENVIRONMENT}" python scripts/post_compute_amber.py --result_path "${RESULT_PATH}" --data_id "${data_id}" &
    CURRENT_AMBER_PID=$!
    wait "${CURRENT_AMBER_PID}"
    CURRENT_AMBER_PID=""
    completed=$((completed + 1))
    echo "[AMBER] Progress: ${completed}/${total_ids}"
done

echo "All sampling + AMBER tasks completed (data ${DATA_START}-${DATA_END})."
