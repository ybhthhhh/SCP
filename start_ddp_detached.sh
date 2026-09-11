#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
OUTPUT_ROOT=${OUTPUT_ROOT:-/root/SCP/outputs/lightweight}
RUN_LABEL=${RUN_LABEL:-grpo-ddp}
RUN_ID=${RUN_ID:-${RUN_LABEL}-$(date -u +%Y%m%dT%H%M%SZ)}
OUTPUT_DIR=${OUTPUT_DIR:-${OUTPUT_ROOT}/${RUN_ID}}

if [[ -e "${OUTPUT_DIR}" ]]; then
  echo "Refusing to reuse existing output directory: ${OUTPUT_DIR}" >&2
  exit 2
fi
mkdir -p "${OUTPUT_DIR}"
ulimit -c unlimited || true

# Preserve the exact launcher and trainer used by this detached run.
cp "${PROJECT_ROOT}/train_policy_microbatch.py" "${OUTPUT_DIR}/train_policy_microbatch.py"
cp "${PROJECT_ROOT}/run_platform_microbatch.sh" "${OUTPUT_DIR}/run_platform_microbatch.sh"
chmod 755 "${OUTPUT_DIR}/run_platform_microbatch.sh"
sha256sum "${OUTPUT_DIR}/train_policy_microbatch.py" "${OUTPUT_DIR}/run_platform_microbatch.sh" \
  >"${OUTPUT_DIR}/source.sha256"

export OUTPUT_DIR
export TRAIN_PY="${OUTPUT_DIR}/train_policy_microbatch.py"
export NUM_GPUS=${NUM_GPUS:-8}
export GPU_IDS=${GPU_IDS:-0,1,2,3,4,5,6,7}
export MAX_STEPS=${MAX_STEPS:-10}
export NUM_GENERATIONS=${NUM_GENERATIONS:-8}
export ROLLOUT_MICRO_BATCH_SIZE=${ROLLOUT_MICRO_BATCH_SIZE:-1}
export TRAIN_MICRO_BATCH_SIZE=${TRAIN_MICRO_BATCH_SIZE:-1}
export MAX_TOKENS=${MAX_TOKENS:-256}
export CHECKPOINT_EVERY=${CHECKPOINT_EVERY:-10}
export NCCL_DEBUG_FILE="${OUTPUT_DIR}/nccl.%h.%p.log"

printf 'run_id=%s\nstarted_utc=%s\n' "${RUN_ID}" "$(date -u +%FT%TZ)" >"${OUTPUT_DIR}/run.env"
printf 'core_limit=%s\n' "$(ulimit -c)" >>"${OUTPUT_DIR}/run.env"
cat /proc/sys/kernel/core_pattern >>"${OUTPUT_DIR}/core_pattern.txt" 2>/dev/null || true
env | grep -E '^(NUM_GPUS|GPU_IDS|MAX_STEPS|NUM_GENERATIONS|ROLLOUT_MICRO_BATCH_SIZE|TRAIN_MICRO_BATCH_SIZE|MAX_TOKENS|CHECKPOINT_EVERY|GENERATION_CACHE|ROLLOUT_ATTENTION|KL_ESTIMATOR|KL_COEF|NCCL_|CUDA_LAUNCH_BLOCKING|TORCH_SHOW_CPP_STACKTRACES)=' \
  | sort >>"${OUTPUT_DIR}/run.env"
printf 'launcher_args=' >>"${OUTPUT_DIR}/run.env"
printf '%q ' "$@" >>"${OUTPUT_DIR}/run.env"
printf '\n' >>"${OUTPUT_DIR}/run.env"

setsid /usr/bin/nohup bash "${OUTPUT_DIR}/run_platform_microbatch.sh" "$@" >"${OUTPUT_DIR}/nohup.log" 2>&1 < /dev/null &
PID=$!
printf 'launcher_pid=%s\n' "${PID}" | tee -a "${OUTPUT_DIR}/run.env"
echo "Started detached DDP run: pid=${PID} output=${OUTPUT_DIR}"
