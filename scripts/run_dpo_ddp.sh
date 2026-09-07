#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
TRAIN_PYTHON=${TRAIN_PYTHON:-/mnt/zsr/venvs/lora32/bin/python}
TRAIN_SCRIPT=${TRAIN_SCRIPT:-${PROJECT_ROOT}/scripts/train_dpo_ddp.py}
BASE_MODEL=${BASE_MODEL:-${PROJECT_ROOT}/outputs/lightweight/sft-merged}
DATASET=${DATASET:-${PROJECT_ROOT}/data/processed/scp-dpo-light-219.jsonl}
OUTPUT_DIR=${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/lightweight/dpo-adapter-1536}
NUM_GPUS=${NUM_GPUS:-8}
if [[ -z ${GPU_IDS:-} ]]; then
  if (( NUM_GPUS == 1 )); then
    GPU_IDS=0
  else
    GPU_IDS=0,1,2,3,4,5,6,7
  fi
fi
MASTER_PORT=${MASTER_PORT:-29517}
P2P_TEST=${P2P_TEST:-/share/platform/p2pBandwidthTest}
P2P_TIMEOUT=${P2P_TIMEOUT:-45s}
P2P_KILL_AFTER=${P2P_KILL_AFTER:-5s}

# Lightweight DPO values validated on a 32 GiB DT1000.
MAX_STEPS=${MAX_STEPS:-28}
SEQ_LEN=${SEQ_LEN:-1536}
PROMPT_MAX_LEN=${PROMPT_MAX_LEN:-512}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-1}
GRAD_ACCUM=${GRAD_ACCUM:-1}
LR=${LR:-1e-7}
BETA=${BETA:-0.1}
LORA_R=${LORA_R:-8}
LORA_ALPHA=${LORA_ALPHA:-16}
LORA_DROPOUT=${LORA_DROPOUT:-0.05}

ARGS=(
  "${TRAIN_SCRIPT}"
  --model-path "${BASE_MODEL}"
  --dataset "${DATASET}"
  --output-dir "${OUTPUT_DIR}"
  --max-steps "${MAX_STEPS}"
  --seq-len "${SEQ_LEN}"
  --prompt-max-len "${PROMPT_MAX_LEN}"
  --per-device-batch-size "${PER_DEVICE_BATCH_SIZE}"
  --grad-accum "${GRAD_ACCUM}"
  --lr "${LR}"
  --beta "${BETA}"
  --loss-type sigmoid
  --finetuning lora
  --lora-r "${LORA_R}"
  --lora-alpha "${LORA_ALPHA}"
  --lora-dropout "${LORA_DROPOUT}"
  --gradient-checkpointing
  --run-name scp-light-dpo-219
)

if [[ ${DRY_RUN:-0} == 1 ]]; then
  printf 'CUDA_VISIBLE_DEVICES=%q ' "${GPU_IDS}"
  if (( NUM_GPUS == 1 )); then
    printf '%q ' "${TRAIN_PYTHON}" "${ARGS[@]}" "$@"
  else
    printf '%q ' "${TRAIN_PYTHON}" -m torch.distributed.run --standalone       --nproc_per_node "${NUM_GPUS}" --master_port "${MASTER_PORT}"       "${ARGS[@]}" "$@"
  fi
  printf '
'
  exit 0
fi

if (( NUM_GPUS > 1 )); then
  P2P_LOG=$(mktemp /tmp/scp-p2p.XXXXXX)
  trap 'rm -f "${P2P_LOG}"' EXIT
  timeout --kill-after="${P2P_KILL_AFTER}" "${P2P_TIMEOUT}" "${P2P_TEST}" >"${P2P_LOG}" 2>&1 || true
  if ! grep -q "P2P Connectivity Matrix" "${P2P_LOG}"; then
    echo "P2P preflight failed; refusing to start DDP preference training" >&2
    exit 3
  fi
fi

if (( NUM_GPUS == 1 )); then
  CUDA_VISIBLE_DEVICES="${GPU_IDS}" "${TRAIN_PYTHON}" "${ARGS[@]}" "$@"
else
  CUDA_VISIBLE_DEVICES="${GPU_IDS}" "${TRAIN_PYTHON}" -m torch.distributed.run     --standalone --nproc_per_node "${NUM_GPUS}" --master_port "${MASTER_PORT}"     "${ARGS[@]}" "$@"
fi
