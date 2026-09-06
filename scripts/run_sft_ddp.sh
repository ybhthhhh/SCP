#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
TRAIN_PYTHON=${TRAIN_PYTHON:-/mnt/zsr/venvs/lora32/bin/python}
TRAIN_SCRIPT=${TRAIN_SCRIPT:-${PROJECT_ROOT}/scripts/train_sft_ddp.py}
BASE_MODEL=${BASE_MODEL:-${PROJECT_ROOT}/models/DeepSeek-R1-Distill-Qwen-1.5B}
DATASET=${DATASET:-${PROJECT_ROOT}/data/processed/scp-sft-237.jsonl}
OUTPUT_DIR=${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/lightweight/sft-adapter}
GPU_IDS=${GPU_IDS:-0,1,2,3,4,5,6,7}
NUM_GPUS=${NUM_GPUS:-8}
MASTER_PORT=${MASTER_PORT:-29517}
P2P_TEST=${P2P_TEST:-/share/platform/p2pBandwidthTest}

# Provisional lightweight values; override after the target-length memory smoke test.
MAX_STEPS=${MAX_STEPS:-30}
SEQ_LEN=${SEQ_LEN:-6144}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-1}
GRAD_ACCUM=${GRAD_ACCUM:-1}
LR=${LR:-1e-5}
LORA_R=${LORA_R:-8}
LORA_ALPHA=${LORA_ALPHA:-16}
LORA_DROPOUT=${LORA_DROPOUT:-0.05}

if (( NUM_GPUS > 1 )); then
  P2P_LOG=$(mktemp /tmp/scp-p2p.XXXXXX)
  trap 'rm -f "${P2P_LOG}"' EXIT
  timeout --kill-after=2s 10s "${P2P_TEST}" >"${P2P_LOG}" 2>&1 || true
  if ! grep -q "P2P Connectivity Matrix" "${P2P_LOG}"; then
    echo "P2P preflight failed; refusing to start DDP training" >&2
    exit 3
  fi
fi

ARGS=(
  "${TRAIN_SCRIPT}"
  --model-path "${BASE_MODEL}"
  --dataset "${DATASET}"
  --output-dir "${OUTPUT_DIR}"
  --max-steps "${MAX_STEPS}"
  --seq-len "${SEQ_LEN}"
  --per-device-batch-size "${PER_DEVICE_BATCH_SIZE}"
  --grad-accum "${GRAD_ACCUM}"
  --lr "${LR}"
  --lora-r "${LORA_R}"
  --lora-alpha "${LORA_ALPHA}"
  --lora-dropout "${LORA_DROPOUT}"
  --gradient-checkpointing
  --run-name scp-light-sft-237-ddp
)

if [[ ${DRY_RUN:-0} == 1 ]]; then
  printf 'CUDA_VISIBLE_DEVICES=%q ' "${GPU_IDS}"
  printf '%q ' "${TRAIN_PYTHON}" -m torch.distributed.run --standalone      --nproc_per_node "${NUM_GPUS}" --master_port "${MASTER_PORT}" "${ARGS[@]}" "$@"
  printf '\n'
  exit 0
fi

if (( NUM_GPUS == 1 )); then
  CUDA_VISIBLE_DEVICES="${GPU_IDS}" "${TRAIN_PYTHON}" "${ARGS[@]}" "$@"
else
  CUDA_VISIBLE_DEVICES="${GPU_IDS}" "${TRAIN_PYTHON}" -m torch.distributed.run \
    --standalone --nproc_per_node "${NUM_GPUS}" --master_port "${MASTER_PORT}" \
    "${ARGS[@]}" "$@"
fi
