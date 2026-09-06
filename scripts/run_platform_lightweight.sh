#!/usr/bin/env bash
set -euo pipefail

STAGE=${1:?usage: bash scripts/run_platform_lightweight.sh sft|dpo|grpo [extra platform args...]}
shift

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PLATFORM_RUN=${PLATFORM_RUN:-/share/platform/scripts/platform_run.py}
GPU_IDS=${GPU_IDS:-0,1,2,3}
NUM_GPUS=${NUM_GPUS:-4}
P2P_TEST=${P2P_TEST:-/share/platform/p2pBandwidthTest}
BASE_MODEL=${BASE_MODEL:-${PROJECT_ROOT}/models/DeepSeek-R1-Distill-Qwen-1.5B}

case "${STAGE}" in
  sft)
    exec bash "${PROJECT_ROOT}/scripts/run_sft_ddp.sh" "$@"
    ;;
  dpo)
    MODEL_PATH=${MODEL_PATH:-${PROJECT_ROOT}/outputs/lightweight/sft-merged}
    DATASET=${DATASET:-${PROJECT_ROOT}/data/processed/scp-dpo-light.jsonl}
    OUTPUT_DIR=${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/lightweight/dpo-adapter}
    ARGS=(--mode train --algo dpo --finetuning lora --model "${MODEL_PATH}"
      --preference-dataset "${DATASET}" --output-dir "${OUTPUT_DIR}" --num-gpus "${NUM_GPUS}"
      --max-steps 25 --seq-len 2048 --prompt-max-len 512
      --per-device-batch-size 1 --grad-accum 4 --lr 1e-6 --beta 0.1
      --lora-r 8 --lora-alpha 16 --lora-dropout 0.05
      --run-name scp-light-dpo)
    ;;
  grpo)
    MODEL_PATH=${MODEL_PATH:-${PROJECT_ROOT}/outputs/lightweight/dpo-merged}
    DATASET=${DATASET:-${PROJECT_ROOT}/data/processed/grpo-prompts-light.jsonl}
    OUTPUT_DIR=${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/lightweight/grpo-adapter}
    REWARD_SPEC=${REWARD_SPEC:?set REWARD_SPEC to /absolute/path/reward.py:function_name}
    ARGS=(--mode train --algo grpo --finetuning lora --model "${MODEL_PATH}"
      --prompt-dataset "${DATASET}" --output-dir "${OUTPUT_DIR}" --num-gpus "${NUM_GPUS}"
      --max-steps 10 --seq-len 1536 --max-tokens 512
      --per-device-batch-size 1 --grad-accum 1 --lr 1e-6
      --num-generations 4 --temperature 1.0 --top-p 1.0
      --rollout-backend hf --reward-backend function --reward-spec "${REWARD_SPEC}"
      --kl-coef 1e-3 --ppo-clip-range 0.2
      --lora-r 8 --lora-alpha 16 --lora-dropout 0.0
      --run-name scp-light-grpo-10)
    ;;
  *)
    echo "unknown stage: ${STAGE}; expected sft, dpo, or grpo" >&2
    exit 2
    ;;
esac

if (( NUM_GPUS > 1 )); then
  P2P_LOG=$(mktemp /tmp/scp-p2p.XXXXXX)
  trap 'rm -f "${P2P_LOG}"' EXIT
  timeout --kill-after=2s 10s "${P2P_TEST}" >"${P2P_LOG}" 2>&1 || true
  if ! grep -q "P2P Connectivity Matrix" "${P2P_LOG}"; then
    echo "P2P preflight failed; refusing to start multi-GPU training" >&2
    exit 3
  fi
fi

CUDA_VISIBLE_DEVICES="${GPU_IDS}" python3 "${PLATFORM_RUN}" "${ARGS[@]}" "$@"
