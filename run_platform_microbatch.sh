#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TRAIN_PY=${TRAIN_PY:-${PROJECT_ROOT}/train_policy_microbatch.py}
PLATFORM_PYTHON=${PLATFORM_PYTHON:-/mnt/zsr/venvs/lora32/bin/python}
P2P_TEST=${P2P_TEST:-/share/platform/p2pBandwidthTest}

NUM_GPUS=${NUM_GPUS:-1}
GPU_IDS=${GPU_IDS:-0}
MODEL_PATH=${MODEL_PATH:-/root/SCP/outputs/lightweight/dpo-merged-1536}
DATASET=${DATASET:-/root/SCP/data/processed/grpo-prompts-light.jsonl}
OUTPUT_DIR=${OUTPUT_DIR:-/root/SCP/outputs/lightweight/grpo-microbatch}
REWARD_SPEC=${REWARD_SPEC:-/root/SCP/scripts/grpo_math_reward.py:score}
RUN_NAME=${RUN_NAME:-scp-grpo-microbatch}

export GRPO_ANSWER_MAP=${GRPO_ANSWER_MAP:-/root/SCP/data/processed/grpo-answer-map-light.json}
export CUDA_VISIBLE_DEVICES=${GPU_IDS}
# Match the platform launcher environment so a detached shell can import the
# CoreX PyTorch build without relying on an interactive login profile.
export PYTHONPATH=/usr/local/corex/lib64/python3/dist-packages${PYTHONPATH:+:${PYTHONPATH}}
export LD_LIBRARY_PATH=/usr/local/openmpi/lib:/usr/local/iluvatar/lib64:/usr/local/corex/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export PATH=/usr/local/openmpi/bin:/usr/local/iluvatar/bin:/usr/local/corex/bin:${PATH}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
export NCCL_NET_GDR_LEVEL=${NCCL_NET_GDR_LEVEL:-0}
export NCCL_ASYNC_ERROR_HANDLING=${NCCL_ASYNC_ERROR_HANDLING:-1}
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-lo}
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export TORCH_SHOW_CPP_STACKTRACES=${TORCH_SHOW_CPP_STACKTRACES:-1}
# Enable only for a diagnostic run: it serializes CUDA work and is much slower.
export CUDA_LAUNCH_BLOCKING=${CUDA_LAUNCH_BLOCKING:-0}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export NCCL_DEBUG=${NCCL_DEBUG:-INFO}
mkdir -p "${OUTPUT_DIR}"
export NCCL_DEBUG_FILE=${NCCL_DEBUG_FILE:-${OUTPUT_DIR}/nccl.%h.%p.log}

if (( NUM_GPUS > 1 )); then
  if [[ "${SKIP_P2P_PREFLIGHT:-0}" != "1" ]]; then
    P2P_LOG=$(mktemp /tmp/scp-microbatch-p2p.XXXXXX)
    trap 'rm -f "${P2P_LOG}"' EXIT
    timeout --kill-after=5s 45s "${P2P_TEST}" >"${P2P_LOG}" 2>&1 || true
    if ! grep -q "P2P Connectivity Matrix" "${P2P_LOG}"; then
      echo "P2P preflight failed; refusing to start multi-GPU training" >&2
      exit 3
    fi
  fi
  LAUNCHER=("${PLATFORM_PYTHON}" -m torch.distributed.run --standalone --nproc_per_node="${NUM_GPUS}")
else
  LAUNCHER=("${PLATFORM_PYTHON}")
fi

exec "${LAUNCHER[@]}" "${TRAIN_PY}" \
  --model-path "${MODEL_PATH}" \
  --prompt-dataset "${DATASET}" \
  --output-dir "${OUTPUT_DIR}" \
  --algo grpo \
  --finetuning lora \
  --max-steps "${MAX_STEPS:-10}" \
  --seq-len "${SEQ_LEN:-512}" \
  --per-device-batch-size "${PER_DEVICE_BATCH_SIZE:-1}" \
  --grad-accum "${GRAD_ACCUM:-1}" \
  --lr "${LR:-1e-6}" \
  --max-tokens "${MAX_TOKENS:-256}" \
  --temperature "${TEMPERATURE:-1.0}" \
  --top-p "${TOP_P:-1.0}" \
  --num-generations "${NUM_GENERATIONS:-8}" \
  --rollout-micro-batch-size "${ROLLOUT_MICRO_BATCH_SIZE:-1}" \
  --train-micro-batch-size "${TRAIN_MICRO_BATCH_SIZE:-1}" \
  --rollout-backend hf \
  --generation-cache "${GENERATION_CACHE:-dynamic}" \
  --rollout-attention "${ROLLOUT_ATTENTION:-eager}" \
  --reward-backend function \
  --reward-spec "${REWARD_SPEC}" \
  --kl-coef "${KL_COEF:-1e-3}" \
  --kl-estimator "${KL_ESTIMATOR:-k3}" \
  --kl-vocab-chunk-size "${KL_VOCAB_CHUNK_SIZE:-4096}" \
  --ppo-clip-range "${PPO_CLIP_RANGE:-0.2}" \
  --lora-r "${LORA_R:-8}" \
  --lora-alpha "${LORA_ALPHA:-16}" \
  --lora-dropout "${LORA_DROPOUT:-0.0}" \
  --checkpoint-every "${CHECKPOINT_EVERY:-0}" \
  --run-name "${RUN_NAME}" \
  --gradient-checkpointing \
  "$@"
