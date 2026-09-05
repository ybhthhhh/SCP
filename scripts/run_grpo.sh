#!/usr/bin/env bash
set -euo pipefail

# Required paths. Use an experiment name that stays constant across restarts;
# trainer.resume_mode=auto then resumes from its latest checkpoint.
: "${VERL_ROOT:?set VERL_ROOT to the pinned verl checkout}"
: "${MODEL_PATH:?set MODEL_PATH to the merged DPO model}"
: "${TRAIN_FILE:?set TRAIN_FILE to dapo-math-17k.parquet}"
: "${VAL_FILE:?set VAL_FILE to an AIME validation parquet}"

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
EXPERIMENT_NAME=${EXPERIMENT_NAME:-scp-grpo-1.5b}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-${PROJECT_ROOT}/outputs/grpo/${EXPERIMENT_NAME}}
LAMBDA_LENGTH=${LAMBDA_LENGTH:-0.1}
LENGTH_TOLERANCE=${LENGTH_TOLERANCE:-256}
LENGTH_GAMMA=${LENGTH_GAMMA:-1.0}

cd "${VERL_ROOT}"
PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}" python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  data.train_files="${TRAIN_FILE}" \
  data.val_files="${VAL_FILE}" \
  data.train_batch_size=64 \
  data.max_prompt_length=2048 \
  data.max_response_length=12000 \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=64 \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=1e-3 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.65 \
  actor_rollout_ref.rollout.n=8 \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.val_kwargs.n=8 \
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=32768 \
  actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=32768 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  reward.num_workers=1 \
  reward.custom_reward_function.path="${PROJECT_ROOT}/verl_ext/group_length_reward.py" \
  reward.custom_reward_function.name=compute_score \
  +reward.custom_reward_function.reward_kwargs.lambda_length="${LAMBDA_LENGTH}" \
  +reward.custom_reward_function.reward_kwargs.tolerance="${LENGTH_TOLERANCE}" \
  +reward.custom_reward_function.reward_kwargs.gamma="${LENGTH_GAMMA}" \
  reward.reward_manager.source=importlib \
  reward.reward_manager.name=GroupLengthRewardManager \
  reward.reward_manager.module.path="${PROJECT_ROOT}/verl_ext/group_length_reward.py" \
  reward.reward_manager.module.name=GroupLengthRewardManager \
  trainer.project_name=scp-reproduction \
  trainer.experiment_name="${EXPERIMENT_NAME}" \
  trainer.n_gpus_per_node=4 \
  trainer.nnodes=1 \
  trainer.total_training_steps=220 \
  trainer.total_epochs=999 \
  trainer.save_freq=5 \
  trainer.test_freq=10 \
  trainer.default_local_dir="${CHECKPOINT_DIR}" \
  trainer.resume_mode=auto \
  trainer.max_actor_ckpt_to_keep=3 \
  trainer.logger='["console","wandb"]' \
  "$@"
