# SCP 论文复现

复现论文 **Graph-Based Chain-of-Thought Pruning for Reducing Redundant
Reflections in Reasoning LLMs**（Findings of ACL 2026）。本仓库实现论文的完整
算法接口，重点恢复最后一阶段的带长度惩罚 GRPO，并针对节点中断提供自动续训。

## 复现范围

论文流程如下：

1. 使用 Appendix B 的触发词切分原始 CoT。
2. 调用 `qwen3-turbo` 类 OpenAI-compatible API，按 Appendix A 逐块构造 DAG。
3. 删除满足 `B(v) < k` 或 `d(v)/d_max > m` 的 review 节点；论文取 `k=2, m=0.9`。
4. 在剪枝后的 Light-R1 轨迹上做 LoRA SFT。
5. 对正确 rollouts 按 `review_ratio + normalized_length` 排序，构造 DPO pair。
6. 在 DAPO-Math-17k 上训练 220 步 GRPO：正确性奖励为主，只对正确轨迹施加
   组内相对长度惩罚。

本仓库将复现范围收缩到 DeepSeek-R1-Distill-Qwen-1.5B；论文实验使用单节点
4×A800。本机如果没有 GPU，只能运行预处理与单元测试，训练命令应在 4 GPU
节点执行。

## 已知论文缺失信息

论文没有报告 GRPO 长度奖励的 `lambda`、容忍边界 `Delta` 和指数 `gamma`，也没有
报告 DPO 的 beta。仓库默认使用以下**复现假设**，它们不是论文原值：

- `lambda=0.1`
- `Delta=256 tokens`
- `gamma=1.0`
- `DPO beta=0.1`

运行消融时通过环境变量 `LAMBDA_LENGTH`、`LENGTH_TOLERANCE`、
`LENGTH_GAMMA` 覆盖前三项。

当前上游 `Light-R1-SFTData/stage2-3k.json` 有 3,533 行，而论文表 2 报告
3,335 行。论文未说明额外删除的 198 行，因此仓库不会武断地截取前 3,335 行；
应在复现报告中同时记录“上游原始数量”和“成功构图/通过质量过滤的数量”。

## 安装和数据

```bash
python3 -m pip install -e '.[data,test]'
bash scripts/download_data.sh
hf download deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
  --local-dir models/DeepSeek-R1-Distill-Qwen-1.5B
```

数据来自官方公开源：

- `qihoo360/Light-R1-SFTData` 的 `stage2-3k.json`
- `BytedTsinghua-SIA/DAPO-Math-17k` 的 `dapo-math-17k.parquet`

大数据、模型、checkpoint、日志和本地论文 PDF 均被 `.gitignore` 排除，不会误传
GitHub。

## 图构建与剪枝

配置兼容 OpenAI Chat Completions 的服务：

```bash
export GRAPH_API_KEY='...'
export GRAPH_BASE_URL='https://your-endpoint/v1'
export GRAPH_MODEL='qwen3-turbo'

python3 scripts/build_graphs.py \
  --input data/raw/light-r1-stage2-3k.json \
  --output data/processed/graphs.jsonl

python3 scripts/build_sft.py \
  --input data/processed/graphs.jsonl \
  --output data/processed/scp-sft.jsonl \
  --k 2 --m 0.9
```

`graphs.jsonl` 每完成一条就 flush + fsync；重新执行同一命令会扫描已完成 ID 后
继续，因此节点崩溃不会重做整批。

## SFT 和 DPO

把 LLaMA-Factory 安装到训练环境后运行：

```bash
bash scripts/run_lf_stage.sh configs/sft_1.5b.yaml outputs/sft-1.5b
```

DPO 输入是 JSONL，每条 rollout 至少包含：`question_id`、`question`、`response`、
`correct`、`review_nodes`、`total_nodes`、`token_length`。构造并训练：

```bash
python3 scripts/build_dpo.py \
  --input data/processed/dpo-rollouts.jsonl \
  --output data/processed/scp-dpo.jsonl

bash scripts/run_lf_stage.sh configs/dpo_1.5b.yaml outputs/dpo-1.5b
llamafactory-cli export configs/export_dpo.yaml
```

## GRPO（崩溃后可恢复）

奖励不是普通的逐样本长度惩罚。论文先在同一问题的 8 个正确 rollout 中计算最短
长度 `L*(x)`，所以 `verl_ext/group_length_reward.py` 按 rollout UID 聚合完整组后再
返回奖励。由于最新版 verl 会把奖励任务分发到多个 actor，启动脚本强制
`reward.num_workers=1`；改大可能把同一个 UID 拆散并造成等待。

奖励适配器锁定并核验于 verl commit
`23af6a7a2e8d6efeeb2adbe5d1689c7a24f503a3`。在训练节点准备该版本：

```bash
git clone https://github.com/volcengine/verl.git
cd verl
git checkout 23af6a7a2e8d6efeeb2adbe5d1689c7a24f503a3
# 按 verl 官方说明安装与本机 CUDA 匹配的 vLLM/FSDP 依赖
```

启动训练：

```bash
export VERL_ROOT=/path/to/verl
export MODEL_PATH=/path/to/SCP/outputs/dpo-1.5b-merged
export TRAIN_FILE=/path/to/dapo-math-17k.parquet
export VAL_FILE=/path/to/aime-2024.parquet
export EXPERIMENT_NAME=scp-grpo-1.5b

bash scripts/run_grpo.sh
```

脚本对齐论文：8 rollouts、temperature 1.0、12,000 最大 response tokens、global
batch 64、AdamW actor learning rate `1e-6`、KL coefficient `1e-3`、220 steps。
它每 5 步保存到稳定目录，`trainer.resume_mode=auto` 会在同名实验再次启动时恢复
最新 checkpoint。不要在重启时更换 `EXPERIMENT_NAME` 或 `CHECKPOINT_DIR`。

## CoreX 节点轻量 GRPO 验证器

`train_policy_microbatch.py` 与 `run_platform_microbatch.sh` 是为当前 CoreX
DT1000 节点加入的最小 GRPO 验证路径。它不替代上面的 verl 论文配置：用于先验证
奖励、rollout、反传和多卡通信，再决定是否扩大到论文的 12,000 response tokens 和
220 步。

当前节点使用 Transformers 4.44.2 和平台提供的 PyTorch 2.1.0。该组合会让 Qwen2
走 eager attention；平台的动态 KV cache 在长 rollout 中出现过原生层崩溃。因此长
rollout 使用预分配 static KV cache，并以 `gqa_online` 替换单 token decode 中的
`repeat_kv + full softmax`，不改变 prompt prefill 或训练 forward。

训练 KL 默认是 sampled-token `k3` 估计：对 rollout 中已采样 action 的
`log_ratio = log p_ref - log p_policy` 计算
`exp(log_ratio) - log_ratio - 1`。在对当前策略的 action 分布取期望时，它等于
`KL(policy || reference)`，但单批数值是采样估计。`--kl-estimator exact` 保留作
数值对照；它需要全词表 KL 的反传图，在本节点每卡 32 GB、4096 response tokens 时
已在第二个训练 microbatch 的 backward 阶段 OOM，不应用于长序列训练。

运行单卡 4096-token 验证：

```bash
cd /root/SCP
NUM_GPUS=1 GPU_IDS=0 MAX_STEPS=1 MAX_TOKENS=4096 \
NUM_GENERATIONS=2 ROLLOUT_MICRO_BATCH_SIZE=1 TRAIN_MICRO_BATCH_SIZE=1 \
GENERATION_CACHE=static ROLLOUT_ATTENTION=gqa_online KL_ESTIMATOR=k3 \
CHECKPOINT_EVERY=1 bash start_ddp_detached.sh
```

`start_ddp_detached.sh` 会复制本次运行的训练源码、写入环境和 core-dump 设置，并以
`setsid /usr/bin/nohup` 启动。不要在多卡 DDP 训练期间从交互终端终止子进程；若需要
停止或迁移，先检查各 rank 和通信状态，再由启动器整体结束。每个 run 的
`rank*.events.jsonl`、`nohup.log`、NCCL 日志和 checkpoint 都保存在自己的输出目录。

验证脚本：

```bash
python test_gqa_online_attention.py train_policy_microbatch.py
python test_chunked_kl.py train_policy_microbatch.py
```

## 验证

```bash
pytest
python3 -m compileall -q src scripts verl_ext
bash -n scripts/*.sh
```

论文最终评测对 AIME24、AIME25、AMC23、MATH500、OlympiadBench 每题采样 10
次，使用 `math-verify` 报平均正确率与平均生成 token 数。完整复现还应记录随机种子、
数据快照、模型 commit、每阶段 checkpoint 和 W&B run ID。

## 上游项目

- [Light-R1](https://github.com/Qihoo360/Light-R1)
- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)
- [verl](https://github.com/volcengine/verl)
- [DAPO](https://github.com/BytedTsinghua-SIA/DAPO)
