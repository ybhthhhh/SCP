# 1.5B 轻量复现说明

本配置只复现 SCP 的最小可运行链路，不追求论文训练规模。已完成的 237 条构图
数据、SFT 数据和三个训练阶段的 LoRA adapter 会提交 GitHub；基座模型和合并后的
完整模型体积过大，仍只保留在本地。

## 本机资源策略

本机有 8 张 32 GiB DT1000。SFT 默认使用 GPU 0–7 的原生 PyTorch DDP，每卡各放
一份 1.5B 基座和 LoRA adapter，不使用 DeepSpeed/ZeRO3。启动前运行
`/share/platform/p2pBandwidthTest`；检查失败时脚本会停止。8 卡能提高吞吐量，但
DDP 不会分摊单条长序列的激活显存，正式长度仍需先做显存探测。

## 当前已确定的范围

- 基座：本地 `DeepSeek-R1-Distill-Qwen-1.5B`
- 候选集：固定种子 42，从上游 3,533 条中无放回抽取 300 条
- 抽样方法与 300 个样本 ID：`configs/lightweight_subset_manifest.json`
- 完成构图并用于 SFT：237 条；清单与哈希见 `configs/sft_237_manifest.json`
- 剩余构图任务已按轻量复现范围主动停止
- SFT 与 DPO 已完成；GRPO 的步数、batch、序列长度和 rollout 数暂未最终确定
- DPO/GRPO 运行脚本中的数值保持轻量化配置，迁移节点后仍需先看 P2P 状态

重新生成相同子集：

```bash
PYTHONPATH=src python3 scripts/prepare_lightweight_subset.py \
  --input data/raw/light-r1-stage2-3k.json \
  --output data/processed/light-r1-300.jsonl \
  --manifest configs/lightweight_subset_manifest.json \
  --size 300 --seed 42
```

最终 237 条剪枝 SFT 文本的 token 长度为：P50 4,680、P95 12,317、最大
16,944；其中 50 条超过 8,192，2 条超过 tokenizer 标称的 16,384。根据显存探测，
SFT 最终采用 6,144：截断 79 条、保留 79.27% token；8,192 虽能保留
88.93%，但实测峰值达到 29.514/31.72 GiB，安全余量不足。6,144 的实测峰值为
23.842 GiB，每卡保留约 7.88 GiB。

## 数据与训练顺序

先对 300 条数据构图并剪枝：

```bash
python3 scripts/build_graphs.py \
  --input data/processed/light-r1-300.jsonl \
  --output data/processed/graphs-237.jsonl

python3 scripts/build_sft.py \
  --input data/processed/graphs-237.jsonl \
  --output data/processed/scp-sft-237.jsonl \
  --k 2 --m 0.9
```

`build_sft.py` 同时输出平台使用的预格式化 `text` 和 LLaMA-Factory 使用的
`conversations`。预格式化避免 DeepSeek chat template 删除 `<think>...</think>`
内容。正式参数确认后，通过统一脚本启动相应阶段：

```bash
GPU_IDS=0,1,2,3,4,5,6,7 NUM_GPUS=8 \
  bash scripts/run_platform_lightweight.sh sft
```

SFT 分支直接通过 `torchrun` 启动 `scripts/train_sft_ddp.py`。脚本固定走 DDP，
237 条记录全部进入 Dataset；8 卡、每卡 batch 1 时一个 epoch 为 30 个 optimizer
step（DistributedSampler 会补齐 3 个样本）。最终 SFT 参数为 6,144 tokens、每卡
batch 1、全局 batch 8、30 steps、BF16、LoRA r=8/alpha=16/dropout=0.05、学习率
1e-5 和梯度检查点。训练完成时平均 loss 0.5554，耗时 215.44 秒，无 OOM。

本地平台只保存 LoRA adapter，且 DPO/GRPO 不能直接把前一阶段 adapter 当作
基座。因此每阶段结束后必须先合并：

```bash
/mnt/zsr/venvs/lora32/bin/python scripts/merge_lora.py \
  --base-model models/DeepSeek-R1-Distill-Qwen-1.5B \
  --adapter outputs/lightweight/sft-adapter \
  --output outputs/lightweight/sft-merged
```

DPO 偏好数据使用 219 条 original/pruned proxy pairs，不再次调用 API。DPO 数据长度
为：prompt+max(response) P50 5,666、P95 14,409、最大 17,222。3072 token 的
1-step smoke 峰值为 26.887 GiB，但正式 28-step 在第 1 步后 OOM；2048 token
2-step smoke 通过、正式运行在第 4 步 OOM。因此最终采用单卡 1536 token、batch
1、grad accum 1、28 steps、BF16、LoRA r=8/alpha=16/dropout=0.05、学习率
1e-7、beta=0.1。1536 的 10-step smoke 和正式 28-step 均通过，正式平均 loss
0.6872，耗时 67.17 秒，峰值保留显存 25.891 GiB。

本轮当前 shell 的 `hostname` 仍显示旧的 `26-c2d67e`，而 30326 的平台健康文件
指向 `34-c9073b`；手动 `/share/platform/p2pBandwidthTest` 在 45 秒内无输出。
因此 DPO 没有强行启动 8 卡 DDP，而是先完成单卡轻量复现。DPO adapter 保存在
`outputs/lightweight/dpo-adapter-1536`，本地 GRPO 前已合并到
`outputs/lightweight/dpo-merged-1536`。

## 验证状态

本轮已通过 Shell 语法检查、Python compileall、平台训练环境导入检查、8 卡
DDP 启动检查、8,192 与 6,144 token 显存探测、完整 237 条 SFT，以及 1536
token DPO 10-step smoke/28-step 正式训练。SFT adapter SHA-256 为
`caa6b5f6f247c0bb7c3d0edc2d3bfca0af8211c4b6d0598caf367ed4923e1251`；DPO adapter
SHA-256 为 `375c775f25dc5b161034db984451b291a901c2d891f7c949b8424bf171a1afca`。
仓库核心单测在此前运行中为 13 项通过；当前环境没有 pytest，因此本轮未重新执行。
GRPO 仍需做目标配置的显存和端到端检查。
