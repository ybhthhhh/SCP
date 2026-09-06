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
- SFT/DPO/GRPO 的步数、batch、序列长度和 rollout 数暂未最终确定
- 运行脚本中的数值只是保守占位值，正式训练前需用显存探测结果覆盖

重新生成相同子集：

```bash
PYTHONPATH=src python3 scripts/prepare_lightweight_subset.py \
  --input data/raw/light-r1-stage2-3k.json \
  --output data/processed/light-r1-300.jsonl \
  --manifest configs/lightweight_subset_manifest.json \
  --size 300 --seed 42
```

最终 237 条剪枝 SFT 文本的 token 长度为：P50 4,680、P95 12,317、最大
16,944；其中 50 条超过 8,192，2 条超过 tokenizer 标称的 16,384。正式截断长度
需在显存探测后确定。

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
step（DistributedSampler 会补齐 3 个样本）。序列长度等参数仍是显存探测前的占位值。

本地平台只保存 LoRA adapter，且 DPO/GRPO 不能直接把前一阶段 adapter 当作
基座。因此每阶段结束后必须先合并：

```bash
/mnt/zsr/venvs/lora32/bin/python scripts/merge_lora.py \
  --base-model models/DeepSeek-R1-Distill-Qwen-1.5B \
  --adapter outputs/lightweight/sft-adapter \
  --output outputs/lightweight/sft-merged
```

## 验证状态

本轮已通过 Shell 语法检查、Python compileall，以及平台训练环境中的
torch/transformers/peft 导入检查。仓库核心单测在此前运行中为 13 项通过；当前
登录环境没有 pytest，因此本轮未重新执行完整单测。正式训练前还需依次进行：

1. 1 步、短序列、多卡启动检查；
2. 1 步、目标序列长度显存检查；
3. 小批量端到端 SFT 检查；
4. 再确定三个阶段的最终轻量参数。
