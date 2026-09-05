# 1.5B 轻量复现说明

本配置只复现 SCP 的最小可运行链路，不追求论文训练规模。训练数据、模型和
checkpoint 保留在本地，不提交 GitHub。

## 本机资源策略

本机有 8 张 32 GiB DT1000。GPU 0–3 位于 NUMA 0，GPU 4–7 位于 NUMA 1；
跨组通信经过 SYS。默认使用同一 NUMA 的 GPU 0,1,2,3，并在多卡训练启动前运行
`/share/platform/p2pBandwidthTest`。检查失败时脚本会停止，不冒险启动训练。

卡数不是论文超参数。若后续显存探测表明两卡已经足够，可设置
`GPU_IDS=0,1 NUM_GPUS=2`；不要用跨 NUMA 的组合。

## 当前已确定的范围

- 基座：本地 `DeepSeek-R1-Distill-Qwen-1.5B`
- SFT：固定种子 42，从上游 3,533 条中无放回抽取 300 条
- 抽样方法与 300 个样本 ID：`configs/lightweight_subset_manifest.json`
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

这 300 条未剪枝轨迹使用本地 tokenizer 的长度统计为：P50 643、P95 963、
最大 1213 tokens，没有样本超过 2048 tokens。

## 数据与训练顺序

先对 300 条数据构图并剪枝：

```bash
python3 scripts/build_graphs.py \
  --input data/processed/light-r1-300.jsonl \
  --output data/processed/graphs-300.jsonl

python3 scripts/build_sft.py \
  --input data/processed/graphs-300.jsonl \
  --output data/processed/scp-sft-300.jsonl \
  --k 2 --m 0.9
```

`build_sft.py` 同时输出平台使用的 `messages` 和 LLaMA-Factory 使用的
`conversations`。正式参数确认后，通过统一脚本启动相应阶段：

```bash
GPU_IDS=0,1,2,3 NUM_GPUS=4 \
  bash scripts/run_platform_lightweight.sh sft
```

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
