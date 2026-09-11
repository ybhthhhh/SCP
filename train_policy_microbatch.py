#!/usr/bin/env python3
import argparse
import base64
import copy
import json
import os
import subprocess
import sys
import tempfile
import time
import socket
import faulthandler
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List, Sequence

# The project-local trainer reuses the platform's support modules without
# modifying the platform installation.
PLATFORM_SCRIPT_DIR = Path("/home/work/platform/scripts")
if str(PLATFORM_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(PLATFORM_SCRIPT_DIR))

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from peft import LoraConfig, PeftModel, get_peft_model, get_peft_model_state_dict
from platform_io import load_records, records_to_prompts
from platform_logging import RunLogger, setup_console_log
from reward_backends import build_reward_backend
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import Cache
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

SCRIPT_DIR = Path(__file__).resolve().parent


class RankEventLogger:
    """Flush one structured event per line so a stuck rank leaves a last marker."""

    def __init__(self, output_dir: Path, rank: int, local_rank: int, world_size: int):
        self.output_dir = output_dir
        self.rank = rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.step = 0
        self.started_at = time.monotonic()
        self.path = output_dir / f"rank{rank}.events.jsonl"
        self.active_operation_path = output_dir / f"rank{rank}.active_operation.json"
        self.stream = self.path.open("a", encoding="utf-8", buffering=1)
        self.stack_stream = (output_dir / f"rank{rank}.stacks.log").open("a", encoding="utf-8")
        faulthandler.enable(file=self.stack_stream, all_threads=True)
        faulthandler.dump_traceback_later(300, repeat=True, file=self.stack_stream, exit=False)
        self.event(
            "process_start",
            {"hostname": socket.gethostname(), "pid": os.getpid(), "cuda_device": local_rank},
        )

    def _gpu_memory(self) -> Dict[str, object]:
        if not torch.cuda.is_available():
            return {}
        try:
            return {
                "gpu_memory_allocated_mb": round(torch.cuda.memory_allocated() / 2**20, 1),
                "gpu_memory_reserved_mb": round(torch.cuda.memory_reserved() / 2**20, 1),
                "gpu_memory_peak_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1),
            }
        except Exception as exc:
            return {"gpu_memory_error": repr(exc)}

    def event(self, name: str, payload: Dict[str, object] | None = None) -> None:
        if payload and "step" in payload:
            self.step = payload["step"]
        row = {
            "timestamp": time.time(),
            "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "event": name,
            "rank": self.rank,
            "local_rank": self.local_rank,
            "world_size": self.world_size,
            "pid": os.getpid(),
            "step": self.step,
            "process_elapsed_s": round(time.monotonic() - self.started_at, 3),
            **self._gpu_memory(),
        }
        if payload:
            row.update(payload)
        self.stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.stream.flush()

    def active_operation(self, name: str, payload: Dict[str, object]) -> None:
        """Atomically preserve the last native call context for fatal crashes."""
        row = {
            "timestamp": time.time(),
            "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "operation": name,
            "status": "active",
            "rank": self.rank,
            "local_rank": self.local_rank,
            "world_size": self.world_size,
            "pid": os.getpid(),
            "step": self.step,
            **self._gpu_memory(),
            **payload,
        }
        temp_path = self.active_operation_path.with_suffix(".tmp")
        with temp_path.open("w", encoding="utf-8") as stream:
            json.dump(row, stream, ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, self.active_operation_path)

    def complete_operation(self, name: str, payload: Dict[str, object]) -> None:
        self.active_operation(name, {"status": "completed", **payload})

    def close(self) -> None:
        faulthandler.cancel_dump_traceback_later()
        faulthandler.disable()
        self.stack_stream.close()
        self.stream.close()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--local_rank", type=int, default=-1)
    p.add_argument("--model-path", required=True)
    p.add_argument("--ref-model", default=None)
    p.add_argument("--prompt-dataset", required=True, help="Prompt JSONL/JSON/TXT dataset.")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--algo", required=True, choices=["ppo", "grpo", "rloo"])
    p.add_argument("--finetuning", choices=["full", "lora"], default="lora")
    p.add_argument("--max-steps", type=int, default=20)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--per-device-batch-size", type=int, default=1)
    p.add_argument(
        "--grad-accum",
        type=int,
        default=1,
        help="Gradient accumulation steps before optimizer.step(). Does NOT reduce GRPO/RLOO "
        "rollout peak memory; lower --num-generations instead.",
    )
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument(
        "--num-generations",
        type=int,
        default=1,
        help="Rollouts per prompt on each GPU. Peak rollout batch per GPU = "
        "per_device_batch_size × num_generations.",
    )
    p.add_argument(
        "--rollout-micro-batch-size",
        type=int,
        default=1,
        help="Maximum number of responses generated at once per GPU.",
    )
    p.add_argument(
        "--train-micro-batch-size",
        type=int,
        default=1,
        help="Maximum number of responses scored in one policy/ref forward pass.",
    )
    p.add_argument("--reward-backend", choices=["function", "rm", "env", "http"], default="function")
    p.add_argument("--reward-spec", default="builtin:length")
    p.add_argument("--rollout-backend", choices=["hf", "vllm"], default="hf")
    p.add_argument("--rollout-visible-devices", default=None, help="Optional CUDA_VISIBLE_DEVICES for the vLLM rollout subprocess.")
    p.add_argument(
        "--generation-cache",
        choices=["dynamic", "static", "none"],
        default="dynamic",
        help=(
            "KV-cache implementation for HF rollouts. 'static' preallocates cache storage instead of "
            "DynamicCache's per-token torch.cat; 'none' disables the cache for diagnosis."
        ),
    )
    p.add_argument(
        "--rollout-attention",
        choices=["eager", "gqa_online"],
        default="eager",
        help=(
            "Attention path for HF rollouts. 'gqa_online' replaces Qwen2's eager "
            "single-token decode attention with grouped online softmax attention."
        ),
    )
    p.add_argument("--kl-coef", type=float, default=0.02)
    p.add_argument(
        "--kl-estimator",
        choices=["k3", "exact"],
        default="k3",
        help="KL penalty estimator: sampled-token k3 (default) or exact full-vocabulary KL.",
    )
    p.add_argument(
        "--kl-vocab-chunk-size",
        type=int,
        default=4096,
        help="Vocabulary chunk size for exact KL; limits temporary [tokens, vocab] tensors.",
    )
    p.add_argument("--ppo-clip-range", type=float, default=0.2)
    p.add_argument("--run-name", default=None)
    p.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable gradient checkpointing during backward (default: on).",
    )
    p.add_argument("--use-deepspeed", action="store_true")
    p.add_argument("--deepspeed-config", default=str(SCRIPT_DIR / "ds_z3_config.json"))
    p.add_argument("--lora-r", type=int, default=8, help="LoRA rank (PEFT r).")
    p.add_argument("--lora-alpha", type=int, default=16, help="LoRA alpha scaling.")
    p.add_argument("--lora-dropout", type=float, default=0.0, help="LoRA dropout.")
    p.add_argument(
        "--checkpoint-every",
        type=int,
        default=0,
        help="Save adapter, optimizer, and per-rank RNG state every N global steps (0 disables periodic checkpoints).",
    )
    p.add_argument(
        "--resume-adapter",
        default=None,
        help="Adapter directory from a previous run or checkpoint to continue from.",
    )
    p.add_argument(
        "--resume-step",
        type=int,
        default=0,
        help="Number of completed global steps before this invocation.",
    )
    p.add_argument(
        "--sync-after-generate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Synchronize the GPU after each generation chunk to attribute async failures.",
    )
    p.add_argument(
        "--empty-cache-after-rollout",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Release unused CUDA allocator blocks after each rollout chunk.",
    )
    return p.parse_args()


def load_json(path: str):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rollout_batch_per_gpu(args) -> int:
    return args.per_device_batch_size * args.num_generations


def log_rollout_memory_hint(args, world_size: int, run_logger: RunLogger | None = None) -> None:
    if args.algo not in {"grpo", "rloo"} or args.num_generations < 2:
        return
    logical_batch = rollout_batch_per_gpu(args)
    rollout_micro_batch = min(args.rollout_micro_batch_size, logical_batch)
    train_micro_batch = min(args.train_micro_batch_size, logical_batch)
    train_batch = logical_batch * args.grad_accum * world_size
    payload = {
        "rollout_batch_per_gpu": logical_batch,
        "rollout_micro_batch_size": rollout_micro_batch,
        "train_micro_batch_size": train_micro_batch,
        "per_device_batch_size": args.per_device_batch_size,
        "num_generations": args.num_generations,
        "grad_accum": args.grad_accum,
        "train_batch_size": train_batch,
        "world_size": world_size,
    }
    if run_logger:
        run_logger.event("rollout_memory_hint", payload)
    print(
        "GRPO/RLOO microbatch: logical rollout batch per GPU = "
        f"per_device_batch_size({args.per_device_batch_size}) × "
        f"num_generations({args.num_generations}) = {logical_batch}; "
        f"generation microbatch={rollout_micro_batch}, "
        f"train microbatch={train_micro_batch}, "
        f"grad_accum={args.grad_accum}, logical train_batch_size={train_batch}.",
        flush=True,
    )


def build_deepspeed_config(args, world_size: int) -> Dict[str, object]:
    config = copy.deepcopy(load_json(args.deepspeed_config))
    micro_batch = rollout_batch_per_gpu(args)
    config["train_micro_batch_size_per_gpu"] = micro_batch
    config["gradient_accumulation_steps"] = args.grad_accum
    config["train_batch_size"] = micro_batch * args.grad_accum * world_size
    zero_config = config.setdefault("zero_optimization", {})
    zero_config["stage"] = 3
    zero_config["stage3_gather_16bit_weights_on_model_save"] = True
    config.setdefault("bf16", {})["enabled"] = True
    return config


def register_zero3_init(config: Dict[str, object]):
    from transformers.integrations import HfDeepSpeedConfig

    return HfDeepSpeedConfig(config)


def load_model(
    model_path: str,
    finetuning: str,
    gradient_checkpointing: bool,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.0,
    resume_adapter: str | None = None,
    move_to_device: torch.device = None,
):
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    model.config.use_cache = False
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
    if finetuning == "lora":
        if resume_adapter:
            adapter_path = Path(resume_adapter)
            if not (adapter_path / "adapter_config.json").is_file():
                raise ValueError(f"--resume-adapter is not a PEFT adapter directory: {adapter_path}")
            # This platform pairs a newer PEFT with Transformers 4.44, whose
            # tensor-parallel helpers do not exist. DDP is not tensor parallel,
            # so adapter tensors must remain unsharded during restoration.
            try:
                import transformers.integrations.tensor_parallel  # noqa: F401
            except ModuleNotFoundError:
                import peft.utils.save_and_load as peft_save_and_load

                peft_save_and_load._maybe_shard_state_dict_for_tp = (  # type: ignore[attr-defined]
                    lambda _model, state_dict, _adapter_name: state_dict
                )
            model = PeftModel.from_pretrained(model, str(adapter_path), is_trainable=True)
        else:
            lora_cfg = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                lora_dropout=lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, lora_cfg)
    if move_to_device is not None:
        model = model.to(move_to_device)
    return model


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def init_distributed(use_deepspeed: bool = False):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        backend = "nccl"
    else:
        device = torch.device("cpu")
        backend = "gloo"
    if world_size > 1 and not dist.is_initialized():
        if use_deepspeed:
            import deepspeed

            deepspeed.init_distributed(dist_backend=backend, auto_mpi_discovery=False)
        else:
            dist.init_process_group(backend=backend)
    return rank, world_size, local_rank, device


def distributed_mean(value: torch.Tensor, world_size: int) -> float:
    tensor = value.detach().float()
    if world_size > 1:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= world_size
    return float(tensor.cpu())


def distributed_extreme(value: float, world_size: int, device: torch.device, op) -> float:
    tensor = torch.tensor(float(value), device=device, dtype=torch.float32)
    if world_size > 1:
        dist.all_reduce(tensor, op=op)
    return float(tensor.cpu())


def normalize_advantages(advantages: torch.Tensor, world_size: int) -> torch.Tensor:
    if advantages.numel() == 0:
        return advantages

    if world_size == 1:
        if advantages.numel() <= 1:
            return advantages
        std = advantages.std(unbiased=False)
        if float(std) > 1e-6:
            advantages = (advantages - advantages.mean()) / std
        return advantages

    local_count = torch.tensor([advantages.numel()], device=advantages.device, dtype=torch.float32)
    local_sum = advantages.sum()
    local_sq_sum = (advantages * advantages).sum()
    dist.all_reduce(local_count, op=dist.ReduceOp.SUM)
    dist.all_reduce(local_sum, op=dist.ReduceOp.SUM)
    dist.all_reduce(local_sq_sum, op=dist.ReduceOp.SUM)
    mean = local_sum / local_count.clamp(min=1.0)
    var = (local_sq_sum / local_count.clamp(min=1.0)) - mean * mean
    std = torch.sqrt(var.clamp(min=0.0))
    if float(std) > 1e-6:
        advantages = (advantages - mean) / std
    return advantages


def sample_next_token(logits: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
    """Match the temperature/top-p portion of HF sampling without GenerationMixin."""
    if temperature <= 0:
        raise ValueError("--temperature must be positive when sampling")
    scores = logits / temperature
    if 0.0 < top_p < 1.0:
        sorted_scores, sorted_indices = torch.sort(scores, descending=True, dim=-1)
        cumulative_probs = torch.softmax(sorted_scores, dim=-1).cumsum(dim=-1)
        remove_sorted = cumulative_probs > top_p
        remove_sorted[..., 1:] = remove_sorted[..., :-1].clone()
        remove_sorted[..., 0] = False
        remove_mask = torch.zeros_like(remove_sorted, dtype=torch.bool).scatter_(
            -1, sorted_indices, remove_sorted
        )
        scores = scores.masked_fill(remove_mask, -float("inf"))
    return torch.multinomial(torch.softmax(scores, dim=-1), num_samples=1).squeeze(-1)


class PreallocatedKVCache(Cache):
    """Fixed K/V storage that exposes only its initialized prefix to Qwen2."""

    def __init__(self, max_cache_len: int):
        super().__init__()
        self.max_cache_len = max_cache_len
        self.key_cache: Dict[int, torch.Tensor] = {}
        self.value_cache: Dict[int, torch.Tensor] = {}
        self.seq_lens: Dict[int, int] = {}
        self._seen_tokens = 0

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]
        start = self.seq_lens.get(layer_idx, 0)
        end = start + key_states.shape[-2]
        if end > self.max_cache_len:
            raise ValueError(f"K/V cache overflow: {end} > {self.max_cache_len}")
        if layer_idx not in self.key_cache:
            shape = (*key_states.shape[:-2], self.max_cache_len, key_states.shape[-1])
            self.key_cache[layer_idx] = key_states.new_empty(shape)
            self.value_cache[layer_idx] = value_states.new_empty(shape)
        self.key_cache[layer_idx][..., start:end, :].copy_(key_states)
        self.value_cache[layer_idx][..., start:end, :].copy_(value_states)
        self.seq_lens[layer_idx] = end
        return self.key_cache[layer_idx][..., :end, :], self.value_cache[layer_idx][..., :end, :]

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self.seq_lens.get(layer_idx, 0)

    def get_max_length(self) -> int:
        return self.max_cache_len

    def __len__(self) -> int:
        return len(self.key_cache)

    def __getitem__(self, layer_idx: int):
        end = self.get_seq_length(layer_idx)
        if not end:
            raise KeyError(f"Cache layer {layer_idx} is uninitialized")
        return self.key_cache[layer_idx][..., :end, :], self.value_cache[layer_idx][..., :end, :]


def grouped_online_attention(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    num_key_value_groups: int,
    block_size: int = 512,
) -> torch.Tensor:
    """GQA attention without repeat_kv or a full-length softmax kernel invocation."""
    batch_size, num_heads, query_length, head_dim = query_states.shape
    num_key_value_heads = key_states.shape[1]
    if num_heads != num_key_value_heads * num_key_value_groups:
        raise ValueError("Qwen2 GQA head configuration is inconsistent")

    query_groups = query_states.float().view(
        batch_size, num_key_value_heads, num_key_value_groups, query_length, head_dim
    )
    key_length = key_states.shape[-2]
    running_max = None
    running_sum = None
    running_value = None
    scale = head_dim**-0.5

    for start in range(0, key_length, block_size):
        end = min(start + block_size, key_length)
        key_block = key_states[..., start:end, :].float().unsqueeze(2)
        value_block = value_states[..., start:end, :].float().unsqueeze(2)
        scores = torch.matmul(query_groups, key_block.transpose(-1, -2)) * scale
        block_max = scores.max(dim=-1, keepdim=True).values

        if running_max is None:
            shifted_scores = torch.exp(scores - block_max)
            running_max = block_max
            running_sum = shifted_scores.sum(dim=-1, keepdim=True)
            running_value = torch.matmul(shifted_scores, value_block)
            continue

        next_max = torch.maximum(running_max, block_max)
        previous_scale = torch.exp(running_max - next_max)
        shifted_scores = torch.exp(scores - next_max)
        running_sum = running_sum * previous_scale + shifted_scores.sum(dim=-1, keepdim=True)
        running_value = running_value * previous_scale + torch.matmul(shifted_scores, value_block)
        running_max = next_max

    if running_value is None or running_sum is None:
        raise ValueError("Grouped attention received an empty K/V cache")
    return (running_value / running_sum).reshape(batch_size, num_heads, query_length, head_dim)


def install_rollout_gqa_attention(model) -> int:
    """Patch only Qwen2's cached, single-token inference forwards used by rollout."""
    patched = 0
    for attention in model.modules():
        if attention.__class__.__name__ != "Qwen2Attention":
            continue
        original_forward = attention.forward

        def rollout_forward(*args, _attention=attention, _original_forward=original_forward, **kwargs):
            hidden_states = kwargs.get("hidden_states", args[0] if args else None)
            attention_mask = kwargs.get("attention_mask", args[1] if len(args) > 1 else None)
            position_ids = kwargs.get("position_ids", args[2] if len(args) > 2 else None)
            past_key_value = kwargs.get("past_key_value", args[3] if len(args) > 3 else None)
            output_attentions = kwargs.get("output_attentions", args[4] if len(args) > 4 else False)
            cache_position = kwargs.get("cache_position", args[6] if len(args) > 6 else None)

            if (
                hidden_states is None
                or _attention.training
                or hidden_states.shape[0] != 1
                or hidden_states.shape[1] != 1
                or output_attentions
                or not isinstance(past_key_value, PreallocatedKVCache)
                or past_key_value.get_seq_length(_attention.layer_idx) == 0
            ):
                return _original_forward(*args, **kwargs)

            query_states = _attention.q_proj(hidden_states)
            key_states = _attention.k_proj(hidden_states)
            value_states = _attention.v_proj(hidden_states)
            query_states = query_states.view(
                1, 1, _attention.num_heads, _attention.head_dim
            ).transpose(1, 2)
            key_states = key_states.view(
                1, 1, _attention.num_key_value_heads, _attention.head_dim
            ).transpose(1, 2)
            value_states = value_states.view(
                1, 1, _attention.num_key_value_heads, _attention.head_dim
            ).transpose(1, 2)

            kv_seq_len = key_states.shape[-2] + past_key_value.get_usable_length(
                key_states.shape[-2], _attention.layer_idx
            )
            cos, sin = _attention.rotary_emb(value_states, seq_len=kv_seq_len)
            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, cos, sin, position_ids
            )
            key_states, value_states = past_key_value.update(
                key_states,
                value_states,
                _attention.layer_idx,
                {"sin": sin, "cos": cos, "cache_position": cache_position},
            )

            # With one valid sequence and q_len=1, every cached token is causally visible.
            del attention_mask
            attn_output = grouped_online_attention(
                query_states,
                key_states,
                value_states,
                _attention.num_key_value_groups,
            )
            attn_output = attn_output.transpose(1, 2).contiguous().reshape(1, 1, _attention.hidden_size)
            attn_output = _attention.o_proj(attn_output.to(query_states.dtype))
            return attn_output, None, past_key_value

        attention.forward = rollout_forward
        patched += 1
    if not patched:
        raise RuntimeError("No Qwen2Attention modules found for rollout attention patch")
    return patched


@torch.no_grad()
def generate_with_manual_static_cache(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    args,
    tokenizer,
    device: torch.device,
) -> torch.Tensor:
    """Autoregressive sampling with a fixed K/V buffer, bypassing HF's cache gate."""
    batch_size, prompt_width = input_ids.shape
    cache = PreallocatedKVCache(max_cache_len=prompt_width + args.max_tokens)
    cache_position = torch.arange(prompt_width, device=device)
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        past_key_values=cache,
        use_cache=True,
        cache_position=cache_position,
        return_dict=True,
    )

    eos_token_ids = tokenizer.eos_token_id
    if isinstance(eos_token_ids, int):
        eos_token_ids = (eos_token_ids,)
    else:
        eos_token_ids = tuple(eos_token_ids or ())
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_token_ids[0]
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
    generated = []
    full_attention_mask = attention_mask

    for token_offset in range(args.max_tokens):
        next_tokens = sample_next_token(outputs.logits[:, -1, :], args.temperature, args.top_p)
        active = ~finished
        next_tokens = torch.where(active, next_tokens, torch.full_like(next_tokens, pad_token_id))
        generated.append(next_tokens)
        if eos_token_ids:
            finished = finished | torch.stack([next_tokens.eq(token_id) for token_id in eos_token_ids]).any(dim=0)
        if bool(finished.all()) or token_offset + 1 == args.max_tokens:
            break

        full_attention_mask = torch.cat([full_attention_mask, active.to(full_attention_mask.dtype).unsqueeze(1)], dim=1)
        outputs = model(
            input_ids=next_tokens.unsqueeze(1),
            attention_mask=full_attention_mask,
            past_key_values=cache,
            use_cache=True,
            cache_position=torch.tensor([prompt_width + token_offset], device=device),
            return_dict=True,
        )

    return torch.cat([input_ids, torch.stack(generated, dim=1)], dim=1)


def rollout_hf(
    model,
    tokenizer,
    prompts: Sequence[str],
    args,
    device: torch.device,
    event_logger: RankEventLogger | None = None,
) -> List[Dict[str, object]]:
    batch_texts = []
    prompt_indices = []
    for idx, prompt in enumerate(prompts):
        for sample_idx in range(args.num_generations):
            batch_texts.append(prompt)
            prompt_indices.append((idx, sample_idx))

    generation_kwargs = {
        "do_sample": True,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "use_cache": args.generation_cache != "none",
    }
    if args.generation_cache == "static":
        generation_kwargs["cache_implementation"] = "manual_static"

    model.eval()
    gc_was_enabled = bool(args.gradient_checkpointing)
    if gc_was_enabled:
        model.gradient_checkpointing_disable()
    prev_use_cache = model.config.use_cache
    model.config.use_cache = args.generation_cache != "none"
    samples = []
    rollout_start = time.time()
    if event_logger:
        event_logger.event(
            "rollout_begin",
            {
                "prompt_count": len(prompts),
                "sample_count": len(batch_texts),
                "generation_cache": args.generation_cache,
            },
        )
    try:
        for start in range(0, len(batch_texts), args.rollout_micro_batch_size):
            chunk_texts = batch_texts[start : start + args.rollout_micro_batch_size]
            chunk_indices = prompt_indices[start : start + args.rollout_micro_batch_size]
            if event_logger:
                event_logger.event(
                    "rollout_chunk_begin",
                    {"start": start, "end": start + len(chunk_texts), "chunk_size": len(chunk_texts)},
                )
            encoded = tokenizer(
                chunk_texts,
                padding=True,
                truncation=True,
                max_length=args.seq_len,
                return_tensors="pt",
            )
            prompt_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)
            prompt_lens = attention_mask.sum(dim=1).tolist()
            chunk_index = start // args.rollout_micro_batch_size
            with torch.no_grad():
                if event_logger:
                    rng_state = base64.b64encode(torch.get_rng_state().numpy().tobytes()).decode("ascii")
                    cuda_rng_state = None
                    if device.type == "cuda":
                        cuda_rng_state = base64.b64encode(
                            torch.cuda.get_rng_state(device).cpu().numpy().tobytes()
                        ).decode("ascii")
                    event_logger.active_operation(
                        "generate",
                        {
                            "chunk_index": chunk_index,
                            "prompt_indices": [list(item) for item in chunk_indices],
                            "prompt_token_ids": prompt_ids.detach().cpu().tolist(),
                            "attention_mask": attention_mask.detach().cpu().tolist(),
                            "generation_kwargs": generation_kwargs,
                            "rollout_attention": args.rollout_attention,
                            "cpu_rng_state_b64": rng_state,
                            "cuda_rng_state_b64": cuda_rng_state,
                        },
                    )
                    event_logger.event("generate_begin", {"chunk_index": chunk_index})
                if args.generation_cache == "static":
                    outputs = generate_with_manual_static_cache(
                        model, prompt_ids, attention_mask, args, tokenizer, device
                    )
                else:
                    outputs = model.generate(
                        input_ids=prompt_ids,
                        attention_mask=attention_mask,
                        **generation_kwargs,
                    )
                if event_logger:
                    event_logger.event("generate_return", {"chunk_index": chunk_index})
                if args.sync_after_generate and device.type == "cuda":
                    if event_logger:
                        event_logger.event("generate_sync_begin", {"chunk_index": chunk_index})
                    torch.cuda.synchronize(device)
                    if event_logger:
                        event_logger.event("generate_sync_end", {"chunk_index": chunk_index})
                if event_logger:
                    event_logger.complete_operation("generate", {"chunk_index": chunk_index})

            if event_logger:
                event_logger.event("decode_begin", {"chunk_index": start // args.rollout_micro_batch_size})
            completion_lengths = []
            hit_max_new_tokens = []
            completion_records = []
            for row_idx, output_ids in enumerate(outputs):
                prompt_len = int(prompt_lens[row_idx])
                prompt_text = chunk_texts[row_idx]
                prompt_token_ids = output_ids[:prompt_len].tolist()
                completion_token_ids = output_ids[prompt_len:].tolist()
                while completion_token_ids and completion_token_ids[-1] == tokenizer.pad_token_id:
                    completion_token_ids.pop()
                completion_lengths.append(len(completion_token_ids))
                hit_max_new_tokens.append(len(completion_token_ids) >= args.max_tokens)
                completion_text = tokenizer.decode(completion_token_ids, skip_special_tokens=True)
                prompt_idx, sample_idx = chunk_indices[row_idx]
                completion_records.append(
                    {
                        "prompt_index": prompt_idx,
                        "sample_index": sample_idx,
                        "prompt_tokens": prompt_len,
                        "completion_tokens": len(completion_token_ids),
                        "hit_max_new_tokens": hit_max_new_tokens[-1],
                        "ended_with_eos": bool(completion_token_ids)
                        and completion_token_ids[-1] == tokenizer.eos_token_id,
                        "completion_token_ids": completion_token_ids,
                        "completion": completion_text,
                    }
                )
                samples.append(
                    {
                        "prompt_index": prompt_idx,
                        "sample_index": sample_idx,
                        "prompt": prompt_text,
                        "completion": completion_text,
                        "prompt_token_ids": prompt_token_ids,
                        "completion_token_ids": completion_token_ids,
                    }
                )
            if event_logger:
                event_logger.event(
                    "rollout_completion_lengths",
                    {
                        "chunk_index": chunk_index,
                        "completion_tokens": completion_lengths,
                        "hit_max_new_tokens": hit_max_new_tokens,
                        "completions": completion_records,
                    },
                )

            del outputs, prompt_ids, attention_mask, encoded
            if args.empty_cache_after_rollout and device.type == "cuda":
                torch.cuda.empty_cache()
            if event_logger:
                event_logger.event(
                    "rollout_chunk_end",
                    {
                        "completed": min(start + args.rollout_micro_batch_size, len(batch_texts)),
                        "elapsed_s": round(time.time() - rollout_start, 3),
                    },
                )
            if os.environ.get("RANK", "0") == "0":
                completed = min(start + args.rollout_micro_batch_size, len(batch_texts))
                print(
                    f"rollout_progress={completed}/{len(batch_texts)} "
                    f"elapsed_s={time.time() - rollout_start:.1f}",
                    flush=True,
                )
    finally:
        model.config.use_cache = prev_use_cache
        if gc_was_enabled:
            model.gradient_checkpointing_enable()
        if event_logger:
            event_logger.event(
                "rollout_end",
                {"samples": len(samples), "elapsed_s": round(time.time() - rollout_start, 3)},
            )
    return samples


def rollout_vllm(prompts: Sequence[str], args) -> List[Dict[str, object]]:
    with tempfile.TemporaryDirectory(prefix="platform_rl_rollout_") as tmpdir:
        tmpdir_path = Path(tmpdir)
        prompt_file = tmpdir_path / "prompts.txt"
        output_file = tmpdir_path / "outputs.jsonl"
        prompt_file.write_text("\n".join(prompts) + "\n", encoding="utf-8")
        cmd = [
            "/mnt/zsr/venvs/infer/bin/python",
            str(Path(__file__).resolve().parent / "rl_rollout_vllm.py"),
            "--model",
            args.model_path,
            "--input-file",
            str(prompt_file),
            "--output-file",
            str(output_file),
            "--max-tokens",
            str(args.max_tokens),
            "--max-model-len",
            str(args.seq_len + args.max_tokens),
            "--temperature",
            str(args.temperature),
            "--top-p",
            str(args.top_p),
            "--tensor-parallel-size",
            "1",
            "--num-generations",
            str(args.num_generations),
            "--input-is-rendered",
        ]
        env = os.environ.copy()
        if args.rollout_visible_devices:
            env["CUDA_VISIBLE_DEVICES"] = args.rollout_visible_devices
        subprocess.check_call(cmd, env=env)
        rows = [json.loads(line) for line in output_file.read_text(encoding="utf-8").splitlines() if line.strip()]

    samples = []
    for prompt_idx, row in enumerate(rows):
        for sample_idx, completion in enumerate(row["completions"]):
            samples.append(
                {
                    "prompt_index": prompt_idx,
                    "sample_index": sample_idx,
                    "prompt": row["prompt"],
                    "completion": completion["text"],
                    "prompt_token_ids": row.get("prompt_token_ids", []),
                    "completion_token_ids": completion.get("token_ids", []),
                }
            )
    return samples


def pad_samples(samples: Sequence[Dict[str, object]], pad_token_id: int, device: torch.device):
    full_sequences = []
    completion_masks = []
    for sample in samples:
        prompt_ids = list(sample["prompt_token_ids"])
        completion_ids = list(sample["completion_token_ids"])
        if not completion_ids:
            completion_ids = [pad_token_id]
        full = prompt_ids + completion_ids
        if len(full) < 2:
            full = full + [pad_token_id]
            completion_ids = completion_ids + [pad_token_id]
        full_sequences.append(full)
        completion_masks.append([0] * len(prompt_ids) + [1] * len(completion_ids))

    max_len = max(len(seq) for seq in full_sequences)
    input_ids = []
    attention_mask = []
    completion_mask = []
    for seq, mask in zip(full_sequences, completion_masks):
        pad_len = max_len - len(seq)
        input_ids.append(seq + [pad_token_id] * pad_len)
        attention_mask.append([1] * len(seq) + [0] * pad_len)
        completion_mask.append(mask + [0] * pad_len)
    return (
        torch.tensor(input_ids, device=device, dtype=torch.long),
        torch.tensor(attention_mask, device=device, dtype=torch.long),
        torch.tensor(completion_mask, device=device, dtype=torch.float32),
    )


def sequence_logprobs(model, input_ids, attention_mask, completion_mask):
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    logits = outputs.logits[:, :-1, :]
    target_ids = input_ids[:, 1:]
    mask = completion_mask[:, 1:]
    log_probs = F.log_softmax(logits, dim=-1)
    selected = torch.gather(log_probs, dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)
    denom = mask.sum(dim=-1).clamp(min=1.0)
    seq_logprob = (selected * mask).sum(dim=-1) / denom
    return seq_logprob, log_probs, mask


def sequence_logprob_for_kl(model, input_ids, attention_mask, completion_mask, retain_logits: bool):
    """Return sequence log-probability and either sampled-token scores or full logits."""
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    logits = outputs.logits[:, :-1, :]
    target_ids = input_ids[:, 1:]
    mask = completion_mask[:, 1:]
    log_normalizer = torch.logsumexp(logits, dim=-1)
    selected = torch.gather(logits, dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1) - log_normalizer
    denom = mask.sum(dim=-1).clamp(min=1.0)
    seq_logprob = (selected * mask).sum(dim=-1) / denom
    return seq_logprob, logits if retain_logits else selected, mask


def forward_kl(current_log_probs, ref_log_probs, mask):
    current_probs = current_log_probs.exp()
    token_kl = (current_probs * (current_log_probs - ref_log_probs)).sum(dim=-1)
    return (token_kl * mask).sum() / mask.sum().clamp(min=1.0)


def sampled_token_k3_kl(
    current_token_logprobs: torch.Tensor,
    ref_token_logprobs: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Estimate KL(policy || reference) from rollout actions using the k3 form."""
    log_ratio = ref_token_logprobs - current_token_logprobs
    token_kl = torch.exp(log_ratio) - log_ratio - 1.0
    return (token_kl * mask).sum() / mask.sum().clamp(min=1.0)


def forward_kl_from_logits_chunked(
    current_logits: torch.Tensor,
    ref_logits: torch.Tensor,
    mask: torch.Tensor,
    vocab_chunk_size: int,
) -> torch.Tensor:
    """Exact forward KL without materializing full-vocabulary temporary differences."""
    if vocab_chunk_size < 1:
        raise ValueError("--kl-vocab-chunk-size must be positive")
    if current_logits.shape != ref_logits.shape:
        raise ValueError("Policy and reference logits must have matching shapes for KL")

    current_normalizer = torch.logsumexp(current_logits, dim=-1, keepdim=True)
    ref_normalizer = torch.logsumexp(ref_logits, dim=-1, keepdim=True)
    total_kl = current_logits.new_zeros((), dtype=torch.float32)
    vocab_size = current_logits.shape[-1]
    for start in range(0, vocab_size, vocab_chunk_size):
        end = min(start + vocab_chunk_size, vocab_size)
        current_log_probs = current_logits[..., start:end] - current_normalizer
        ref_log_probs = ref_logits[..., start:end] - ref_normalizer
        chunk_kl = (current_log_probs.exp() * (current_log_probs - ref_log_probs)).sum(dim=-1)
        total_kl = total_kl + (chunk_kl * mask).sum().float()
    return total_kl / mask.sum().clamp(min=1.0)


def optimizer_zero_grad(optimizer, use_deepspeed: bool):
    if optimizer is not None and not use_deepspeed:
        optimizer.zero_grad()


def backward_step(model, optimizer, loss, use_deepspeed: bool):
    if use_deepspeed:
        model.backward(loss)
        model.step()
    else:
        loss.backward()
        optimizer.step()


def backward_only(model, loss, use_deepspeed: bool):
    if use_deepspeed:
        model.backward(loss)
    else:
        loss.backward()


def optimizer_step(model, optimizer, use_deepspeed: bool):
    if use_deepspeed:
        model.step()
    else:
        optimizer.step()


def save_policy_model(train_model, tokenizer, output_dir: Path, finetuning: str, use_deepspeed: bool, is_main: bool):
    if use_deepspeed:
        module = unwrap_model(train_model)
        if finetuning == "lora":
            state_dict = train_model._zero3_consolidated_16bit_state_dict()
            if is_main:
                lora_state = get_peft_model_state_dict(module, state_dict=state_dict)
                output_dir.mkdir(parents=True, exist_ok=True)
                module.save_pretrained(
                    str(output_dir),
                    state_dict=lora_state,
                    is_main_process=True,
                    safe_serialization=True,
                )
                tokenizer.save_pretrained(str(output_dir))
            return

        train_model.save_16bit_model(str(output_dir), save_filename="pytorch_model.bin")
        if is_main:
            module.config.save_pretrained(str(output_dir))
            if getattr(module, "generation_config", None) is not None:
                module.generation_config.save_pretrained(str(output_dir))
            tokenizer.save_pretrained(str(output_dir))
        return

    if is_main:
        unwrap_model(train_model).save_pretrained(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))


def checkpoint_path(output_dir: Path, step: int) -> Path:
    return output_dir / "checkpoints" / f"step-{step:06d}"


def save_training_checkpoint(
    train_model,
    optimizer,
    tokenizer,
    output_dir: Path,
    step: int,
    finetuning: str,
    use_deepspeed: bool,
    is_main: bool,
    rank: int,
    device: torch.device,
) -> Path:
    checkpoint_dir = checkpoint_path(output_dir, step)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    save_policy_model(train_model, tokenizer, checkpoint_dir, finetuning, use_deepspeed, is_main)
    rng_state = {"cpu": torch.get_rng_state()}
    if device.type == "cuda":
        rng_state["cuda"] = torch.cuda.get_rng_state(device)
    torch.save(rng_state, checkpoint_dir / f"rng_rank{rank}.pt")
    if is_main and not use_deepspeed:
        torch.save(
            {"completed_steps": step, "optimizer": optimizer.state_dict()},
            checkpoint_dir / "trainer_state.pt",
        )
        (checkpoint_dir / "checkpoint_complete.json").write_text(
            json.dumps({"completed_steps": step, "world_size": int(os.environ.get("WORLD_SIZE", "1"))}),
            encoding="utf-8",
        )
    return checkpoint_dir


def restore_training_state(optimizer, resume_adapter: str | None, rank: int, device: torch.device) -> int | None:
    if not resume_adapter:
        return None
    checkpoint_dir = Path(resume_adapter)
    state_path = checkpoint_dir / "trainer_state.pt"
    if state_path.is_file():
        state = torch.load(state_path, map_location="cpu")
        optimizer.load_state_dict(state["optimizer"])
        completed_steps = int(state["completed_steps"])
    else:
        completed_steps = None
    rng_path = checkpoint_dir / f"rng_rank{rank}.pt"
    if rng_path.is_file():
        rng_state = torch.load(rng_path, map_location="cpu")
        torch.set_rng_state(rng_state["cpu"])
        if device.type == "cuda" and "cuda" in rng_state:
            torch.cuda.set_rng_state(rng_state["cuda"], device)
    return completed_steps


def compute_advantages(algo: str, rewards: List[float], num_generations: int) -> torch.Tensor:
    rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
    if algo == "ppo":
        return rewards_tensor
    if num_generations < 2:
        raise ValueError(f"{algo} requires --num-generations >= 2")
    grouped = rewards_tensor.view(-1, num_generations)
    if algo == "grpo":
        mean = grouped.mean(dim=1, keepdim=True)
        std = grouped.std(dim=1, keepdim=True, unbiased=False).clamp(min=1e-6)
        return ((grouped - mean) / std).reshape(-1)
    if algo == "rloo":
        sums = grouped.sum(dim=1, keepdim=True)
        baseline = (sums - grouped) / (num_generations - 1)
        return (grouped - baseline).reshape(-1)
    raise ValueError(f"Unsupported algo {algo}")


def main():
    args = parse_args()
    rank, world_size, local_rank, device = init_distributed(args.use_deepspeed)
    is_main = rank == 0
    if args.algo in {"grpo", "rloo"} and args.num_generations < 2:
        raise SystemExit(f"{args.algo} requires --num-generations >= 2")
    if args.rollout_micro_batch_size < 1 or args.train_micro_batch_size < 1:
        raise SystemExit("micro-batch sizes must be positive")
    if args.rollout_attention == "gqa_online" and args.generation_cache != "static":
        raise SystemExit("--rollout-attention gqa_online requires --generation-cache static")
    if world_size > 1 and args.rollout_backend != "hf":
        raise SystemExit(f"{args.algo} multi-GPU currently supports only --rollout-backend hf")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rank_logger = RankEventLogger(out_dir, rank, local_rank, world_size)
    log_path = setup_console_log(str(out_dir))
    run_logger = None
    if is_main:
        run_logger = RunLogger(
            run_name=args.run_name or out_dir.name,
            mode=args.algo,
            output_dir=str(out_dir),
            metadata={
                "model_path": args.model_path,
                "ref_model": args.ref_model,
                "prompt_dataset": args.prompt_dataset,
                "max_steps": args.max_steps,
                "seq_len": args.seq_len,
                "per_device_batch_size": args.per_device_batch_size,
                "grad_accum": args.grad_accum,
                "lr": args.lr,
                "algo": args.algo,
                "finetuning": args.finetuning,
                "reward_backend": args.reward_backend,
                "reward_spec": args.reward_spec,
                "rollout_backend": args.rollout_backend,
                "generation_cache": args.generation_cache,
                "rollout_attention": args.rollout_attention,
                "num_generations": args.num_generations,
                "rollout_micro_batch_size": args.rollout_micro_batch_size,
                "train_micro_batch_size": args.train_micro_batch_size,
                "kl_coef": args.kl_coef,
                "kl_estimator": args.kl_estimator,
                "kl_vocab_chunk_size": args.kl_vocab_chunk_size,
                "use_deepspeed": args.use_deepspeed,
                "deepspeed_config": args.deepspeed_config if args.use_deepspeed else None,
                "world_size": world_size,
                "lora_r": args.lora_r,
                "lora_alpha": args.lora_alpha,
                "lora_dropout": args.lora_dropout,
                "console_log": str(log_path),
            },
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompts = records_to_prompts(load_records(args.prompt_dataset), tokenizer=tokenizer, add_generation_prompt=True)
    if not prompts:
        raise ValueError(f"No usable prompts found in {args.prompt_dataset}")

    ds_config = build_deepspeed_config(args, world_size) if args.use_deepspeed else None
    log_rollout_memory_hint(args, world_size, run_logger)
    ds_zero3_config = register_zero3_init(ds_config) if args.use_deepspeed else None
    policy_model = load_model(
        args.model_path,
        args.finetuning,
        args.gradient_checkpointing,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        resume_adapter=args.resume_adapter,
        move_to_device=None if args.use_deepspeed else device,
    )
    if args.rollout_attention == "gqa_online":
        patched_layers = install_rollout_gqa_attention(policy_model)
        rank_logger.event(
            "rollout_attention_patch_installed",
            {"implementation": args.rollout_attention, "patched_layers": patched_layers},
        )
    train_model = policy_model
    optimizer = torch.optim.AdamW((param for param in policy_model.parameters() if param.requires_grad), lr=args.lr)
    if args.use_deepspeed:
        import deepspeed

        _ = ds_zero3_config
        train_model, optimizer, _, _ = deepspeed.initialize(
            model=policy_model,
            optimizer=optimizer,
            model_parameters=(param for param in policy_model.parameters() if param.requires_grad),
            config=ds_config,
            dist_init_required=False,
        )
        device = train_model.device
    elif world_size > 1:
        train_model = DDP(
            policy_model,
            device_ids=[local_rank] if device.type == "cuda" else None,
            output_device=local_rank if device.type == "cuda" else None,
            broadcast_buffers=False,
            find_unused_parameters=False,
        )

    restored_steps = restore_training_state(optimizer, args.resume_adapter, rank, device)
    if restored_steps is not None:
        if args.resume_step not in {0, restored_steps}:
            raise ValueError(
                f"--resume-step ({args.resume_step}) disagrees with checkpoint ({restored_steps})"
            )
        args.resume_step = restored_steps

    ref_model = AutoModelForCausalLM.from_pretrained(
        args.ref_model or args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    if args.use_deepspeed:
        import deepspeed

        ref_config = copy.deepcopy(ds_config)
        ref_config["train_batch_size"] = ds_config["train_micro_batch_size_per_gpu"] * world_size
        ref_config["gradient_accumulation_steps"] = 1
        ref_model, _, _, _ = deepspeed.initialize(
            model=ref_model,
            config=ref_config,
            dist_init_required=False,
        )
        device = train_model.device
    else:
        ref_model = ref_model.to(device)
    ref_model.eval()

    reward_backend = build_reward_backend(args.reward_backend, args.reward_spec, device=str(device), max_length=args.seq_len + args.max_tokens)

    history = []
    train_start = time.time()
    effective_rollout_backend = args.rollout_backend
    global_batch_size = args.per_device_batch_size * world_size
    if run_logger:
        run_logger.event("train_begin", {"rows": len(prompts), "world_size": world_size})
    rank_logger.event("train_begin", {"rows": len(prompts)})

    try:
        for local_step in range(args.max_steps):
            step = args.resume_step + local_step
            batch_prompts = []
            base_index = step * global_batch_size + rank * args.per_device_batch_size
            for offset in range(args.per_device_batch_size):
                batch_prompts.append(prompts[(base_index + offset) % len(prompts)])

            step_start = time.time()
            rank_logger.event(
                "step_begin",
                {"step": step + 1, "prompt_count": len(batch_prompts)},
            )
            if args.rollout_backend == "hf":
                samples = rollout_hf(
                    unwrap_model(train_model), tokenizer, batch_prompts, args, device, rank_logger
                )
            else:
                try:
                    samples = rollout_vllm(batch_prompts, args)
                    effective_rollout_backend = "vllm"
                except Exception as exc:
                    effective_rollout_backend = "hf_fallback"
                    if run_logger:
                        run_logger.event(
                            "rollout_backend_fallback",
                            {
                                "requested": "vllm",
                                "fallback": "hf",
                                "error": repr(exc),
                            },
                        )
                    print(f"vllm rollout failed, falling back to hf: {exc}")
                    samples = rollout_hf(
                        unwrap_model(train_model), tokenizer, batch_prompts, args, device, rank_logger
                    )
                for sample in samples:
                    if not sample["prompt_token_ids"]:
                        sample["prompt_token_ids"] = tokenizer(sample["prompt"], add_special_tokens=True)["input_ids"]
                    if not sample["completion_token_ids"]:
                        sample["completion_token_ids"] = tokenizer(sample["completion"], add_special_tokens=False)["input_ids"]

            rank_logger.event("rollout_complete", {"step": step + 1, "samples": len(samples)})
            rank_logger.event("reward_begin", {"step": step + 1, "samples": len(samples)})
            rewards = reward_backend.score(samples)
            rank_logger.event(
                "reward_end",
                {
                    "step": step + 1,
                    "samples": len(samples),
                    "mean_reward_local": sum(rewards) / len(rewards),
                },
            )
            rank_logger.event("advantage_begin", {"step": step + 1})
            advantages_tensor = compute_advantages(args.algo, rewards, args.num_generations).to(device=device)
            rank_logger.event("advantage_allreduce_begin")
            advantages_tensor = normalize_advantages(advantages_tensor, world_size)
            rank_logger.event("advantage_allreduce_end")
            rank_logger.event("advantage_end", {"step": step + 1})
            for sample, reward, advantage in zip(samples, rewards, advantages_tensor.detach().cpu().tolist()):
                sample["reward"] = reward
                sample["advantage"] = advantage

            sample_count = len(samples)
            train_micro_batch_size = min(max(1, args.train_micro_batch_size), sample_count)
            old_seq_logprobs = []

            unwrap_model(train_model).eval()
            with torch.no_grad():
                rank_logger.event("old_logprobs_begin", {"step": step + 1, "samples": sample_count})
                for start in range(0, sample_count, train_micro_batch_size):
                    end = min(start + train_micro_batch_size, sample_count)
                    rank_logger.event(
                        "old_logprobs_chunk_begin",
                        {
                            "step": step + 1,
                            "chunk_index": start // train_micro_batch_size,
                            "start": start,
                            "end": end,
                        },
                    )
                    chunk = samples[start:end]
                    input_ids, attention_mask, completion_mask = pad_samples(
                        chunk, tokenizer.pad_token_id, device
                    )
                    old_chunk, _, _ = sequence_logprobs(
                        unwrap_model(train_model), input_ids, attention_mask, completion_mask
                    )
                    old_seq_logprobs.append(old_chunk.detach().cpu())
                    del input_ids, attention_mask, completion_mask, old_chunk
                    rank_logger.event(
                        "old_logprobs_chunk_end",
                        {
                            "step": step + 1,
                            "chunk_index": start // train_micro_batch_size,
                            "start": start,
                            "end": end,
                        },
                    )
                rank_logger.event("old_logprobs_end", {"step": step + 1})

            train_model.train()
            optimizer_zero_grad(optimizer, args.use_deepspeed)
            loss_total = 0.0
            policy_loss_total = 0.0
            kl_total = 0.0
            for chunk_index, start in enumerate(range(0, sample_count, train_micro_batch_size)):
                end = min(start + train_micro_batch_size, sample_count)
                chunk = samples[start:end]
                rank_logger.event(
                    "train_microbatch_begin",
                    {"step": step + 1, "chunk_index": chunk_index, "start": start, "end": end},
                )
                input_ids, attention_mask, completion_mask = pad_samples(
                    chunk, tokenizer.pad_token_id, device
                )
                old_seq_logprob = old_seq_logprobs[chunk_index].to(device=device)
                weight = float(end - start) / float(sample_count)
                sync_context = nullcontext()
                should_sync = True
                if (
                    world_size > 1
                    and not args.use_deepspeed
                    and hasattr(train_model, "no_sync")
                    and end < sample_count
                ):
                    sync_context = train_model.no_sync()
                    should_sync = False
                # DDP decides whether to reduce gradients during its forward pass.
                with sync_context:
                    rank_logger.event("policy_forward_begin", {"chunk_index": chunk_index})
                    exact_kl = args.kl_estimator == "exact"
                    new_seq_logprob, new_kl_state, mask = sequence_logprob_for_kl(
                        train_model,
                        input_ids,
                        attention_mask,
                        completion_mask,
                        retain_logits=exact_kl,
                    )
                    rank_logger.event("policy_forward_end", {"chunk_index": chunk_index})
                    rank_logger.event("ref_forward_begin", {"chunk_index": chunk_index})
                    with torch.no_grad():
                        _, ref_kl_state, _ = sequence_logprob_for_kl(
                            ref_model,
                            input_ids,
                            attention_mask,
                            completion_mask,
                            retain_logits=exact_kl,
                        )
                    rank_logger.event("ref_forward_end", {"chunk_index": chunk_index})
                    ratio = torch.exp(new_seq_logprob - old_seq_logprob)
                    advantage_chunk = advantages_tensor[start:end].to(dtype=new_seq_logprob.dtype)
                    clipped_ratio = torch.clamp(
                        ratio, 1.0 - args.ppo_clip_range, 1.0 + args.ppo_clip_range
                    )
                    pg_loss = -torch.minimum(
                        ratio * advantage_chunk, clipped_ratio * advantage_chunk
                    ).mean()
                    if exact_kl:
                        kl_value = forward_kl_from_logits_chunked(
                            new_kl_state, ref_kl_state, mask, args.kl_vocab_chunk_size
                        )
                    else:
                        kl_value = sampled_token_k3_kl(new_kl_state, ref_kl_state, mask)
                    loss = pg_loss + args.kl_coef * kl_value
                    rank_logger.event(
                        "backward_begin", {"chunk_index": chunk_index, "ddp_sync": should_sync}
                    )
                    backward_only(train_model, loss * weight, args.use_deepspeed)
                    rank_logger.event("backward_end", {"chunk_index": chunk_index})
                rank_logger.event(
                    "train_microbatch_end",
                    {
                        "step": step + 1,
                        "chunk_index": chunk_index,
                        "start": start,
                        "end": end,
                        "loss": float(loss.detach()),
                        "ddp_sync": should_sync,
                    },
                )
                loss_total += float(loss.detach()) * weight
                policy_loss_total += float(pg_loss.detach()) * weight
                kl_total += float(kl_value.detach()) * weight
                del (
                    input_ids,
                    attention_mask,
                    completion_mask,
                    old_seq_logprob,
                    new_seq_logprob,
                    new_kl_state,
                    ref_kl_state,
                    mask,
                    ratio,
                    advantage_chunk,
                    clipped_ratio,
                    pg_loss,
                    kl_value,
                    loss,
                )
            rank_logger.event("optimizer_step_begin", {"step": step + 1})
            optimizer_step(train_model, optimizer, args.use_deepspeed)
            rank_logger.event("optimizer_step_end", {"step": step + 1})
            loss = torch.tensor(loss_total, device=device)
            pg_loss = torch.tensor(policy_loss_total, device=device)
            kl_value = torch.tensor(kl_total, device=device)

            rank_logger.event("metrics_allreduce_begin")
            step_metrics = {
                "step": step + 1,
                "loss": distributed_mean(loss, world_size),
                "policy_loss": distributed_mean(pg_loss, world_size),
                "kl": distributed_mean(kl_value, world_size),
                "mean_reward": distributed_mean(torch.tensor(sum(rewards) / len(rewards), device=device), world_size),
                "max_reward": distributed_extreme(max(rewards), world_size, device, dist.ReduceOp.MAX),
                "min_reward": distributed_extreme(min(rewards), world_size, device, dist.ReduceOp.MIN),
                "step_time_s": time.time() - step_start,
                "rollout_backend": effective_rollout_backend,
                "world_size": world_size,
            }
            rank_logger.event("metrics_allreduce_end")
            if is_main:
                history.append(step_metrics)
                run_logger.event("policy_step", step_metrics)
                print(json.dumps(step_metrics, ensure_ascii=False))
            rank_logger.event("step_end", {**step_metrics, "step_elapsed_s": time.time() - step_start})
            if args.checkpoint_every and (step + 1) % args.checkpoint_every == 0:
                rank_logger.event("checkpoint_begin", {"step": step + 1})
                saved_checkpoint = save_training_checkpoint(
                    train_model,
                    optimizer,
                    tokenizer,
                    out_dir,
                    step + 1,
                    args.finetuning,
                    args.use_deepspeed,
                    is_main,
                    rank,
                    device,
                )
                rank_logger.event("checkpoint_end", {"step": step + 1, "path": str(saved_checkpoint)})
    except RuntimeError as e:
        rank_logger.event("exception", {"exception_type": type(e).__name__, "error": str(e)})
        msg = str(e).lower()
        if "out of memory" in msg or "cuda out of memory" in msg or "oom" in msg:
            if is_main:
                print("OOM_DETECTED")
                if run_logger:
                    run_logger.finish("oom", {"error": str(e)})
            raise SystemExit(42)
        if run_logger:
            run_logger.finish("failed", {"error": str(e)})
        rank_logger.close()
        raise
    except Exception as e:
        rank_logger.event("exception", {"exception_type": type(e).__name__, "error": repr(e)})
        if is_main and run_logger:
            run_logger.finish("failed", {"error": repr(e)})
        rank_logger.close()
        raise

    rank_logger.event("save_begin")
    elapsed = time.time() - train_start
    save_policy_model(train_model, tokenizer, out_dir, args.finetuning, args.use_deepspeed, is_main)
    rank_logger.event("save_end")
    if is_main:
        metrics = {
            "mode": args.algo,
            "finetuning": args.finetuning,
            "requested_rollout_backend": args.rollout_backend,
            "effective_rollout_backend": effective_rollout_backend,
            "reward_backend": args.reward_backend,
            "distributed_backend": "deepspeed_zero3" if args.use_deepspeed else ("ddp" if world_size > 1 else "single"),
            "world_size": world_size,
            "global_batch_size": global_batch_size,
            "total_time_s": elapsed,
            "avg_step_s": elapsed / args.max_steps,
            "final_mean_reward": history[-1]["mean_reward"] if history else 0.0,
            "final_kl": history[-1]["kl"] if history else 0.0,
        }
        (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2))
        run_logger.finish("success", metrics)
        print(json.dumps(metrics, ensure_ascii=False, indent=2))

    rank_logger.event("process_end", {"status": "success"})
    rank_logger.close()
    if world_size > 1 and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    main()
