#!/usr/bin/env python3
"""Native DDP trainer for the lightweight SCP DPO stage."""
import argparse
import json
import os
import sys

PLATFORM_SCRIPTS = os.environ.get("PLATFORM_SCRIPTS", "/share/platform/scripts")
sys.path.insert(0, PLATFORM_SCRIPTS)
import time
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from platform_io import load_records, records_to_preferences
from platform_logging import RunLogger, setup_console_log
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import DPOTrainer



def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--local-rank", "--local_rank", type=int, default=-1)
    p.add_argument("--model-path", required=True)
    p.add_argument("--ref-model", default=None)
    p.add_argument("--dataset", required=True, help="Preference JSONL/JSON/TXT dataset. Needs prompt + chosen + rejected.")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--max-steps", type=int, default=100)
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--prompt-max-len", type=int, default=512)
    p.add_argument("--per-device-batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--loss-type", default="sigmoid", choices=["sigmoid", "hinge", "ipo", "kto_pair"])
    p.add_argument("--finetuning", choices=["full", "lora"], default="lora")
    p.add_argument("--run-name", default=None)
    p.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable gradient checkpointing during training (default: on).",
    )
    p.add_argument("--lora-r", type=int, default=8, help="LoRA rank (PEFT r).")
    p.add_argument("--lora-alpha", type=int, default=16, help="LoRA alpha scaling.")
    p.add_argument("--lora-dropout", type=float, default=0.0, help="LoRA dropout.")
    return p.parse_args()


def main():
    args = parse_args()
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    is_main = rank == 0

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = setup_console_log(str(out_dir))
    run_logger = None
    if is_main:
        run_logger = RunLogger(
            run_name=args.run_name or out_dir.name,
            mode="dpo",
            output_dir=str(out_dir),
            metadata={
                "model_path": args.model_path,
                "ref_model": args.ref_model,
                "dataset": args.dataset,
                "max_steps": args.max_steps,
                "seq_len": args.seq_len,
                "prompt_max_len": args.prompt_max_len,
                "per_device_batch_size": args.per_device_batch_size,
                "grad_accum": args.grad_accum,
                "lr": args.lr,
                "beta": args.beta,
                "finetuning": args.finetuning,
                "lora_r": args.lora_r,
                "lora_alpha": args.lora_alpha,
                "lora_dropout": args.lora_dropout,
                "world_size": world_size,
                "console_log": str(log_path),
            },
        )

    targs = TrainingArguments(
        output_dir=str(out_dir),
        do_train=True,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        max_steps=args.max_steps,
        logging_steps=max(1, min(10, args.max_steps)),
        save_strategy="no",
        evaluation_strategy="no",
        bf16=True,
        report_to=[],
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
        deepspeed=None,
        gradient_checkpointing=args.gradient_checkpointing,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.bos_token_id is None:
        tokenizer.bos_token = tokenizer.eos_token
    tokenizer.model_input_names = [name for name in tokenizer.model_input_names if name != "token_type_ids"]

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model.config.use_cache = False
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    ref_model = None
    if args.finetuning == "full":
        ref_path = args.ref_model or args.model_path
        ref_model = AutoModelForCausalLM.from_pretrained(
            ref_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        ref_model.config.use_cache = False

    peft_config = None
    if args.finetuning == "lora":
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )

    train_rows = records_to_preferences(load_records(args.dataset), tokenizer=tokenizer, add_generation_prompt=True)
    if not train_rows:
        raise ValueError(f"No usable preference records found in {args.dataset}")
    train_ds = Dataset.from_list(train_rows)

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        beta=args.beta,
        loss_type=args.loss_type,
        args=targs,
        train_dataset=train_ds,
        tokenizer=tokenizer,
        max_length=args.seq_len,
        max_prompt_length=args.prompt_max_len,
        peft_config=peft_config,
    )

    if args.gradient_checkpointing and hasattr(trainer.model, "enable_input_require_grads"):
        trainer.model.enable_input_require_grads()

    start = time.time()
    try:
        if run_logger:
            run_logger.event("train_begin", {"rows": len(train_ds)})
        trainer.train()
    except RuntimeError as e:
        msg = str(e).lower()
        if "out of memory" in msg or "cuda out of memory" in msg or "oom" in msg:
            if is_main:
                print("OOM_DETECTED")
                if run_logger:
                    run_logger.finish("oom", {"error": str(e)})
            raise SystemExit(42)
        if run_logger:
            run_logger.finish("failed", {"error": str(e)})
        raise
    elapsed = time.time() - start
    peak_reserved = torch.tensor(
        torch.cuda.max_memory_reserved(), device=torch.cuda.current_device()
    )
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(peak_reserved, op=torch.distributed.ReduceOp.MAX)
    trainer.save_model(str(out_dir))

    if is_main:
        tokenizer.save_pretrained(str(out_dir))
        global_batch = args.per_device_batch_size * args.grad_accum * world_size
        samples = args.max_steps * global_batch
        metrics = {
            "mode": "ddp" if world_size > 1 else "single_gpu",
            "rows": len(train_ds),
            "world_size": world_size,
            "seq_len": args.seq_len,
            "prompt_max_len": args.prompt_max_len,
            "learning_rate": args.lr,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "finetuning": args.finetuning,
            "total_time_s": elapsed,
            "avg_step_s": elapsed / args.max_steps,
            "samples_per_s": samples / elapsed if elapsed > 0 else 0.0,
            "beta": args.beta,
            "max_peak_reserved_gib": round(peak_reserved.item() / 2**30, 3),
        }
        (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2))
        if run_logger:
            run_logger.finish("success", metrics)
        print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    main()
