#!/usr/bin/env python3
"""Native DDP trainer for the 237-sample lightweight SCP SFT run."""
import argparse
import json
import os
import sys
import time

PLATFORM_SCRIPTS = os.environ.get("PLATFORM_SCRIPTS", "/share/platform/scripts")
sys.path.insert(0, PLATFORM_SCRIPTS)
from pathlib import Path

import torch
from datasets import Dataset
from platform_io import load_records, records_to_texts
from platform_logging import RunLogger, setup_console_log
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


def build_dataset(tokenizer, seq_len: int, total_samples: int, dataset_path: str = None) -> Dataset:
    if dataset_path:
        texts = records_to_texts(load_records(dataset_path), tokenizer=tokenizer, add_generation_prompt=False)
        if not texts:
            raise ValueError(f"No usable records found in {dataset_path}")
    else:
        texts = [
            "Explain what LoRA is in one short sentence.",
            "Summarize distributed training in one sentence.",
        ]
    rows = []
    sample_count = len(texts) if dataset_path else total_samples
    for i in range(sample_count):
        txt = texts[i % len(texts)]
        ids = tokenizer(
            txt,
            truncation=True,
            max_length=seq_len,
            padding="max_length",
            return_tensors=None,
        )
        rows.append({"input_ids": ids["input_ids"], "attention_mask": ids["attention_mask"]})
    return Dataset.from_list(rows)


def parse_args():
    p = argparse.ArgumentParser()
    # torchrun injects --local_rank; Trainer reads the distributed environment.
    p.add_argument("--local-rank", "--local_rank", type=int, default=-1)
    p.add_argument("--model-path", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--max-steps", type=int, default=100)
    p.add_argument("--seq-len", type=int, default=64)
    p.add_argument("--per-device-batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--dataset", default=None, help="JSONL/JSON/TXT dataset. See README.md for schema.")
    p.add_argument("--run-name", default=None)
    p.add_argument("--no-lora", action="store_true", help="Smoke-test base model only (no PEFT).")
    p.add_argument("--lora-r", type=int, default=8, help="LoRA rank (PEFT r).")
    p.add_argument("--lora-alpha", type=int, default=16, help="LoRA alpha scaling.")
    p.add_argument("--lora-dropout", type=float, default=0.0, help="LoRA dropout.")
    p.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable gradient checkpointing during training (default: on).",
    )
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
            mode="lora",
            output_dir=str(out_dir),
            metadata={
                "model_path": args.model_path,
                "dataset": args.dataset,
                "max_steps": args.max_steps,
                "seq_len": args.seq_len,
                "per_device_batch_size": args.per_device_batch_size,
                "grad_accum": args.grad_accum,
                "lr": args.lr,
                "lora_r": args.lora_r,
                "lora_alpha": args.lora_alpha,
                "lora_dropout": args.lora_dropout,
                "gradient_checkpointing": args.gradient_checkpointing,
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
        logging_steps=10,
        save_strategy="no",
        evaluation_strategy="no",
        bf16=True,
        report_to=[],
        ddp_find_unused_parameters=False,
        gradient_checkpointing=args.gradient_checkpointing,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model.config.use_cache = False
    if not args.no_lora:
        model = get_peft_model(model, lora_cfg)
        if args.gradient_checkpointing:
            model.enable_input_require_grads()

    total_samples = args.max_steps * args.per_device_batch_size * args.grad_accum
    train_ds = build_dataset(tokenizer, args.seq_len, total_samples, args.dataset)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        data_collator=collator,
        tokenizer=tokenizer,
    )

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
        samples_per_s = samples / elapsed if elapsed > 0 else 0.0
        tokens_per_s = samples_per_s * args.seq_len
        metrics = {
            "mode": "ddp" if world_size > 1 else "single_gpu",
            "rows": len(train_ds),
            "world_size": world_size,
            "seq_len": args.seq_len,
            "learning_rate": args.lr,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "total_time_s": elapsed,
            "avg_step_s": elapsed / args.max_steps,
            "samples_per_s": samples_per_s,
            "tokens_per_s": tokens_per_s,
            "max_peak_reserved_gib": round(peak_reserved.item() / 2**30, 3),
        }
        metrics_path = out_dir / "metrics.json"
        metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2))
        if run_logger:
            run_logger.finish("success", metrics)
        print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    main()
