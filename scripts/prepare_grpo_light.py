#!/usr/bin/env python3
"""Prepare a tiny DAPO prompt subset plus answer sidecar for lightweight GRPO."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def prompt_text(value) -> str:
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, dict):
                parts.append(str(item.get("content", "")))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part).strip()
    return str(value).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw/dapo-math-17k.parquet")
    parser.add_argument("--output", default="data/processed/grpo-prompts-light.jsonl")
    parser.add_argument("--answer-map", default="data/processed/grpo-answer-map-light.json")
    parser.add_argument("--size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_parquet(args.input)
    sampled = df.sample(n=min(args.size, len(df)), random_state=args.seed).reset_index(drop=True)
    output = Path(args.output)
    answer_map_path = Path(args.answer_map)
    output.parent.mkdir(parents=True, exist_ok=True)
    answer_map_path.parent.mkdir(parents=True, exist_ok=True)

    answer_map = {}
    with output.open("w", encoding="utf-8") as handle:
        for _, row in sampled.iterrows():
            prompt = prompt_text(row["prompt"])
            reward_model = row.get("reward_model", {})
            if not isinstance(reward_model, dict):
                reward_model = json.loads(reward_model)
            answer = str(reward_model.get("ground_truth", "")).strip()
            if not prompt or not answer:
                continue
            handle.write(json.dumps({"prompt": prompt}, ensure_ascii=False) + "\n")
            answer_map[prompt] = answer

    answer_map_path.write_text(json.dumps(answer_map, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {len(answer_map)} prompts to {output}")
    print(f"wrote answer map to {answer_map_path}")


if __name__ == "__main__":
    main()
