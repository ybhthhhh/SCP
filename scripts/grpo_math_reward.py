#!/usr/bin/env python3
"""Lightweight math correctness plus group-relative length reward for platform GRPO."""

from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from pathlib import Path


def _load_answer_map() -> dict[str, str]:
    path = os.environ.get("GRPO_ANSWER_MAP", "data/processed/grpo-answer-map-light.json")
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _normalize(text: str) -> str:
    text = str(text).strip()
    text = text.replace("\\boxed", "")
    text = re.sub(r"[{}$\s]", "", text)
    text = text.rstrip(".,;:")
    return text.lower()


def _extract_answer(completion: str) -> str:
    text = completion.strip()
    matches = re.findall(r"Answer:\s*([^\n]+)", text, flags=re.IGNORECASE)
    if matches:
        return matches[-1].strip()
    boxed = re.findall(r"\\boxed\{([^{}]+)\}", text)
    if boxed:
        return boxed[-1].strip()
    numbers = re.findall(r"[-+]?\d+(?:/\d+)?(?:\.\d+)?", text)
    return numbers[-1] if numbers else ""


def _target_for_prompt(prompt: str, answer_map: dict[str, str]) -> str:
    if prompt in answer_map:
        return answer_map[prompt]
    for raw_prompt, answer in answer_map.items():
        if raw_prompt in prompt:
            return answer
    return ""


def _group_rewards(correctness: list[bool], lengths: list[int]) -> list[float]:
    lam = float(os.environ.get("LAMBDA_LENGTH", "0.1"))
    tolerance = int(os.environ.get("LENGTH_TOLERANCE", "64"))
    gamma = float(os.environ.get("LENGTH_GAMMA", "1.0"))
    correct_lengths = [n for ok, n in zip(correctness, lengths) if ok]
    if not correct_lengths:
        return [0.0] * len(correctness)
    shortest = min(correct_lengths)
    denominator = max(shortest + tolerance, 1)
    rewards = []
    for ok, length in zip(correctness, lengths):
        if not ok:
            rewards.append(0.0)
            continue
        excess = max(length - shortest - tolerance, 0) / denominator
        rewards.append(1.0 - lam * (excess ** gamma))
    return rewards


def score(samples):
    answer_map = _load_answer_map()
    grouped = defaultdict(list)
    for index, sample in enumerate(samples):
        grouped[str(sample.get("prompt", ""))].append((index, sample))

    rewards = [0.0] * len(samples)
    for prompt, items in grouped.items():
        target = _target_for_prompt(prompt, answer_map)
        correctness = []
        lengths = []
        for _, sample in items:
            completion = str(sample.get("completion", ""))
            pred = _extract_answer(completion)
            correctness.append(bool(target) and _normalize(pred) == _normalize(target))
            lengths.append(len(completion))
        group_values = _group_rewards(correctness, lengths)
        for (index, _), value in zip(items, group_values):
            rewards[index] = float(value)
    return rewards
