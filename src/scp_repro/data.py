"""Input normalization for Light-R1 and generic reproduction records."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from .jsonl import read_jsonl


_THINK = re.compile(r"<think>\s*(.*?)\s*</think>\s*(.*)", flags=re.DOTALL | re.IGNORECASE)


def load_records(path: str | Path) -> Iterator[dict[str, Any]]:
    source = Path(path)
    if source.suffix == ".jsonl":
        yield from read_jsonl(source)
        return
    with source.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, dict):
        data = data.get("train", data.get("data", [data]))
    if not isinstance(data, list):
        raise ValueError("JSON input must be a list, or a dict containing train/data")
    for row in data:
        if not isinstance(row, dict):
            raise ValueError("every input row must be an object")
        yield row


def stable_id(question: str) -> str:
    return hashlib.sha256(question.strip().encode("utf-8")).hexdigest()[:20]


def normalize_sft_row(row: dict[str, Any]) -> dict[str, str]:
    if "conversations" in row:
        turns = row["conversations"]
        user = next((x["value"] for x in turns if x.get("from") in {"user", "human"}), None)
        assistant = next((x["value"] for x in turns if x.get("from") in {"assistant", "gpt"}), None)
        if not user or not assistant:
            raise ValueError("ShareGPT row requires user and assistant turns")
        question, response = str(user), str(assistant)
    else:
        question = str(row.get("question") or row.get("problem") or row.get("instruction") or "")
        response = str(row.get("response") or row.get("solution") or row.get("answer") or "")
        if not question or not response:
            raise ValueError("row requires question/problem and response/solution")

    match = _THINK.fullmatch(response.strip())
    if match:
        reasoning, final_answer = match.group(1).strip(), match.group(2).strip()
    else:
        reasoning, final_answer = response.strip(), ""
    return {
        "id": str(row.get("id") or stable_id(question)),
        "question": question.strip(),
        "reasoning": reasoning,
        "final_answer": final_answer,
        "response": response.strip(),
    }
