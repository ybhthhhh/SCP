"""Small crash-tolerant JSONL helpers."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSONL at {path}:{line_number}: {error}") from error
            if not isinstance(value, dict):
                raise ValueError(f"expected object at {path}:{line_number}")
            yield value


def completed_ids(path: str | Path, id_field: str = "id") -> set[str]:
    output = Path(path)
    if not output.exists():
        return set()
    return {str(row[id_field]) for row in read_jsonl(output) if id_field in row}


class DurableJsonlWriter:
    """Append records and fsync each one so a node crash loses at most one row."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a", encoding="utf-8")

    def write(self, row: dict[str, Any]) -> None:
        self.handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.handle.flush()
        os.fsync(self.handle.fileno())

    def close(self) -> None:
        self.handle.close()

    def __enter__(self) -> "DurableJsonlWriter":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
