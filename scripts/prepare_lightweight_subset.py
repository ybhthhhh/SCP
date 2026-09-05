#!/usr/bin/env python3
"""Create a deterministic, order-preserving subset for the lightweight run."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from scp_repro.data import load_records, normalize_sft_row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--size", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = list(load_records(args.input))
    if args.size <= 0:
        raise SystemExit("--size must be positive")
    if args.size > len(rows):
        raise SystemExit(f"requested {args.size} rows from a dataset of {len(rows)}")

    selected_indices = sorted(random.Random(args.seed).sample(range(len(rows)), args.size))
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for index in selected_indices:
            handle.write(json.dumps(rows[index], ensure_ascii=False) + "\n")

    normalized = [normalize_sft_row(rows[index]) for index in selected_indices]
    manifest_path = Path(args.manifest) if args.manifest else target.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(
            {
                "source": str(Path(args.input)),
                "source_rows": len(rows),
                "subset_rows": len(selected_indices),
                "sampling": "random_without_replacement_then_source_order",
                "seed": args.seed,
                "source_indices": selected_indices,
                "sample_ids": [row["id"] for row in normalized],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(selected_indices)} rows to {target}")
    print(f"wrote sampling manifest to {manifest_path}")


if __name__ == "__main__":
    main()
