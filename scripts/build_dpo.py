#!/usr/bin/env python3
"""Rank correct rollout trajectories by the paper's redundancy score."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from scp_repro.jsonl import read_jsonl
from scp_repro.rewards import redundancy_score


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="JSONL rollout records with graph statistics")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-pairs-per-question", type=int, default=1)
    args = parser.parse_args()

    groups: dict[str, list[dict]] = defaultdict(list)
    for row in read_jsonl(args.input):
        if bool(row.get("correct")):
            groups[str(row.get("question_id") or row["question"])].append(row)

    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    pairs = 0
    with target.open("w", encoding="utf-8") as output:
        for trajectories in groups.values():
            if len(trajectories) < 2:
                continue
            mean_length = sum(int(x["token_length"]) for x in trajectories) / len(trajectories)
            ranked = sorted(
                trajectories,
                key=lambda x: redundancy_score(
                    review_nodes=int(x["review_nodes"]),
                    total_nodes=int(x["total_nodes"]),
                    token_length=int(x["token_length"]),
                    mean_group_length=mean_length,
                ),
            )
            for offset in range(min(args.max_pairs_per_question, len(ranked) // 2)):
                chosen, rejected = ranked[offset], ranked[-1 - offset]
                output.write(
                    json.dumps(
                        {
                            "conversations": [{"from": "human", "value": chosen["question"]}],
                            "chosen": {"from": "gpt", "value": chosen["response"]},
                            "rejected": {"from": "gpt", "value": rejected["response"]},
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                pairs += 1
    print(f"wrote {pairs} preference pairs to {target}")


if __name__ == "__main__":
    main()
