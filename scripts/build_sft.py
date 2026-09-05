#!/usr/bin/env python3
"""Prune graph records and emit portable SFT JSONL.

Each record contains both OpenAI-style messages (used by the local training
platform) and ShareGPT-style conversations (used by LLaMA-Factory).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scp_repro.response import compose_response
from scp_repro.graph import ReasoningGraph
from scp_repro.jsonl import read_jsonl
from scp_repro.pruning import prune_graph


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--stats", default="data/processed/pruning-stats.json")
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--m", type=float, default=0.9)
    args = parser.parse_args()

    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    total_nodes = kept_nodes = total_words = kept_words = count = 0
    with target.open("w", encoding="utf-8") as output:
        for row in read_jsonl(args.input):
            graph = ReasoningGraph.model_validate(row["graph"])
            result = prune_graph(graph, k=args.k, m=args.m)
            kept_indices = sorted({i for node in result.graph.nodes for i in node.chunks})
            final_index = len(row["chunks"]) - 1 if row.get("final_answer") else None
            reasoning_indices = [i for i in kept_indices if i != final_index]
            retained_reasoning = "\n\n".join(row["chunks"][i].strip() for i in reasoning_indices)
            retained = compose_response(retained_reasoning, row.get("final_answer", ""))
            messages = [
                {"role": "user", "content": row["question"]},
                {"role": "assistant", "content": retained},
            ]
            output.write(
                json.dumps(
                    {
                        "id": row["id"],
                        "messages": messages,
                        "conversations": [
                            {"from": "human", "value": row["question"]},
                            {"from": "gpt", "value": retained},
                        ],
                        "pruning": {
                            "removed": sorted(result.removed),
                            "branch": sorted(result.branch_redundant),
                            "depth": sorted(result.depth_redundant),
                        },
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            count += 1
            total_nodes += len(graph.nodes)
            kept_nodes += len(result.graph.nodes)
            total_words += len(row["response"].split())
            kept_words += len(retained.split())

    stats = {
        "samples": count,
        "k": args.k,
        "m": args.m,
        "avg_nodes_before": total_nodes / count if count else 0,
        "avg_nodes_after": kept_nodes / count if count else 0,
        "avg_whitespace_words_before": total_words / count if count else 0,
        "avg_whitespace_words_after": kept_words / count if count else 0,
    }
    stats_path = Path(args.stats)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
