#!/usr/bin/env python3
"""Rank correct rollout trajectories by the paper's redundancy score."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from transformers import AutoTokenizer

from scp_repro.graph import ReasoningGraph

from scp_repro.jsonl import read_jsonl
from scp_repro.pruning import prune_graph
from scp_repro.rewards import redundancy_score


def pruned_proxy_rollouts(
    graph_path: str,
    sft_path: str,
    tokenizer_path: str,
    *,
    k: int,
    m: float,
) -> list[dict]:
    """Build zero-API proxy rollouts from matching original/pruned traces."""

    sft_by_id = {str(row["id"]): row for row in read_jsonl(sft_path)}
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    rows: list[dict] = []
    for row in read_jsonl(graph_path):
        row_id = str(row["id"])
        if row_id not in sft_by_id:
            raise ValueError(f"missing SFT row for graph id {row_id}")
        original_graph = ReasoningGraph.model_validate(row["graph"])
        concise_graph = prune_graph(original_graph, k=k, m=m).graph
        conversations = sft_by_id[row_id].get("conversations", [])
        concise = next(
            (item.get("value", "") for item in conversations if item.get("from") == "gpt"),
            "",
        )
        if not concise:
            raise ValueError(f"missing concise assistant response for id {row_id}")
        variants = (
            ("pruned", concise_graph, concise),
            ("original", original_graph, row["response"]),
        )
        for variant, graph, response in variants:
            rows.append(
                {
                    "question_id": row_id,
                    "question": row["question"],
                    "response": response,
                    "correct": True,
                    "review_nodes": sum(node.type == "review" for node in graph.nodes),
                    "total_nodes": len(graph.nodes),
                    "token_length": len(
                        tokenizer(response, add_special_tokens=False)["input_ids"]
                    ),
                    "variant": variant,
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="JSONL rollout records with graph statistics")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-pairs-per-question", type=int, default=1)
    parser.add_argument("--pruned-sft", help="Use original/pruned lightweight proxy pairs")
    parser.add_argument("--tokenizer", help="Tokenizer path required with --pruned-sft")
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--m", type=float, default=0.9)
    args = parser.parse_args()

    if args.pruned_sft and not args.tokenizer:
        raise SystemExit("--tokenizer is required with --pruned-sft")
    source_rows = (
        pruned_proxy_rollouts(
            args.input, args.pruned_sft, args.tokenizer, k=args.k, m=args.m
        )
        if args.pruned_sft
        else read_jsonl(args.input)
    )

    groups: dict[str, list[dict]] = defaultdict(list)
    for row in source_rows:
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
                if args.pruned_sft and chosen.get("variant") != "pruned":
                    continue
                chosen_score = redundancy_score(
                    review_nodes=int(chosen["review_nodes"]),
                    total_nodes=int(chosen["total_nodes"]),
                    token_length=int(chosen["token_length"]),
                    mean_group_length=mean_length,
                )
                rejected_score = redundancy_score(
                    review_nodes=int(rejected["review_nodes"]),
                    total_nodes=int(rejected["total_nodes"]),
                    token_length=int(rejected["token_length"]),
                    mean_group_length=mean_length,
                )
                if chosen_score >= rejected_score:
                    continue
                output.write(
                    json.dumps(
                        {
                            "id": str(chosen.get("question_id") or chosen["question"]),
                            "prompt": chosen["question"],
                            "conversations": [{"from": "human", "value": chosen["question"]}],
                            "chosen": chosen["response"],
                            "rejected": rejected["response"],
                            "preference": {
                                "method": "pruned_vs_original_proxy" if args.pruned_sft else "sampled_rollouts",
                                "chosen_score": chosen_score,
                                "rejected_score": rejected_score,
                                "chosen_review_nodes": int(chosen["review_nodes"]),
                                "rejected_review_nodes": int(rejected["review_nodes"]),
                                "chosen_total_nodes": int(chosen["total_nodes"]),
                                "rejected_total_nodes": int(rejected["total_nodes"]),
                                "chosen_token_length": int(chosen["token_length"]),
                                "rejected_token_length": int(rejected["token_length"]),
                            },
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                pairs += 1
    print(f"wrote {pairs} preference pairs to {target}")


if __name__ == "__main__":
    main()
