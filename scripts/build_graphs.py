#!/usr/bin/env python3
"""Build one LLM reasoning DAG per SFT example with durable resumption."""

from __future__ import annotations

import argparse
import os
import sys

from openai import OpenAI

from scp_repro.builder import GraphBuilder
from scp_repro.chunks import split_cot
from scp_repro.data import load_records, normalize_sft_row
from scp_repro.jsonl import DurableJsonlWriter, completed_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default=os.getenv("GRAPH_MODEL", "qwen3-turbo"))
    parser.add_argument("--base-url", default=os.getenv("GRAPH_BASE_URL"))
    parser.add_argument("--api-key-env", default="GRAPH_API_KEY")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-retries", type=int, default=5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    api_key = os.getenv(args.api_key_env)
    if not api_key:
        raise SystemExit(f"missing API key environment variable: {args.api_key_env}")
    client = OpenAI(api_key=api_key, base_url=args.base_url, max_retries=args.max_retries)

    def complete(system: str, user: str) -> str:
        result = client.chat.completions.create(
            model=args.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0,
            response_format={"type": "json_object"},
        )
        content = result.choices[0].message.content
        if not content:
            raise RuntimeError("graph model returned empty content")
        return content

    builder = GraphBuilder(complete)
    done = completed_ids(args.output)
    seen = 0
    with DurableJsonlWriter(args.output) as output:
        for raw in load_records(args.input):
            row = normalize_sft_row(raw)
            if row["id"] in done:
                continue
            chunks = split_cot(row["reasoning"])
            if row["final_answer"]:
                chunks.append(row["final_answer"])
            graph = builder.build(chunks)
            output.write({**row, "chunks": chunks, "graph": graph.model_dump(mode="json")})
            seen += 1
            print(f"completed={seen} id={row['id']} chunks={len(chunks)}", flush=True)
            if args.limit is not None and seen >= args.limit:
                break
    return 0


if __name__ == "__main__":
    sys.exit(main())
