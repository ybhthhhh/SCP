#!/usr/bin/env python3
"""Build one LLM reasoning DAG per SFT example with durable resumption."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import sys
import threading
import time

from openai import OpenAI

from scp_repro.builder import GraphBuilder
from scp_repro.chunks import split_cot
from scp_repro.data import load_records, normalize_sft_row
from scp_repro.jsonl import DurableJsonlWriter, completed_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default=os.getenv("GRAPH_MODEL", "deepseek-v4-flash"))
    parser.add_argument("--base-url", default=os.getenv("GRAPH_BASE_URL"))
    parser.add_argument("--host-header", default=os.getenv("GRAPH_HOST_HEADER"))
    parser.add_argument("--api-key-env", default="GRAPH_API_KEY")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-retries", type=int, default=0, help="Retries inside the OpenAI client.")
    parser.add_argument("--failure-retries", type=int, default=1, help="Whole-row retries after a real failure.")
    parser.add_argument("--empty-retry-delay", type=float, default=1.0)
    parser.add_argument("--failure-retry-delay", type=float, default=2.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1")
    if args.failure_retries < 0:
        raise SystemExit("--failure-retries cannot be negative")

    api_key = os.getenv(args.api_key_env)
    if not api_key:
        raise SystemExit(f"missing API key environment variable: {args.api_key_env}")
    default_headers = {"Host": args.host_header} if args.host_header else None
    thread_state = threading.local()

    def get_client() -> OpenAI:
        client = getattr(thread_state, "client", None)
        if client is None:
            client = OpenAI(
                api_key=api_key,
                base_url=args.base_url,
                max_retries=args.max_retries,
                default_headers=default_headers,
            )
            thread_state.client = client
        return client

    def complete(system: str, user: str) -> str:
        empty_retries = 0
        while True:
            result = get_client().chat.completions.create(
                model=args.model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                temperature=0,
                response_format={"type": "json_object"},
            )
            content = result.choices[0].message.content
            if content and content.strip():
                return content
            empty_retries += 1
            delay = min(args.empty_retry_delay * (2 ** min(empty_retries - 1, 4)), 10.0)
            print(f"empty_response retry={empty_retries} delay_s={delay:g}", flush=True)
            time.sleep(delay)

    def build_row(raw: dict) -> tuple[dict, list[str], object]:
        row = normalize_sft_row(raw)
        chunks = split_cot(row["reasoning"])
        if row["final_answer"]:
            chunks.append(row["final_answer"])

        for attempt in range(args.failure_retries + 1):
            try:
                graph = GraphBuilder(complete).build(chunks)
                return row, chunks, graph
            except Exception as error:
                if attempt >= args.failure_retries:
                    raise
                print(
                    f"retrying id={row['id']} failure={attempt + 1} "
                    f"error={type(error).__name__}: {error}",
                    flush=True,
                )
                time.sleep(args.failure_retry_delay)
        raise AssertionError("unreachable")

    done = completed_ids(args.output)
    pending = []
    for raw in load_records(args.input):
        row_id = normalize_sft_row(raw)["id"]
        if row_id in done:
            continue
        pending.append(raw)
        if args.limit is not None and len(pending) >= args.limit:
            break

    if not pending:
        print("no pending rows", flush=True)
        return 0

    completed = failures = 0
    with DurableJsonlWriter(args.output) as output:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(build_row, raw): raw for raw in pending}
            for future in as_completed(futures):
                try:
                    row, chunks, graph = future.result()
                except Exception as error:
                    failures += 1
                    failed_id = normalize_sft_row(futures[future])["id"]
                    print(
                        f"failed id={failed_id} error={type(error).__name__}: {error}",
                        flush=True,
                    )
                    continue
                output.write({**row, "chunks": chunks, "graph": graph.model_dump(mode="json")})
                completed += 1
                print(f"completed={completed} id={row['id']} chunks={len(chunks)}", flush=True)

    print(f"summary completed={completed} failed={failures}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
