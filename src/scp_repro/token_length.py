"""Token-level reasoning span measurement."""

from __future__ import annotations

from collections.abc import Sequence


def _find_subsequence(sequence: list[int], needle: list[int], start: int = 0) -> int:
    if not needle:
        return -1
    for index in range(start, len(sequence) - len(needle) + 1):
        if sequence[index : index + len(needle)] == needle:
            return index
    return -1


def reasoning_token_length(token_ids: Sequence[int], tokenizer) -> int:
    """Count tokens inside <think>...</think>, falling back to full response."""

    ids = [int(token) for token in token_ids]
    start_tokens = tokenizer.encode("<think>", add_special_tokens=False)
    end_tokens = tokenizer.encode("</think>", add_special_tokens=False)
    start = _find_subsequence(ids, start_tokens)
    left = start + len(start_tokens) if start >= 0 else 0
    end = _find_subsequence(ids, end_tokens, start=left)
    return max((end if end >= 0 else len(ids)) - left, 0)
