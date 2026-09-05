"""Split a raw chain-of-thought into the paper's step-level chunks."""

from __future__ import annotations

import re

# Appendix B. Longest strings come first so "But wait" is not split at "Wait".
SPLIT_TOKENS = (
    "I don't see any errors",
    "Let me double-check",
    "That seems right",
    "Another approach",
    "Another angle",
    "Looking back",
    "Seems solid",
    "That's correct",
    "Alternatively",
    "Similarly",
    "Therefore",
    "Remember",
    "Starting",
    "But wait",
    "Hold on",
    "Let me",
    "Correct",
    "Compute",
    "Alright",
    "First",
    "Then",
    "Okay",
    "Wait",
    "Hmm",
    "Maybe",
    "Good",
    "Got it",
    "I think",
    "Let's see",
    "Now",
    "So",
    "Thus",
)

_BOUNDARY = re.compile(
    r"(?i)(?<![A-Za-z])(?:" + "|".join(re.escape(x) for x in SPLIT_TOKENS) + r")(?:\b|(?=\s))"
)


def split_cot(text: str) -> list[str]:
    """Return non-empty chunks while keeping each trigger with its following text.

    New paragraphs are also treated as boundaries. Trigger matching is
    case-insensitive but does not split substrings inside another word.
    """

    text = text.strip()
    if not text:
        return []

    chunks: list[str] = []
    for paragraph in re.split(r"\n\s*\n+", text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        starts = [m.start() for m in _BOUNDARY.finditer(paragraph)]
        starts = sorted(set([0, *starts, len(paragraph)]))
        for left, right in zip(starts, starts[1:]):
            chunk = paragraph[left:right].strip()
            if chunk:
                chunks.append(chunk)
    return chunks
