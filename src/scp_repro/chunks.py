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
    r"(?=" + "|".join(re.escape(f" {token} ") for token in SPLIT_TOKENS) + r")"
)


def split_cot(text: str) -> list[str]:
    """Split on Appendix B's literal, case-sensitive, space-padded tokens.

    A lookahead keeps the trigger at the beginning of its following chunk.
    Padding lets a trigger at the beginning or end obey the same literal rule.
    """

    text = text.strip()
    if not text:
        return []

    padded = f" {text} "
    return [chunk.strip() for chunk in _BOUNDARY.split(padded) if chunk.strip()]
