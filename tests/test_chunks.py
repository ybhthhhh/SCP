from scp_repro.chunks import split_cot


def test_split_keeps_triggers_and_paragraphs():
    text = "First derive x. But wait this is wrong. Therefore use y.\n\nFinal result."
    assert split_cot(text) == [
        "First derive x.",
        "But wait this is wrong.",
        "Therefore use y.",
        "Final result.",
    ]


def test_split_empty():
    assert split_cot("  \n") == []
