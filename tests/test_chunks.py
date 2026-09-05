from scp_repro.chunks import split_cot


def test_split_keeps_literal_triggers():
    text = "First derive x. But wait this is wrong. Therefore use y.\n\nFinal result."
    assert split_cot(text) == [
        "First derive x.",
        "But wait this is wrong.",
        "Therefore use y.\n\nFinal result.",
    ]


def test_split_is_literal_and_case_sensitive():
    assert split_cot("Okay, so continue. Then use x.") == ["Okay, so continue.", "Then use x."]


def test_split_empty():
    assert split_cot("  \n") == []
