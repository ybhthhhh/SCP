from scp_repro.token_length import reasoning_token_length


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        return {"<think>": [10], "</think>": [11]}[text]


def test_reasoning_span_excludes_tags_and_final_answer():
    assert reasoning_token_length([10, 1, 2, 3, 11, 4, 5], FakeTokenizer()) == 3


def test_reasoning_span_falls_back_to_full_response():
    assert reasoning_token_length([1, 2, 3], FakeTokenizer()) == 3
