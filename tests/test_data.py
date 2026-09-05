from scp_repro.data import normalize_sft_row
from scp_repro.response import compose_response


def test_light_r1_normalization():
    row = {
        "conversations": [
            {"from": "user", "value": "2+2?"},
            {"from": "assistant", "value": "<think>calculate</think>\\boxed{4}"},
        ]
    }
    normalized = normalize_sft_row(row)
    assert normalized["reasoning"] == "calculate"
    assert normalized["final_answer"] == r"\boxed{4}"
    assert compose_response(normalized["reasoning"], normalized["final_answer"]).startswith("<think>")
