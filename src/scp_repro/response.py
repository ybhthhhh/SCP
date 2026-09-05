"""Formatting helpers for reasoning-model responses."""


def compose_response(reasoning: str, final_answer: str) -> str:
    if final_answer:
        return f"<think>\n{reasoning.strip()}\n</think>\n{final_answer.strip()}"
    return reasoning.strip()
