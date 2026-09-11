"""Numerical check for the rollout-only GQA online attention helper."""
import ast
import sys
from pathlib import Path

import torch


def load_grouped_attention(source_path: Path):
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "grouped_online_attention"
    )
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source_path), "exec"), namespace)
    return namespace["grouped_online_attention"]


def repeat_kv(states: torch.Tensor, groups: int) -> torch.Tensor:
    batch, kv_heads, sequence, head_dim = states.shape
    return states[:, :, None].expand(batch, kv_heads, groups, sequence, head_dim).reshape(
        batch, kv_heads * groups, sequence, head_dim
    )


if __name__ == "__main__":
    source = Path(sys.argv[1])
    grouped_online_attention = load_grouped_attention(source)
    torch.manual_seed(23)
    query = torch.randn(1, 12, 1, 16)
    key = torch.randn(1, 2, 1297, 16)
    value = torch.randn(1, 2, 1297, 16)

    repeated_key = repeat_kv(key, 6)
    repeated_value = repeat_kv(value, 6)
    reference_weights = torch.softmax(torch.matmul(query, repeated_key.transpose(2, 3)) / 4.0, dim=-1)
    reference = torch.matmul(reference_weights, repeated_value)

    for block_size in (1, 127, 512):
        actual = grouped_online_attention(query, key, value, 6, block_size=block_size)
        torch.testing.assert_close(actual, reference, rtol=2e-5, atol=2e-5)
    print("PASS grouped online GQA attention matches repeat_kv + softmax")
