"""Numerical check for exact vocabulary-chunked forward KL."""
import ast
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


def load_functions(source_path: Path):
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    names = {"forward_kl", "forward_kl_from_logits_chunked", "sampled_token_k3_kl"}
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(source_path), "exec"), namespace)
    return namespace


if __name__ == "__main__":
    functions = load_functions(Path(sys.argv[1]))
    torch.manual_seed(29)
    current_logits = torch.randn(2, 37, 113, requires_grad=True)
    ref_logits = torch.randn(2, 37, 113)
    mask = torch.randint(0, 2, (2, 37), dtype=torch.float32)
    mask[:, 0] = 1

    reference = functions["forward_kl"](
        F.log_softmax(current_logits, dim=-1), F.log_softmax(ref_logits, dim=-1), mask
    )
    actual = functions["forward_kl_from_logits_chunked"](current_logits, ref_logits, mask, 17)
    torch.testing.assert_close(actual, reference, rtol=1e-5, atol=1e-6)
    actual.backward()
    assert current_logits.grad is not None and torch.isfinite(current_logits.grad).all()

    policy_logits = torch.randn(1, 1, 11, requires_grad=True)
    reference_logits = torch.randn(1, 1, 11)
    policy_log_probs = F.log_softmax(policy_logits, dim=-1)
    reference_log_probs = F.log_softmax(reference_logits, dim=-1)
    sampled_token = torch.tensor([[7]])
    policy_action_logprob = policy_log_probs.gather(-1, sampled_token.unsqueeze(-1)).squeeze(-1)
    reference_action_logprob = reference_log_probs.gather(-1, sampled_token.unsqueeze(-1)).squeeze(-1)
    k3 = functions["sampled_token_k3_kl"](
        policy_action_logprob, reference_action_logprob, torch.ones_like(policy_action_logprob)
    )
    manual = torch.exp(reference_action_logprob - policy_action_logprob)
    manual = manual - (reference_action_logprob - policy_action_logprob) - 1.0
    torch.testing.assert_close(k3, manual.mean())
    k3.backward()
    assert policy_logits.grad is not None and torch.isfinite(policy_logits.grad).all()

    log_ratio = reference_log_probs - policy_log_probs.detach()
    k3_expectation = (policy_log_probs.detach().exp() * (torch.exp(log_ratio) - log_ratio - 1.0)).sum()
    exact_kl = (policy_log_probs.detach().exp() * (policy_log_probs.detach() - reference_log_probs)).sum()
    torch.testing.assert_close(k3_expectation, exact_kl, rtol=1e-5, atol=1e-6)
    print("PASS exact chunked KL and sampled-token k3 KL checks")
