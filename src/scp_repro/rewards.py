"""DPO redundancy and paper-faithful group-relative GRPO rewards."""

from __future__ import annotations

from collections.abc import Sequence


def redundancy_score(*, review_nodes: int, total_nodes: int, token_length: int, mean_group_length: float) -> float:
    if total_nodes <= 0:
        raise ValueError("total_nodes must be positive")
    if mean_group_length <= 0:
        raise ValueError("mean_group_length must be positive")
    return review_nodes / total_nodes + token_length / mean_group_length


def group_length_rewards(
    correctness: Sequence[bool | int | float],
    lengths: Sequence[int],
    *,
    lambda_length: float,
    tolerance: int,
    gamma: float,
) -> list[float]:
    """Compute R=V-lambda*1[V=1]*delta**gamma for one rollout group."""

    if len(correctness) != len(lengths):
        raise ValueError("correctness and lengths must have the same size")
    if lambda_length < 0 or tolerance < 0 or gamma < 1:
        raise ValueError("require lambda>=0, tolerance>=0, gamma>=1")
    if any(length < 0 for length in lengths):
        raise ValueError("lengths must be non-negative")

    correct_lengths = [length for ok, length in zip(correctness, lengths) if bool(ok)]
    if not correct_lengths:
        return [0.0] * len(lengths)
    shortest = min(correct_lengths)
    denominator = shortest + tolerance
    if denominator == 0:
        denominator = 1

    rewards: list[float] = []
    for ok, length in zip(correctness, lengths):
        if not bool(ok):
            rewards.append(0.0)
            continue
        excess = max(length - shortest - tolerance, 0) / denominator
        rewards.append(1.0 - lambda_length * (excess**gamma))
    return rewards
