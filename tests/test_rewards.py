import pytest

from scp_repro.rewards import group_length_rewards, redundancy_score


def test_group_length_reward_only_penalizes_long_correct_answers():
    rewards = group_length_rewards(
        [True, True, False], [100, 160, 80], lambda_length=0.5, tolerance=20, gamma=1
    )
    assert rewards == pytest.approx([1.0, 1.0 - 0.5 * (40 / 120), 0.0])


def test_no_correct_rollout_has_zero_reward():
    assert group_length_rewards([False, False], [10, 20], lambda_length=1, tolerance=0, gamma=2) == [0, 0]


def test_redundancy_score_matches_paper_equation():
    assert redundancy_score(review_nodes=2, total_nodes=10, token_length=120, mean_group_length=100) == pytest.approx(1.4)
