"""verl adapter for the paper's group-relative correctness/length reward.

This targets verl commit 23af6a7a2e8d6efeeb2adbe5d1689c7a24f503a3.
The paper's L*(x) requires all eight rollouts for a prompt. Current verl scores
individual samples asynchronously, so this manager uses the rollout UID as a
barrier. ``reward.num_workers`` MUST be 1 or a UID group could be split across
actors and deadlock.
"""

from __future__ import annotations

import asyncio
import inspect
from collections import defaultdict

from scp_repro.rewards import group_length_rewards
from scp_repro.token_length import reasoning_token_length
from verl import DataProto
from verl.experimental.reward_loop.reward_manager.base import RewardManagerBase
from verl.utils.reward_score import default_compute_score


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    """Use verl's math verifier; the manager converts its result to binary V."""

    return default_compute_score(
        data_source=data_source,
        solution_str=solution_str,
        ground_truth=ground_truth,
        extra_info=extra_info,
        **kwargs,
    )


class GroupLengthRewardManager(RewardManagerBase):
    def __init__(self, config, tokenizer, compute_score=None, **_kwargs):
        super().__init__(config, tokenizer, compute_score)
        self.compute_score = compute_score or default_compute_score
        self.is_async_score = inspect.iscoroutinefunction(self.compute_score)
        self.group_size = int(config.actor_rollout_ref.rollout.n)
        reward_kwargs = config.reward.custom_reward_function.get("reward_kwargs", {})
        self.lambda_length = float(reward_kwargs.get("lambda_length", 0.1))
        self.tolerance = int(reward_kwargs.get("tolerance", 256))
        self.gamma = float(reward_kwargs.get("gamma", 1.0))
        self.correct_threshold = float(reward_kwargs.get("correct_threshold", 0.999))
        self._lock = asyncio.Lock()
        self._pending = defaultdict(list)

    async def _correctness(self, data_source, response, ground_truth, extra_info) -> bool:
        kwargs = dict(
            data_source=data_source,
            solution_str=response,
            ground_truth=ground_truth,
            extra_info=extra_info,
        )
        if self.is_async_score:
            raw = await self.compute_score(**kwargs)
        else:
            raw = await self.loop.run_in_executor(None, lambda: self.compute_score(**kwargs))
        value = raw.get("score", raw.get("acc", 0.0)) if isinstance(raw, dict) else raw
        return float(value) >= self.correct_threshold

    async def run_single(self, data: DataProto) -> dict:
        item = data[-1:][0]
        response_ids = item.batch["responses"]
        response_length = response_ids.shape[-1]
        valid_length = int(item.batch["attention_mask"][-response_length:].sum().item())
        valid_ids = response_ids[:valid_length]
        response = await self.loop.run_in_executor(
            None, lambda: self.tokenizer.decode(valid_ids, skip_special_tokens=True)
        )
        reasoning_length = reasoning_token_length(valid_ids, self.tokenizer)
        uid = str(item.non_tensor_batch["uid"])
        source = item.non_tensor_batch["data_source"]
        ground_truth = item.non_tensor_batch["reward_model"]["ground_truth"]
        extra_info = dict(item.non_tensor_batch.get("extra_info", {}))
        correct = await self._correctness(source, response, ground_truth, extra_info)
        future = asyncio.get_running_loop().create_future()

        async with self._lock:
            group = self._pending[uid]
            group.append((correct, reasoning_length, valid_length, future))
            if len(group) > self.group_size:
                raise RuntimeError(f"UID {uid} received more than {self.group_size} rollouts")
            if len(group) == self.group_size:
                correctness = [entry[0] for entry in group]
                lengths = [entry[1] for entry in group]
                rewards = group_length_rewards(
                    correctness,
                    lengths,
                    lambda_length=self.lambda_length,
                    tolerance=self.tolerance,
                    gamma=self.gamma,
                )
                shortest = min((n for ok, n in zip(correctness, lengths) if ok), default=None)
                for (ok, length, total_length, waiter), reward in zip(group, rewards):
                    waiter.set_result(
                        {
                            "reward_score": reward,
                            "reward_extra_info": {
                                "acc": float(ok),
                                "reasoning_length": length,
                                "response_length": total_length,
                                "shortest_correct_length": -1 if shortest is None else shortest,
                                "length_reward": reward - float(ok),
                            },
                        }
                    )
                del self._pending[uid]
        return await future
