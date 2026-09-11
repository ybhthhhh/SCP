"""Exercise the trainer's actual microbatch loop with two CPU/Gloo ranks."""
import ast
import copy
import os
import sys
import tempfile
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Sequence

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP


class ToyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(16, 8)
        self.head = torch.nn.Linear(8, 16)

    def forward(self, input_ids, **kwargs):
        return SimpleNamespace(logits=self.head(self.embedding(input_ids)))


class SilentLogger:
    def event(self, *args, **kwargs):
        pass


def count_reduce(state, bucket):
    state["calls"] += 1
    value = bucket.buffer()
    dist.all_reduce(value)
    value /= dist.get_world_size()
    future = torch.futures.Future()
    future.set_result(value)
    return future


def worker(rank, rendezvous, source_path):
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)
    try:
        torch.set_num_threads(1)
        tree = ast.parse(Path(source_path).read_text(encoding="utf-8"))
        names = {"pad_samples", "sequence_logprobs", "forward_kl", "backward_only"}
        funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
        main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
        loop = next(n for n in ast.walk(main) if isinstance(n, ast.For)
                    and isinstance(n.target, ast.Tuple)
                    and any(isinstance(t, ast.Name) and t.id == "chunk_index" for t in n.target.elts))
        env = dict(torch=torch, F=F, nullcontext=nullcontext, Dict=Dict, Sequence=Sequence)
        exec(compile(ast.Module(body=funcs, type_ignores=[]), source_path, "exec"), env)
        code = compile(ast.Module(body=[loop], type_ignores=[]), source_path, "exec")
        for count, micro in [(8, 1), (7, 3)]:
            torch.manual_seed(17)
            model = ToyPolicy()
            baseline = copy.deepcopy(model)
            ref_model = copy.deepcopy(model).eval()
            ddp = DDP(model)
            state = {"calls": 0}
            ddp.register_comm_hook(state, count_reduce)
            samples = [dict(prompt_token_ids=[1, 2 + rank],
                            completion_token_ids=[3 + i, 12]) for i in range(count)]
            ids, attention, completion = env["pad_samples"](samples, 0, torch.device("cpu"))
            with torch.no_grad():
                old, _, _ = env["sequence_logprobs"](baseline, ids, attention, completion)
            advantage = torch.arange(count, dtype=torch.float32) - count / 2 + rank * 0.25
            new, new_probs, mask = env["sequence_logprobs"](baseline, ids, attention, completion)
            with torch.no_grad():
                _, ref_probs, _ = env["sequence_logprobs"](ref_model, ids, attention, completion)
            ratio = (new - old).exp()
            loss = -torch.minimum(ratio * advantage, ratio.clamp(0.8, 1.2) * advantage).mean()
            loss = loss + 0.001 * env["forward_kl"](new_probs, ref_probs, mask)
            loss.backward()
            for param in baseline.parameters():
                dist.all_reduce(param.grad)
                param.grad /= 2
            env.update(sample_count=count, train_micro_batch_size=micro, samples=samples,
                       train_model=ddp, ref_model=ref_model, rank_logger=SilentLogger(),
                       tokenizer=SimpleNamespace(pad_token_id=0), step=0, world_size=2,
                       device=torch.device("cpu"),
                       old_seq_logprobs=list(old.split(micro)), advantages_tensor=advantage,
                       args=SimpleNamespace(use_deepspeed=False, ppo_clip_range=0.2, kl_coef=0.001),
                       loss_total=0.0, policy_loss_total=0.0, kl_total=0.0)
            exec(code, env)
            assert state["calls"] == 1, state
            for actual, expected in zip(model.parameters(), baseline.parameters()):
                torch.testing.assert_close(actual.grad, expected.grad, rtol=1e-5, atol=1e-6)
            print(f"PASS rank={rank} samples={count} micro={micro} reductions=1 gradients=match", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
    with tempfile.TemporaryDirectory(prefix="scp-ddp-cpu-") as directory:
        rendezvous = Path(directory, "rendezvous").as_uri()
        mp.spawn(worker, args=(rendezvous, sys.argv[1]), nprocs=2, join=True)
