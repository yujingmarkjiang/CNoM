from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class DistributedContext:
    enabled: bool
    rank: int
    local_rank: int
    world_size: int
    device: torch.device

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def initialize_distributed(device_name: str = "auto", backend: str = "nccl") -> DistributedContext:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    enabled = world_size > 1
    if enabled and not dist.is_initialized():
        selected_backend = backend if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=selected_backend, init_method="env://")
    if torch.cuda.is_available() and device_name != "cpu":
        device = torch.device("cuda", local_rank if enabled else 0)
        torch.cuda.set_device(device)
    elif device_name.startswith("cuda"):
        raise RuntimeError("CUDA was requested but is unavailable")
    elif device_name == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return DistributedContext(enabled, rank, local_rank, world_size, device)


def all_gather_objects(value):
    if not dist.is_available() or not dist.is_initialized():
        return [value]
    gathered = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, value)
    return gathered


def barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def finalize_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
