"""Distributed-training utilities (FSDP / DDP).

Public API (re-exported from :mod:`rtdmd.parallel.utils`):

Process-group lifecycle:
    setup_distributed, cleanup_distributed, barrier

Rank queries:
    get_rank, get_local_rank, get_world_size, get_local_world_size,
    is_main_process

FSDP / DDP wrappers and policies:
    fsdp_wrap_model, ddp_wrap_model, free_fsdp_unsharded_params,
    get_transformer_wrap_policy, get_sharding_strategy,
    get_mixed_precision_policy, get_device_mesh
"""

from rtdmd.parallel.utils import (
    barrier,
    cleanup_distributed,
    ddp_wrap_model,
    free_fsdp_unsharded_params,
    fsdp_wrap_model,
    get_device_mesh,
    get_local_rank,
    get_local_world_size,
    get_mixed_precision_policy,
    get_rank,
    get_sharding_strategy,
    get_transformer_wrap_policy,
    get_world_size,
    is_main_process,
    setup_distributed,
)

__all__ = [
    "barrier",
    "cleanup_distributed",
    "ddp_wrap_model",
    "free_fsdp_unsharded_params",
    "fsdp_wrap_model",
    "get_device_mesh",
    "get_local_rank",
    "get_local_world_size",
    "get_mixed_precision_policy",
    "get_rank",
    "get_sharding_strategy",
    "get_transformer_wrap_policy",
    "get_world_size",
    "is_main_process",
    "setup_distributed",
]
