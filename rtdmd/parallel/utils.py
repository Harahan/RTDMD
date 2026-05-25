"""
Distributed training utilities for RTDMD.

Provides FSDP and DDP wrapping, process group management, and device mesh
initialization. Includes utilities with video-specific
logic removed.

Future extensions:
- Sequence parallelism for video DiT models
- Tensor parallelism for very large models
- Pipeline parallelism for multi-node training
"""

from __future__ import annotations

import os
import functools
from typing import Callable

import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    CPUOffload,
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    MixedPrecision,
    BackwardPrefetch,
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    size_based_auto_wrap_policy,
)
from torch.distributed.device_mesh import init_device_mesh


# ---------------------------------------------------------------------------
# Process group and rank helpers
# ---------------------------------------------------------------------------

def setup_distributed() -> None:
    """Initialize the default distributed process group.

    Expects standard torchrun environment variables (RANK, WORLD_SIZE,
    LOCAL_RANK, MASTER_ADDR, MASTER_PORT) to be set.
    """
    if dist.is_initialized():
        return
    # NCCL watchdog timeout default is 10 min; this kills training if rank 0
    # does slow work (e.g. wandb media upload during eval) while other ranks
    # hit a barrier. Extend to 3h so eval/image-upload can't kill training.
    from datetime import timedelta
    timeout_seconds = int(os.environ.get("NCCL_TIMEOUT_SEC", "10800"))
    dist.init_process_group(backend="nccl", timeout=timedelta(seconds=timeout_seconds))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)


def cleanup_distributed() -> None:
    """Destroy the default process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


def get_rank() -> int:
    return int(os.environ.get("RANK", 0))


def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", 0))


def get_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", 1))


def get_local_world_size() -> int:
    return int(os.environ.get("LOCAL_WORLD_SIZE", 1))


def is_main_process() -> bool:
    return get_rank() == 0


def barrier() -> None:
    """Synchronize all processes."""
    if dist.is_initialized():
        dist.barrier()


# ---------------------------------------------------------------------------
# FSDP sharding strategy helpers
# ---------------------------------------------------------------------------

_STRATEGY_MAP = {
    "full_shard": ShardingStrategy.FULL_SHARD,
    "shard_grad_op": ShardingStrategy.SHARD_GRAD_OP,
    "hybrid": ShardingStrategy.HYBRID_SHARD,
    "hybrid_zero2": ShardingStrategy._HYBRID_SHARD_ZERO2,
    "no_shard": ShardingStrategy.NO_SHARD,
}


def get_sharding_strategy(name: str) -> ShardingStrategy:
    """Map a string name to a torch FSDP ShardingStrategy enum.

    Args:
        name: One of "full_shard", "shard_grad_op", "hybrid", "hybrid_zero2", "no_shard".

    Returns:
        Corresponding ShardingStrategy enum value.
    """
    if name not in _STRATEGY_MAP:
        raise ValueError(
            f"Unknown sharding strategy: {name}. "
            f"Choose from: {list(_STRATEGY_MAP.keys())}"
        )
    return _STRATEGY_MAP[name]


# ---------------------------------------------------------------------------
# Mixed precision helpers
# ---------------------------------------------------------------------------

_DTYPE_ALIASES = {"bfloat16": "bf16", "float16": "fp16", "float32": "fp32"}


def get_mixed_precision_policy(fsdp_precision: str) -> MixedPrecision | None:
    """Create an FSDP MixedPrecision policy from a string identifier.

    Args:
        fsdp_precision: "bf16", "fp16", "fp32", or "no" (full fp32).
            Also accepts PyTorch-style names ("bfloat16", "float16", "float32").

    Returns:
        MixedPrecision policy or None for fp32.
    """
    fsdp_precision = _DTYPE_ALIASES.get(fsdp_precision, fsdp_precision)
    if fsdp_precision == "bf16":
        return MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )
    elif fsdp_precision == "fp16":
        return MixedPrecision(
            param_dtype=torch.float16,
            reduce_dtype=torch.float16,
            buffer_dtype=torch.float16,
        )
    elif fsdp_precision in {"no", "fp32"}:
        return None
    else:
        raise ValueError(
            f"Unknown precision: {fsdp_precision}. Choose from: bf16, fp16, fp32, no"
        )


# ---------------------------------------------------------------------------
# FSDP wrapping
# ---------------------------------------------------------------------------

def fsdp_wrap_model(
    model: torch.nn.Module,
    sharding_strategy: str = "full_shard",
    fsdp_precision: str = "bf16",
    auto_wrap_policy: Callable | None = None,
    device_id: int | None = None,
    use_orig_params: bool = True,
    ignored_modules: list | None = None,
    cpu_offload: bool = False,
) -> FSDP:
    """Wrap a model with FSDP.

    Args:
        model: The PyTorch module to wrap.
        sharding_strategy: FSDP sharding strategy name.
        fsdp_precision: FSDP precision policy name (param/reduce/buffer dtype).
            This is intentionally decoupled from autocast precision to keep
            gradient reduction stable (e.g., bf16 reduce with fp16 autocast).
        auto_wrap_policy: Custom FSDP auto-wrap policy. If None, uses
            size-based policy wrapping modules with >1M parameters.
        device_id: CUDA device ID. Defaults to LOCAL_RANK.
        use_orig_params: If True, FSDP preserves original parameter structure
            instead of flattening into FlatParameters. Required for correct
            behavior with LoRA (mixed frozen/trainable parameters) and
            recommended for PyTorch 2.0+.
        ignored_modules: List of submodules to exclude from FSDP sharding.
            Useful for modules containing Conv2d with 4D weights (e.g.,
            PatchEmbed.proj) which can cause shape mismatch in FSDP
            use_orig_params mode.
        cpu_offload: If True, keep FSDP parameters on CPU and copy to GPU per
            forward. Recommended for frozen inference-only models (e.g. teacher)
            to avoid overlapping unshard peaks with the trainable generator.

    Returns:
        FSDP-wrapped model.
    """
    if device_id is None:
        device_id = get_local_rank()

    strategy = get_sharding_strategy(sharding_strategy)
    mp_policy = get_mixed_precision_policy(fsdp_precision)

    if auto_wrap_policy is None:
        auto_wrap_policy = functools.partial(
            size_based_auto_wrap_policy, min_num_params=1_000_000
        )

    # For HYBRID sharding strategies, provide an explicit 2D device mesh so
    # that FSDP shards within a node (LOCAL_WORLD_SIZE ranks) and replicates
    # across nodes. Without this, FSDP's automatic subgroup construction can
    # behave inconsistently on multi-node setups.
    fsdp_extra_kwargs: dict = {}
    if strategy in (ShardingStrategy.HYBRID_SHARD, ShardingStrategy._HYBRID_SHARD_ZERO2):
        try:
            mesh = get_device_mesh(use_hybrid=True)
            fsdp_extra_kwargs["device_mesh"] = mesh
        except Exception:
            # Fall back to FSDP's default subgroup construction.
            pass

    if ignored_modules:
        fsdp_extra_kwargs["ignored_modules"] = ignored_modules

    cpu_offload_policy = CPUOffload(offload_params=True) if cpu_offload else None

    wrapped = FSDP(
        model,
        sharding_strategy=strategy,
        mixed_precision=mp_policy,
        auto_wrap_policy=auto_wrap_policy,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        device_id=torch.device("cuda", device_id),
        limit_all_gathers=True,
        use_orig_params=use_orig_params,
        cpu_offload=cpu_offload_policy,
        **fsdp_extra_kwargs,
    )
    return wrapped


def free_fsdp_unsharded_params(module: torch.nn.Module | None) -> None:
    """Release FSDP1 unsharded parameters after an inference-only forward.

    PyTorch FSDP1 does not expose a public ``reshard()`` on the wrapper (calling
    ``module.reshard()`` delegates to the inner nn.Module and fails). Use the
    internal reshard helper with ``free_unsharded_flat_param=True`` so a second
    large FSDP model can unshard without peak OOM.
    """
    if module is None or not isinstance(module, FSDP):
        return
    handle = getattr(module, "_handle", None)
    if handle is None:
        return
    from torch.distributed.fsdp._runtime_utils import _reshard as _fsdp_internal_reshard

    _fsdp_internal_reshard(module, handle, free_unsharded_flat_param=True)


def get_transformer_wrap_policy(transformer_block_cls: type | set[type] | tuple[type, ...]) -> Callable:
    """Create an FSDP auto-wrap policy that wraps at transformer block boundaries.

    This is the recommended wrapping granularity for transformer models,
    as it provides good memory/communication tradeoff.

    Args:
        transformer_block_cls: The class (or classes) of transformer blocks to wrap at
            (e.g., SD3TransformerBlock, Flux2TransformerBlock).

    Returns:
        A callable auto-wrap policy for FSDP.
    """
    if isinstance(transformer_block_cls, type):
        transformer_layer_cls = {transformer_block_cls}
    else:
        transformer_layer_cls = set(transformer_block_cls)

    return functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls=transformer_layer_cls,
    )


# ---------------------------------------------------------------------------
# DDP wrapping
# ---------------------------------------------------------------------------

def ddp_wrap_model(
    model: torch.nn.Module,
    device_id: int | None = None,
    find_unused_parameters: bool = False,
    broadcast_buffers: bool = True,
) -> torch.nn.parallel.DistributedDataParallel:
    """Wrap a model with DDP.

    Args:
        model: The PyTorch module to wrap (should already be on the correct device).
        device_id: CUDA device ID. Defaults to LOCAL_RANK.
        find_unused_parameters: Whether DDP should find unused parameters.
        broadcast_buffers: Whether DDP broadcasts module buffers before forward.
            Keep True by default for behavior parity with vanilla DDP.

    Returns:
        DDP-wrapped model.
    """
    if device_id is None:
        device_id = get_local_rank()
    return torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[device_id],
        find_unused_parameters=find_unused_parameters,
        broadcast_buffers=broadcast_buffers,
    )


# ---------------------------------------------------------------------------
# Device mesh (for hybrid sharding across nodes)
# ---------------------------------------------------------------------------

def get_device_mesh(use_hybrid: bool = False) -> torch.distributed.device_mesh.DeviceMesh:
    """Initialize a device mesh for FSDP hybrid sharding.

    For multi-node training, hybrid sharding shards within a node and
    replicates across nodes, reducing inter-node communication.

    Args:
        use_hybrid: If True, creates a 2D mesh (replicate x shard).
            If False, creates a 1D mesh (shard only).

    Returns:
        DeviceMesh instance.
    """
    world_size = get_world_size()
    if use_hybrid:
        n_nodes = world_size // get_local_world_size()
        local_size = get_local_world_size()
        mesh = init_device_mesh(
            "cuda",
            (n_nodes, local_size),
            mesh_dim_names=("replicate", "shard"),
        )
    else:
        mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("shard",))
    return mesh
