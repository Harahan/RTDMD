"""
Checkpoint utilities for RTDMD.

Saves checkpoints in a split format:
  checkpoint-{step}/
  ├── transformer/              # diffusers-format generator (save_pretrained)
  ├── fake_score_net.pt         # fake score network state_dict
  ├── optimizer_generator.pt    # generator optimizer state_dict
  ├── optimizer_fake_score.pt   # fake score optimizer state_dict
  └── meta.pt                   # step, LR schedulers, RNG states

The generator is saved in standard diffusers format so it can be loaded
directly with SD3Transformer2DModel.from_pretrained() for inference.

When LoRA is enabled, LoRA weights are saved in standard peft format
(adapter_config.json + adapter_model.safetensors) so they can be loaded
with pipe.load_lora_weights(). Optimizer states remain as .pt files.

Future extensions:
- Distributed checkpointing (torch.distributed.checkpoint) for large models
- Automatic checkpoint management (keep last N, best K by metric)
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, TYPE_CHECKING

import torch
from safetensors.torch import save_file as safetensors_save_file
from safetensors.torch import load_file as safetensors_load_file
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    FullStateDictConfig,
    StateDictType,
)
from torch.nn.parallel import DistributedDataParallel as DDP

if TYPE_CHECKING:
    from rtdmd.config import LoRAConfig

from rtdmd.parallel.utils import is_main_process, barrier

logger = logging.getLogger(__name__)


def _gather_fsdp_state_dict(model: torch.nn.Module) -> dict:
    """Gather a full, wrapper-free state dict from an arbitrary model.

    For FSDP models, gathers shards to rank 0 with CPU offloading.
    For DDP models, returns the inner module's state dict (strips the
    ``module.`` prefix that ``DDP.state_dict()`` would otherwise add) so the
    on-disk key naming is identical to the unwrapped transformer regardless of
    the distributed strategy used at training time.
    For plain modules, returns ``state_dict()`` directly.
    """
    if isinstance(model, FSDP):
        with FSDP.state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
        ):
            return model.state_dict()
    if isinstance(model, DDP):
        return model.module.state_dict()
    return model.state_dict()


def _filter_lora_state_dict(state_dict: dict) -> dict:
    """Filter a state_dict to only keep LoRA parameters (lora_A/lora_B).

    Args:
        state_dict: Full model state_dict.

    Returns:
        Filtered state_dict containing only LoRA adapter weights.
    """
    return {k: v for k, v in state_dict.items()
            if "lora_A" in k or "lora_B" in k}


def _is_lora_checkpoint(state_dict: dict) -> bool:
    """Detect if a state_dict is a LoRA-only checkpoint.

    A LoRA checkpoint contains only lora_A/lora_B keys and is much smaller
    than a full model checkpoint. We check if ALL keys contain lora_ patterns.

    Args:
        state_dict: A loaded state_dict.

    Returns:
        True if this appears to be a LoRA-only checkpoint.
    """
    if not state_dict:
        return False
    return all("lora_A" in k or "lora_B" in k for k in state_dict.keys())


def _strip_ddp_module_prefix(state_dict: dict) -> dict:
    """Remove the ``module.`` prefix added by older DDP-saved checkpoints.

    Older checkpoints saved before the DDP-aware ``_gather_fsdp_state_dict``
    fix store keys with a ``module.`` prefix (because ``DDP.state_dict()`` adds
    it). New checkpoints save without that prefix so the on-disk key naming is
    consistent across FSDP / DDP / single-GPU.

    To preserve forward compatibility we always strip the prefix on load when
    every key starts with it; new prefix-free checkpoints are returned
    unchanged. The downstream ``load_state_dict`` is invoked against the inner
    (unwrapped) module so the stripped keys match.
    """
    if not state_dict:
        return state_dict
    if all(k.startswith("module.") for k in state_dict.keys()):
        prefix_len = len("module.")
        return {k[prefix_len:]: v for k, v in state_dict.items()}
    return state_dict


def _load_state_dict_unwrapped(
    model: torch.nn.Module,
    state_dict: dict,
    strict: bool,
) -> None:
    """Load a wrapper-free state_dict into a possibly-wrapped model.

    Mirrors :func:`_gather_fsdp_state_dict` semantics on the load side:
    - FSDP-wrapped models are loaded via ``FSDP.state_dict_type``.
    - DDP-wrapped models load through ``model.module.load_state_dict``.
    - Plain modules load directly.

    Legacy DDP checkpoints that still carry a ``module.`` prefix are normalized
    via :func:`_strip_ddp_module_prefix` before the inner load, so both old and
    new checkpoint formats interoperate.
    """
    state_dict = _strip_ddp_module_prefix(state_dict)
    if isinstance(model, FSDP):
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT):
            model.load_state_dict(state_dict, strict=strict)
        return
    inner = model.module if isinstance(model, DDP) else model
    inner.load_state_dict(state_dict, strict=strict)


def save_lora_peft_format(
    state_dict: dict,
    lora_config: LoRAConfig,
    save_dir: str,
) -> None:
    """Save LoRA weights in standard peft format.

    Creates a directory with:
      - adapter_config.json: peft LoRA configuration
      - adapter_model.safetensors: LoRA weights (lora_A/lora_B only)

    This format is compatible with:
      - pipe.load_lora_weights(save_dir)
      - PeftModel.from_pretrained(model, save_dir)

    Args:
        state_dict: Full model state_dict (will be filtered to LoRA keys).
        lora_config: LoRA configuration used during training.
        save_dir: Target directory for the peft-format output.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Filter to LoRA-only keys
    lora_state = _filter_lora_state_dict(state_dict)

    # Save weights in safetensors format
    safetensors_save_file(lora_state, os.path.join(save_dir, "adapter_model.safetensors"))

    # Write adapter_config.json (peft-compatible)
    adapter_config = {
        "r": lora_config.rank,
        "lora_alpha": lora_config.lora_alpha,
        "target_modules": list(lora_config.target_modules),
        "lora_dropout": 0.0,
        "bias": "none",
        "peft_type": "LORA",
        "task_type": None,
        "init_lora_weights": lora_config.init_lora_weights,
        "use_rslora": False,
    }
    with open(os.path.join(save_dir, "adapter_config.json"), "w") as f:
        json.dump(adapter_config, f, indent=2)

    logger.info(f"  LoRA saved in peft format: {save_dir} ({len(lora_state)} keys)")


def load_lora_peft_format(
    model: torch.nn.Module,
    lora_dir: str,
    map_location: str = "cpu",
) -> None:
    """Load LoRA weights from peft format directory.

    Loads adapter_model.safetensors and applies to the model with strict=False.

    Args:
        model: Model with LoRA adapters already injected.
        lora_dir: Path to directory containing adapter_model.safetensors.
        map_location: Device to map tensors to.
    """
    safetensors_path = os.path.join(lora_dir, "adapter_model.safetensors")
    if os.path.exists(safetensors_path):
        state_dict = safetensors_load_file(safetensors_path, device=map_location)
    else:
        # Fallback to legacy .pt format
        pt_path = os.path.join(lora_dir, "adapter_model.pt")
        if os.path.exists(pt_path):
            state_dict = torch.load(pt_path, map_location=map_location, weights_only=False)
        else:
            raise FileNotFoundError(
                f"No adapter_model.safetensors or adapter_model.pt found in {lora_dir}"
            )
    _load_state_dict_unwrapped(model, state_dict, strict=False)
    logger.info(f"  Loaded LoRA weights from peft format: {lora_dir} ({len(state_dict)} keys)")


def is_peft_lora_dir(path: str) -> bool:
    """Check if a path is a peft-format LoRA directory.

    A peft LoRA directory contains adapter_config.json and adapter_model.safetensors.
    """
    return os.path.isdir(path) and (
        os.path.exists(os.path.join(path, "adapter_model.safetensors"))
        or os.path.exists(os.path.join(path, "adapter_config.json"))
    )


def _maybe_restore_conv2d_patch_embed(
    state_dict: dict,
    target_model: torch.nn.Module,
) -> dict:
    """Reshape a flattened Linear PatchEmbed.proj weight back to Conv2d 4-D shape.

    When training uses ``model.fsdp_patch_embed_linear=True``, the patched model
    has ``pos_embed.proj`` as ``nn.Linear`` and its ``state_dict`` contains a
    2-D weight of shape ``[out_ch, in_ch*kH*kW]``. Vanilla
    ``SD3Transformer2DModel`` (and any code that round-trips through diffusers'
    config.json instantiation) expects a 4-D Conv2d weight of shape
    ``[out_ch, in_ch, kH, kW]``. This helper reshapes the saved 2-D weight back
    to the 4-D Conv2d shape so ``load_state_dict`` succeeds against a fresh
    ``SD3Transformer2DModel``.

    Returns a (shallow-copied) state_dict with the reshape applied. The input
    is NOT mutated, so callers can keep using the original Linear-shaped
    state_dict afterwards (e.g., for ``.pt`` saving where the on-disk shape
    must match the in-memory patched model for resume).

    Safe to call unconditionally — no-op when:
      * target model has no ``pos_embed.proj`` (e.g., Flux2),
      * target ``pos_embed.proj`` is already a Linear (no patch needed),
      * saved weight is already 4-D,
      * saved weight shape doesn't match the target Conv2d's flattened shape.
    """
    if not hasattr(target_model, "pos_embed"):
        return state_dict
    patch_embed = target_model.pos_embed
    if not hasattr(patch_embed, "proj"):
        return state_dict
    target_proj = patch_embed.proj
    if not isinstance(target_proj, torch.nn.Conv2d):
        return state_dict
    weight_key = "pos_embed.proj.weight"
    if weight_key not in state_dict:
        return state_dict
    saved_w = state_dict[weight_key]
    if saved_w.ndim != 2:
        return state_dict
    out_ch, in_ch, kH, kW = target_proj.weight.shape
    if saved_w.shape != (out_ch, in_ch * kH * kW):
        return state_dict
    restored = dict(state_dict)
    restored[weight_key] = saved_w.view(out_ch, in_ch, kH, kW).contiguous()
    logger.info(
        f"  Restored Linear PatchEmbed weight {tuple(saved_w.shape)} -> "
        f"Conv2d shape {(out_ch, in_ch, kH, kW)} for diffusers save"
    )
    return restored


def _save_diffusers_transformer(
    state_dict: dict,
    config_path: str,
    save_dir: str,
) -> None:
    """Save a transformer state_dict in diffusers format.

    Creates a fresh model instance from config (to avoid FSDP sharded parameter
    issues), loads the gathered state_dict, and calls save_pretrained().

    If the saved state_dict was produced by a model with the SD3-Medium FSDP
    PatchEmbed Conv2d -> Linear workaround (``fsdp_patch_embed_linear=True``),
    the 2-D Linear weight is reshaped back to the 4-D Conv2d shape on the fly,
    so the resulting ``transformer/`` folder is a valid drop-in for vanilla
    ``SD3Transformer2DModel.from_pretrained()`` regardless of how the model
    was trained.

    Args:
        state_dict: Full (unsharded) model state_dict.
        config_path: Path to the pretrained model dir containing transformer/config.json.
        save_dir: Target directory for the diffusers-format output.
    """
    import diffusers
    import json
    from rtdmd.utils.fast_init import fast_init

    # Load config from the original pretrained model
    config_json = os.path.join(config_path, "transformer", "config.json")
    if not os.path.exists(config_json):
        logger.warning(
            f"Cannot save diffusers format: config not found at {config_json}. "
            "Skipping save_pretrained."
        )
        return

    # Create a fresh (unsharded) model from config, load weights, save
    # Use fast_init to avoid RNG consumption (this runs inside the RNG bracket)
    with open(config_json, "r") as f:
        config = json.load(f)
    class_name = config.get("_class_name", "SD3Transformer2DModel")
    model_cls = getattr(diffusers, class_name, None)
    if model_cls is None:
        logger.warning(
            f"Cannot save diffusers format: unsupported transformer class '{class_name}' "
            f"from {config_json}. Skipping save_pretrained."
        )
        return

    with fast_init(torch.device("cpu")):
        model = model_cls(**{k: v for k, v in config.items() if not k.startswith("_")})
    # Reshape Linear PatchEmbed weight back to Conv2d shape if needed. The
    # original state_dict is not mutated; resume .pt files keep Linear shape.
    state_dict_for_load = _maybe_restore_conv2d_patch_embed(state_dict, model)
    model.load_state_dict(state_dict_for_load)
    model.save_pretrained(save_dir)


def _resolve_lora_config(
    model_name: str,
    lora_configs: dict[str, "LoRAConfig"],
) -> "LoRAConfig | None":
    """Resolve the LoRAConfig for a given model name.

    Handles aliasing: generator_ema uses the same LoRA config as generator.

    Args:
        model_name: Name of the model (e.g., "generator", "generator_ema",
            "fake_score_net").
        lora_configs: Dict mapping model name -> LoRAConfig.

    Returns:
        The matching LoRAConfig, or None if not found.
    """
    if model_name in lora_configs:
        return lora_configs[model_name]
    # generator_ema shares config with generator
    if model_name == "generator_ema" and "generator" in lora_configs:
        return lora_configs["generator"]
    return None


def save_checkpoint(
    output_dir: str,
    step: int,
    models: dict[str, torch.nn.Module],
    optimizers: dict[str, torch.optim.Optimizer],
    extra_state: dict[str, Any] | None = None,
    pretrained_path: str = "",
    lora_models: set[str] | None = None,
    lora_configs: dict[str, "LoRAConfig"] | None = None,
    extra_state_dicts: dict[str, dict] | None = None,
) -> None:
    """Save a training checkpoint in split format.

    Saves each component as a separate file to avoid a single huge file.

    For full-weight models:
    - generator is saved in diffusers format (transformer/ subfolder)
    - all models saved as .pt state_dicts

    For LoRA models:
    - Resume files: {name}.pt containing only LoRA keys (fast, for resume)
    - Inference files: transformer/ subfolder in peft format
      (adapter_config.json + adapter_model.safetensors), just like full-weight
      saves a transformer/ folder. Prefers EMA generator if available.

    Optimizer states are always saved as .pt files regardless of LoRA.

    All ranks participate in FSDP state dict gathering, but only rank 0 writes.

    Args:
        output_dir: Base output directory.
        step: Current training step.
        models: Dict of named models to save (state_dicts).
        optimizers: Dict of named optimizers to save.
        extra_state: Additional state (step, schedulers, rng_state, etc.).
        pretrained_path: Path to the original pretrained model dir (for reading
            transformer/config.json). If provided, saves generator in diffusers
            format under transformer/ subfolder.
        lora_models: Set of model names that use LoRA. Their .pt files will
            contain only LoRA keys (for resume).
        lora_configs: Dict mapping model name -> LoRAConfig for peft format
            metadata. Required when lora_models is non-empty.
        extra_state_dicts: Optional pre-gathered state dicts (typically rank-0
            only) to be saved alongside the regular models. Each entry is
            written as ``{name}.pt`` and participates in the ``transformer/``
            selection rule (e.g., "generator_ema" wins over "generator"). Used
            by GRPO-style trainers to dump EMA snapshots without needing a
            separate ``nn.Module`` in ``self.models``.
    """
    if lora_models is None:
        lora_models = set()
    if lora_configs is None:
        lora_configs = {}
    if extra_state_dicts is None:
        extra_state_dicts = {}

    ckpt_dir = os.path.join(output_dir, f"checkpoint-{step}")

    # Gather model state dicts (all ranks participate for FSDP)
    model_states = {}
    for name, model in models.items():
        model_states[name] = _gather_fsdp_state_dict(model)

    # Gather optimizer state dicts (all ranks participate for FSDP)
    # Under FSDP, optimizer.state_dict() returns rank-local sharded state.
    # We must use FSDP.full_optim_state_dict() to gather the full state to rank 0.
    optimizer_states = {}
    for name, optimizer in optimizers.items():
        # Find the corresponding FSDP model for this optimizer.
        # Convention: optimizer name matches model name, or model name + "_net" suffix.
        fsdp_model = None
        for model_name, model in models.items():
            if isinstance(model, FSDP) and (
                model_name == name or model_name.startswith(name)
            ):
                fsdp_model = model
                break
        if fsdp_model is not None:
            optimizer_states[name] = FSDP.full_optim_state_dict(fsdp_model, optimizer)
        else:
            optimizer_states[name] = optimizer.state_dict()

    # Only rank 0 writes to disk
    if is_main_process():
        os.makedirs(ckpt_dir, exist_ok=True)

        # Merge in pre-gathered state dicts (e.g. EMA snapshots). These have
        # already been gathered to rank 0 by the caller, so they only appear
        # here and never participate in the collective gather above.
        for name, sd in extra_state_dicts.items():
            if name in model_states and model_states[name]:
                raise ValueError(
                    f"Duplicate state dict name '{name}': both passed via "
                    "`models` and `extra_state_dicts`. Use a different name "
                    "for the extra entry."
                )
            model_states[name] = sd

        # --- Determine which generator state to use for the transformer/ folder ---
        # Prefer EMA generator if available (more stable for inference)
        gen_source_name = None
        if "generator_ema" in model_states:
            gen_source_name = "generator_ema"
        elif "generator" in model_states:
            gen_source_name = "generator"

        has_lora_generator = any(
            name in lora_models
            for name in ["generator", "generator_ema"]
            if name in model_states
        )

        # 1. Save transformer/ folder for easy inference loading
        if gen_source_name:
            transformer_dir = os.path.join(ckpt_dir, "transformer")
            if has_lora_generator:
                # LoRA mode: save peft format in transformer/ folder
                lora_cfg = _resolve_lora_config(gen_source_name, lora_configs)
                if lora_cfg is not None:
                    save_lora_peft_format(
                        model_states[gen_source_name], lora_cfg, transformer_dir,
                    )
                    logger.info(
                        f"  {gen_source_name} LoRA saved in peft format: {transformer_dir}"
                    )
                else:
                    logger.warning(
                        f"  No LoRAConfig for {gen_source_name}, "
                        f"skipping transformer/ peft save"
                    )
            elif pretrained_path:
                # Full-weight mode: save in diffusers format
                _save_diffusers_transformer(
                    state_dict=model_states[gen_source_name],
                    config_path=pretrained_path,
                    save_dir=transformer_dir,
                )
                logger.info(
                    f"  {gen_source_name} saved in diffusers format: {transformer_dir}"
                )

        # 2. Save each model state dict as .pt (for resume)
        #    For LoRA models: filter to LoRA keys only
        #    For full-weight models: save full state_dict
        for name, state_dict in model_states.items():
            if name in lora_models:
                lora_state = _filter_lora_state_dict(state_dict)
                path = os.path.join(ckpt_dir, f"{name}.pt")
                torch.save(lora_state, path)
                logger.info(
                    f"  Model saved (LoRA resume): {path} ({len(lora_state)} keys)"
                )
            else:
                path = os.path.join(ckpt_dir, f"{name}.pt")
                torch.save(state_dict, path)
                logger.info(f"  Model saved: {path}")

        # 3. Save each optimizer as separate files
        for name, state_dict in optimizer_states.items():
            path = os.path.join(ckpt_dir, f"optimizer_{name}.pt")
            torch.save(state_dict, path)
            logger.info(f"  Optimizer saved: {path}")

        # 4. Save meta state (step, schedulers, rng)
        meta = {"step": step}
        if extra_state:
            meta.update(extra_state)
        meta_path = os.path.join(ckpt_dir, "meta.pt")
        torch.save(meta, meta_path)
        logger.info(f"  Meta saved: {meta_path}")

        logger.info(f"Checkpoint saved: {ckpt_dir} (step={step})")

    barrier()


def load_checkpoint(
    ckpt_path: str,
    models: dict[str, torch.nn.Module],
    optimizers: dict[str, torch.optim.Optimizer] | None = None,
    map_location: str = "cpu",
) -> dict[str, Any]:
    """Load a training checkpoint.

    Supports both the new split format and legacy single-file format.

    Args:
        ckpt_path: Path to the checkpoint directory or legacy file.
        models: Dict of named models to load state into.
        optimizers: Optional dict of named optimizers to load state into.
        map_location: Device to map tensors to during loading.

    Returns:
        Extra state dict (contains 'step', 'schedulers', 'rng_state', etc.).
    """
    # Determine format: split (directory with meta.pt) or legacy (single .pt)
    if os.path.isdir(ckpt_path):
        meta_path = os.path.join(ckpt_path, "meta.pt")
        if os.path.exists(meta_path):
            return _load_split_checkpoint(ckpt_path, models, optimizers, map_location)
        # Legacy: directory containing checkpoint.pt
        legacy_path = os.path.join(ckpt_path, "checkpoint.pt")
        if os.path.exists(legacy_path):
            return _load_legacy_checkpoint(legacy_path, models, optimizers, map_location)
        raise FileNotFoundError(
            f"Checkpoint directory {ckpt_path} contains neither meta.pt nor checkpoint.pt"
        )
    elif os.path.isfile(ckpt_path):
        return _load_legacy_checkpoint(ckpt_path, models, optimizers, map_location)
    else:
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")


def _load_split_checkpoint(
    ckpt_dir: str,
    models: dict[str, torch.nn.Module],
    optimizers: dict[str, torch.optim.Optimizer] | None,
    map_location: str,
) -> dict[str, Any]:
    """Load from the new split checkpoint format."""
    logger.info(f"Loading split checkpoint from: {ckpt_dir}")

    # Load model states
    for name, model in models.items():
        path = os.path.join(ckpt_dir, f"{name}.pt")
        if os.path.exists(path):
            state_dict = torch.load(path, map_location=map_location, weights_only=False)
            # LoRA checkpoints contain only lora_A/lora_B keys — use strict=False
            is_lora = _is_lora_checkpoint(state_dict)
            _load_state_dict_unwrapped(model, state_dict, strict=not is_lora)
            suffix = " (LoRA partial)" if is_lora else ""
            logger.info(f"  Loaded model: {name}{suffix}")
        else:
            logger.warning(f"  Model file not found: {path}, skipping")

    # Load optimizer states
    # Under FSDP, the saved state is the full (gathered) optimizer state.
    # We must use FSDP.optim_state_dict_to_load() to scatter it back to shards.
    if optimizers:
        for name, optimizer in optimizers.items():
            path = os.path.join(ckpt_dir, f"optimizer_{name}.pt")
            if os.path.exists(path):
                full_osd = torch.load(path, map_location=map_location, weights_only=False)
                # Find the corresponding FSDP model
                fsdp_model = None
                for model_name, model in models.items():
                    if isinstance(model, FSDP) and (
                        model_name == name or model_name.startswith(name)
                    ):
                        fsdp_model = model
                        break
                if fsdp_model is not None:
                    # Scatter the full optimizer state back to FSDP shards
                    sharded_osd = FSDP.optim_state_dict_to_load(
                        fsdp_model, optimizer, full_osd,
                    )
                    optimizer.load_state_dict(sharded_osd)
                else:
                    optimizer.load_state_dict(full_osd)
                logger.info(f"  Loaded optimizer: {name}")
            else:
                logger.warning(f"  Optimizer file not found: {path}, skipping")

    # Load meta state
    meta_path = os.path.join(ckpt_dir, "meta.pt")
    meta = torch.load(meta_path, map_location=map_location, weights_only=False)
    logger.info(f"  Resumed from step {meta.get('step', 'unknown')}")
    return meta


def _load_legacy_checkpoint(
    ckpt_path: str,
    models: dict[str, torch.nn.Module],
    optimizers: dict[str, torch.optim.Optimizer] | None,
    map_location: str,
) -> dict[str, Any]:
    """Load from the legacy single-file checkpoint format (backward compat)."""
    logger.info(f"Loading legacy checkpoint from: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=map_location, weights_only=False)

    # Load model states
    for name, model in models.items():
        key = f"model_{name}"
        if key in state:
            _load_state_dict_unwrapped(model, state[key], strict=True)
            logger.info(f"  Loaded model: {name}")
        else:
            logger.warning(f"  Model '{name}' not found in checkpoint, skipping")

    # Load optimizer states
    if optimizers:
        for name, optimizer in optimizers.items():
            key = f"optimizer_{name}"
            if key in state:
                optimizer.load_state_dict(state[key])
                logger.info(f"  Loaded optimizer: {name}")
            else:
                logger.warning(f"  Optimizer '{name}' not found in checkpoint, skipping")

    # Return extra state
    extra = {k: v for k, v in state.items()
             if not k.startswith("model_") and not k.startswith("optimizer_")}
    logger.info(f"  Resumed from step {extra.get('step', 'unknown')}")
    return extra
