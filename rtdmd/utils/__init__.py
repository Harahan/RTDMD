"""Shared utilities: logging, checkpointing, LoRA, EMA, fast init, image decode.

Public API (re-exported from the corresponding submodule):

Logging (:mod:`rtdmd.utils.logging`):
    WandbLogger, Timer, get_gpu_memory_gb, setup_logging

Checkpointing (:mod:`rtdmd.utils.checkpoint`):
    save_checkpoint, load_checkpoint,
    save_lora_peft_format, load_lora_peft_format, is_peft_lora_dir

LoRA (:mod:`rtdmd.utils.lora`):
    inject_lora, setup_lora

EMA (:mod:`rtdmd.utils.ema`):
    EMAModuleWrapper

Fast model init (:mod:`rtdmd.utils.fast_init`):
    fast_init

Image / latent helpers (:mod:`rtdmd.utils.image`):
    decode_latents_to_tensor, decode_latents_to_pil

Internal helpers prefixed with ``_`` (e.g. ``_resolve_lora_config``,
``_gather_fsdp_state_dict``) are not re-exported here -- import them from
the submodule directly if you really need them.
"""

from rtdmd.utils.checkpoint import (
    is_peft_lora_dir,
    load_checkpoint,
    load_lora_peft_format,
    save_checkpoint,
    save_lora_peft_format,
)
from rtdmd.utils.ema import EMAModuleWrapper
from rtdmd.utils.fast_init import fast_init
from rtdmd.utils.image import decode_latents_to_pil, decode_latents_to_tensor
from rtdmd.utils.logging import (
    Timer,
    WandbLogger,
    get_gpu_memory_gb,
    setup_logging,
)
from rtdmd.utils.lora import inject_lora, setup_lora

__all__ = [
    "EMAModuleWrapper",
    "Timer",
    "WandbLogger",
    "decode_latents_to_pil",
    "decode_latents_to_tensor",
    "fast_init",
    "get_gpu_memory_gb",
    "inject_lora",
    "is_peft_lora_dir",
    "load_checkpoint",
    "load_lora_peft_format",
    "save_checkpoint",
    "save_lora_peft_format",
    "setup_lora",
    "setup_logging",
]
