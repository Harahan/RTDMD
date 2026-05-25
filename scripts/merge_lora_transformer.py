#!/usr/bin/env python3
"""
Merge a PEFT LoRA adapter into a transformer and save merged weights.

Supports all backbones used in this repo:

    * sd35          -> SD3Transformer2DModel        (SD3 / SD3.5 Medium)
    * flux          -> FluxTransformer2DModel       (FLUX.1-dev)
    * flux2_klein   -> Flux2Transformer2DModel      (FLUX.2-klein 4B / 9B)

The model class is auto-detected from the base transformer's ``config.json``
(``_class_name`` field). Override with ``--model_type`` when needed.

Examples:
    # SD3.5 Medium
    python scripts/merge_lora_transformer.py \\
        --pretrained_path stabilityai/stable-diffusion-3.5-medium \\
        --lora_path /path/to/checkpoint-15000/transformer \\
        --output_path /path/to/merged/sd35m_transformer

    # FLUX.1-dev
    python scripts/merge_lora_transformer.py \\
        --pretrained_path black-forest-labs/FLUX.1-dev \\
        --lora_path /path/to/checkpoint-15000/transformer \\
        --output_path /path/to/merged/flux1_dev_transformer

    # FLUX.2-klein 4B (with explicit model_type override)
    python scripts/merge_lora_transformer.py \\
        --pretrained_path black-forest-labs/FLUX.2-klein-base-4B \\
        --lora_path /path/to/checkpoint-15000/transformer \\
        --output_path /path/to/merged/flux2_4b_transformer \\
        --model_type flux2_klein
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from diffusers import (
    Flux2Transformer2DModel,
    FluxTransformer2DModel,
    SD3Transformer2DModel,
)
from peft.tuners.tuners_utils import BaseTunerLayer

# Make project root importable when running as a standalone script.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rtdmd.config import LoRAConfig
from rtdmd.utils.checkpoint import is_peft_lora_dir, load_lora_peft_format
from rtdmd.utils.lora import inject_lora


# Maps `model_type` (CLI value) -> transformer class.
TRANSFORMER_CLASSES = {
    "sd35":        SD3Transformer2DModel,
    "flux":        FluxTransformer2DModel,
    "flux2_klein": Flux2Transformer2DModel,
}

# Maps the diffusers `_class_name` field -> `model_type`.
_CLASS_NAME_TO_MODEL_TYPE = {
    "SD3Transformer2DModel":   "sd35",
    "FluxTransformer2DModel":  "flux",
    "Flux2Transformer2DModel": "flux2_klein",
}

# Default LoRA `target_modules` fallback (only used when the adapter dir's
# `adapter_config.json` is missing the field; in practice PEFT always writes
# it, so this is just a safety net).
_DEFAULT_TARGET_MODULES = {
    "sd35": [
        "attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0",
        "attn.add_k_proj", "attn.add_v_proj", "attn.add_q_proj", "attn.to_add_out",
    ],
    "flux": [
        "attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0",
        "attn.add_q_proj", "attn.add_k_proj", "attn.add_v_proj", "attn.to_add_out",
    ],
    "flux2_klein": [
        "to_q", "to_k", "to_v", "to_out.0",
        "add_q_proj", "add_k_proj", "add_v_proj", "to_add_out",
        "to_qkv_mlp_proj",
    ],
}


def _get_dtype(dtype_str: str) -> torch.dtype:
    dtype_map = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    if dtype_str not in dtype_map:
        raise ValueError(f"Unsupported dtype: {dtype_str}. Choose from fp32/fp16/bf16.")
    return dtype_map[dtype_str]


def _resolve_transformer_dir(path: str, desc: str) -> str:
    """Resolve the path to a diffusers transformer subdir.

    Accepts both:
      1) the transformer dir itself (``<path>/config.json``)
      2) a parent checkpoint / pipeline dir (``<path>/transformer/config.json``)
    """
    if os.path.exists(os.path.join(path, "config.json")):
        return path
    nested = os.path.join(path, "transformer")
    if os.path.exists(os.path.join(nested, "config.json")):
        return nested
    raise FileNotFoundError(
        f"Cannot find transformer config.json for {desc}: {path}. "
        "Expected either <path>/config.json or <path>/transformer/config.json."
    )


def _autodetect_model_type(transformer_dir: str) -> str | None:
    """Read ``_class_name`` from ``config.json`` and map to a model_type key."""
    cfg_path = os.path.join(transformer_dir, "config.json")
    try:
        with open(cfg_path, "r") as f:
            cfg = json.load(f)
    except OSError:
        return None
    class_name = str(cfg.get("_class_name", "")).strip()
    return _CLASS_NAME_TO_MODEL_TYPE.get(class_name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge a PEFT LoRA into a SD3 / FLUX.1 / FLUX.2-klein transformer.",
    )
    parser.add_argument(
        "--pretrained_path",
        type=str,
        required=True,
        help="Base pretrained model directory or HF Hub repo id (containing `transformer/`).",
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        required=True,
        help="PEFT LoRA directory (with adapter_config.json + adapter_model.safetensors).",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Output directory for the merged transformer (diffusers format).",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        choices=sorted(TRANSFORMER_CLASSES.keys()),
        help=(
            "Backbone family. If unset, auto-detected from the base transformer's "
            "config.json (`_class_name`)."
        ),
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bf16",
        choices=["fp32", "fp16", "bf16"],
        help="Load dtype for the base transformer (default: bf16).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    base_transformer_dir = _resolve_transformer_dir(args.pretrained_path, "pretrained_path")

    model_type = args.model_type or _autodetect_model_type(base_transformer_dir)
    if model_type is None:
        raise ValueError(
            f"Could not auto-detect model_type from {base_transformer_dir}/config.json. "
            f"Please pass --model_type explicitly (one of: "
            f"{sorted(TRANSFORMER_CLASSES.keys())})."
        )
    transformer_cls = TRANSFORMER_CLASSES[model_type]

    lora_dir = args.lora_path
    if not is_peft_lora_dir(lora_dir):
        raise FileNotFoundError(
            f"Invalid LoRA dir: {lora_dir}. "
            "Expected PEFT directory with adapter_config.json and adapter_model.safetensors."
        )

    dtype = _get_dtype(args.dtype)

    print(f"[1/4] Loading base {transformer_cls.__name__} from: {base_transformer_dir}")
    transformer = transformer_cls.from_pretrained(base_transformer_dir, torch_dtype=dtype)

    print(f"[2/4] Loading LoRA adapter from: {lora_dir}")
    adapter_cfg_path = os.path.join(lora_dir, "adapter_config.json")
    with open(adapter_cfg_path, "r") as f:
        adapter_cfg = json.load(f)
    lora_cfg = LoRAConfig(
        enabled=True,
        rank=adapter_cfg.get("r", 32),
        lora_alpha=adapter_cfg.get("lora_alpha", 64),
        target_modules=adapter_cfg.get(
            "target_modules", _DEFAULT_TARGET_MODULES[model_type]
        ),
        init_lora_weights=adapter_cfg.get("init_lora_weights", "gaussian"),
    )
    inject_lora(transformer, lora_cfg)
    load_lora_peft_format(transformer, lora_dir)

    print("[3/4] Merging LoRA into base weights")
    for module in transformer.modules():
        if isinstance(module, BaseTunerLayer):
            module.merge()

    # Materialize the merged plain state_dict into a fresh base transformer so
    # the output dir has a clean, peft-free checkpoint.
    merged_state = transformer.state_dict()
    merged_transformer = transformer_cls.from_pretrained(
        base_transformer_dir,
        torch_dtype=dtype,
    )
    merged_transformer.load_state_dict(merged_state, strict=False)

    os.makedirs(args.output_path, exist_ok=True)
    print(f"[4/4] Saving merged transformer ({model_type}) to: {args.output_path}")
    merged_transformer.save_pretrained(args.output_path)

    print("Done.")


if __name__ == "__main__":
    main()
