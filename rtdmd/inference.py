"""
RTDMD Inference Engine.

Supports two inference modes:
- ``distilled``: few-step generation with the CPS scheduler
  (``cps_eta = 0`` is Euler-equivalent; ``cps_eta > 0`` injects fresh
  noise per step).
- ``normal``: standard multi-step generation
  (FlowMatchEulerDiscreteScheduler with optional CFG).

The distilled loop is the inference-side mirror of
``DMDTrainer._generate_latents`` so eval-time and stand-alone numerics agree.

Three LoRA modes (selected by ``lora_paths`` length):
- ``[]``         -- no LoRA, plain inference on the pretrained checkpoint
- ``[distilled]``-- single distilled LoRA merged into the base weights
- ``[distilled, rl]`` -- distilled + RL LoRAs merged in order

Reward evaluation (``eval_reward = true``) supports per-reward dataset
overrides via ``reward_dataset_map`` and is distributed-safe with padded
symmetric batching.

Supported backbones (``model_type``): ``sd35``, ``flux``, ``flux2_klein``.

Usage:
    config = InferenceConfig.from_yaml("configs/inference/sd35m.yaml")
    engine = RTDMDInference(config)
    engine.setup()
    engine.run()
"""

from __future__ import annotations

import copy
import json
import logging
import os
import random
import re
import time
from contextlib import nullcontext
from dataclasses import dataclass, field, fields, asdict
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from diffusers import (
    Flux2Transformer2DModel,
    FluxTransformer2DModel,
    SD3Transformer2DModel,
)
from PIL import Image
try:
    from safetensors.torch import load_file as safetensors_load_file
except Exception:
    safetensors_load_file = None

from rtdmd.models.flux import (
    compute_mu_flux,
    compute_sigmas_flux,
    encode_prompts_flux,
    load_flux_models,
    predict_noise_flux,
)
from rtdmd.models.flux2_klein import (
    compute_empirical_mu,
    compute_sigmas_flux2_klein,
    encode_prompts_flux2_klein,
    load_flux2_klein_models,
    predict_noise_flux2_klein,
    unpatchify_flux2_latents,
)
from rtdmd.models.sd35 import (
    compute_sigmas_sd35,
    encode_prompts_sd35,
    load_sd35_models,
    predict_noise_sd35,
    _get_torch_dtype,
)
from rtdmd.schedulers import CPSScheduler
from rtdmd.utils.image import decode_latents_to_tensor
from rtdmd.utils.checkpoint import is_peft_lora_dir, _is_lora_checkpoint
from rtdmd.utils.fast_init import fast_init
from rtdmd.parallel.utils import (
    setup_distributed,
    cleanup_distributed,
    get_rank,
    get_world_size,
    is_main_process,
    barrier,
    fsdp_wrap_model,
    ddp_wrap_model,
    get_transformer_wrap_policy,
)

logger = logging.getLogger(__name__)

SUPPORTED_MODEL_TYPES = {"sd35", "flux", "flux2_klein"}


# ---------------------------------------------------------------------------
# LoRA helpers
# ---------------------------------------------------------------------------

def _strip_wrapper_prefixes(name: str) -> str:
    """Remove common distributed/wrapper prefixes from state_dict keys."""
    out = name
    while True:
        stripped = False
        for prefix in ("module.", "_orig_mod.", "base_model.model."):
            if out.startswith(prefix):
                out = out[len(prefix):]
                stripped = True
        if not stripped:
            break
    return out


def _resolve_lora_path(path: str) -> str:
    """Resolve a LoRA input to a concrete PEFT dir or .pt/.pth file."""
    resolved = os.path.expanduser(str(path).strip())
    if not resolved:
        raise ValueError("LoRA path is empty.")
    if is_peft_lora_dir(resolved):
        return resolved
    nested = os.path.join(resolved, "transformer")
    if is_peft_lora_dir(nested):
        return nested
    if os.path.isfile(resolved):
        return resolved
    raise FileNotFoundError(
        f"Cannot resolve LoRA path: {path}. Expected a PEFT dir, "
        f"a checkpoint dir containing transformer/, or a .pt/.pth file."
    )


def _load_lora_state_dict(resolved_path: str) -> dict[str, torch.Tensor]:
    """Load LoRA weights from a PEFT directory or LoRA-only .pt/.pth file."""
    if os.path.isdir(resolved_path):
        safetensors_path = os.path.join(resolved_path, "adapter_model.safetensors")
        pt_path = os.path.join(resolved_path, "adapter_model.pt")
        if os.path.exists(safetensors_path):
            if safetensors_load_file is None:
                raise ImportError(
                    "safetensors is required to load adapter_model.safetensors "
                    f"from {resolved_path}"
                )
            return safetensors_load_file(safetensors_path, device="cpu")
        if os.path.exists(pt_path):
            return torch.load(pt_path, map_location="cpu", weights_only=False)
        raise FileNotFoundError(
            f"No adapter_model.safetensors or adapter_model.pt found in {resolved_path}"
        )

    state_dict = torch.load(resolved_path, map_location="cpu", weights_only=False)
    if not _is_lora_checkpoint(state_dict):
        raise ValueError(
            f"File does not look like a LoRA checkpoint: {resolved_path}."
        )
    return state_dict


def _read_lora_merge_config(
    resolved_path: str,
    fallback_rank: int,
    fallback_alpha: int,
) -> dict[str, Any]:
    """Read rank/alpha from adapter_config.json when available, else fall back."""
    if os.path.isdir(resolved_path):
        adapter_cfg_path = os.path.join(resolved_path, "adapter_config.json")
        if os.path.exists(adapter_cfg_path):
            with open(adapter_cfg_path, "r", encoding="utf-8") as f:
                adapter_cfg = json.load(f)
            return {
                "rank": int(adapter_cfg.get("r", fallback_rank)),
                "lora_alpha": int(adapter_cfg.get("lora_alpha", fallback_alpha)),
            }
    return {"rank": int(fallback_rank), "lora_alpha": int(fallback_alpha)}


def _merge_lora_into_base_transformer(
    model: nn.Module,
    resolved_lora_path: str,
    fallback_rank: int,
    fallback_alpha: int,
) -> dict[str, float]:
    """Merge one LoRA checkpoint into base transformer weights in-place."""
    cfg = _read_lora_merge_config(
        resolved_path=resolved_lora_path,
        fallback_rank=fallback_rank,
        fallback_alpha=fallback_alpha,
    )
    lora_state = _load_lora_state_dict(resolved_lora_path)

    rank = int(cfg["rank"])
    if rank <= 0:
        raise ValueError(f"Invalid LoRA rank={rank} for {resolved_lora_path}")
    scaling = float(cfg["lora_alpha"]) / float(rank)

    pairs: dict[str, dict[str, torch.Tensor]] = {}
    for key, weight in lora_state.items():
        norm_key = _strip_wrapper_prefixes(key)
        if ".lora_A" in norm_key:
            base_key = re.sub(r"\.lora_A(?:\.[^.]+)?\.weight$", ".weight", norm_key)
            pairs.setdefault(base_key, {})["A"] = weight
        elif ".lora_B" in norm_key:
            base_key = re.sub(r"\.lora_B(?:\.[^.]+)?\.weight$", ".weight", norm_key)
            pairs.setdefault(base_key, {})["B"] = weight

    named_params = dict(model.named_parameters())
    merged = 0
    skipped = 0
    for base_key, ab in pairs.items():
        if "A" not in ab or "B" not in ab:
            skipped += 1
            continue
        if base_key not in named_params:
            skipped += 1
            continue
        param = named_params[base_key]
        A = ab["A"].to(device=param.device, dtype=torch.float32)
        B = ab["B"].to(device=param.device, dtype=torch.float32)
        delta = B @ A
        if delta.shape != param.shape:
            if delta.t().shape == param.shape:
                delta = delta.t()
            else:
                skipped += 1
                continue
        param.data.add_((scaling * delta).to(dtype=param.dtype))
        merged += 1

    if merged == 0:
        raise RuntimeError(
            f"No transformer parameters were merged from LoRA: {resolved_lora_path}"
        )
    return {
        "merged_params": float(merged),
        "skipped_pairs": float(skipped),
        "scaling": float(scaling),
    }


# ---------------------------------------------------------------------------
# YAML / dataclass helpers
# ---------------------------------------------------------------------------

def _apply_flat_overrides(d: dict, overrides: dict[str, Any]) -> dict:
    for key, value in overrides.items():
        parts = key.split(".")
        target = d
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = value
    return d


def _dataclass_from_dict(cls, data: Any):
    if not isinstance(data, dict):
        return data
    field_names = {f.name for f in fields(cls)}
    kwargs = {name: data[name] for name in field_names if name in data}
    return cls(**kwargs)


# ---------------------------------------------------------------------------
# InferenceConfig
# ---------------------------------------------------------------------------

@dataclass
class InferenceConfig:
    """Configuration for RTDMD inference."""

    # Backbone
    pretrained_path: str = ""
    model_type: str = "sd35"
    dtype: str = "bf16"
    image_resolution: int = 512

    # distilled (CPS few-step) or normal (multi-step Flow-Matching + CFG)
    mode: str = "distilled"

    # LoRA stack -- 0/1/2 entries selects the three supported regimes:
    #   []                       -> plain inference (no LoRA)
    #   [distilled]              -> distilled-only LoRA
    #   [distilled, rl]          -> distilled + RL LoRAs (merged in order)
    # Each entry can be a PEFT directory, a checkpoint dir containing
    # transformer/, or a flat LoRA .pt/.pth file.
    lora_paths: list = field(default_factory=list)
    # Fallback rank/alpha for flat .pt LoRAs without adapter_config.json.
    # Matches the training-time defaults.
    lora_rank: int = 32
    lora_alpha: int = 64

    # Distilled-mode parameters
    num_steps: int = 4
    denoising_step_list: list = field(default_factory=lambda: [1000, 750, 500, 250])
    cps_eta: float = 0.9

    # Normal-mode parameters
    guidance_scale: float = 1.0

    # Sampling
    seed: int = 42
    batch_size: int = 4

    # Prompt sources
    prompts: list = field(default_factory=list)
    prompt_file: str = ""
    eval_prompt_source: str = "dataset"   # prompt_file | dataset
    eval_prompt_path: str = ""
    dataset: str = "drawbench"
    dataset_path: str = "dataset"

    # Output
    output_dir: str = "./inference_outputs"

    # Reward eval (mirrors training EvalConfig)
    eval_reward: bool = False
    reward_fn: dict = field(default_factory=dict)
    reward_dataset_map: dict = field(default_factory=dict)
    reward_ckpt_path: str = ""
    preload_reward_models: bool = True
    # 0 = no cap. When > 0, evaluate only the first N prompts of the dataset
    # (matches the training-time semantics of `eval.num_media_images`).
    num_media_images: int = 0

    # Distributed
    distributed: bool = False
    strategy: str = "fsdp"

    @classmethod
    def from_yaml(cls, path: str, overrides: dict[str, Any] | None = None) -> InferenceConfig:
        import yaml
        with open(path, "r") as f:
            raw = yaml.safe_load(f) or {}
        if overrides:
            raw = _apply_flat_overrides(raw, overrides)
        return _dataclass_from_dict(cls, raw)

    def validate(self) -> None:
        self.model_type = str(self.model_type).lower()
        if self.model_type not in SUPPORTED_MODEL_TYPES:
            raise ValueError(
                f"Unsupported model_type: {self.model_type}. "
                f"Supported: {sorted(SUPPORTED_MODEL_TYPES)}"
            )

        # Normalize lora_paths into a list[str].
        if isinstance(self.lora_paths, str):
            raw = self.lora_paths.strip()
            if not raw:
                self.lora_paths = []
            elif raw.startswith("["):
                parsed = json.loads(raw)
                if not isinstance(parsed, list):
                    raise TypeError("lora_paths JSON literal must decode to a list")
                self.lora_paths = [str(x).strip() for x in parsed if str(x).strip()]
            else:
                self.lora_paths = [x.strip() for x in raw.split(",") if x.strip()]
        elif isinstance(self.lora_paths, (tuple, set, list)):
            self.lora_paths = [str(x).strip() for x in self.lora_paths if str(x).strip()]
        else:
            raise TypeError(
                f"lora_paths must be list/tuple/set/str, got {type(self.lora_paths)}"
            )

        if self.mode not in ("distilled", "normal"):
            raise ValueError(f"Unknown mode: {self.mode!r}. Expected 'distilled' or 'normal'.")

        if self.eval_prompt_source not in ("prompt_file", "dataset"):
            raise ValueError(
                f"eval_prompt_source must be 'prompt_file' or 'dataset', got: {self.eval_prompt_source!r}"
            )

        if not isinstance(self.reward_dataset_map, dict):
            raise TypeError(
                f"reward_dataset_map must be a dict, got {type(self.reward_dataset_map)}"
            )

        if self.mode == "distilled":
            actual = min(self.num_steps, len(self.denoising_step_list))
            if actual != self.num_steps:
                logger.warning(
                    f"distilled: num_steps={self.num_steps} > len(denoising_step_list)="
                    f"{len(self.denoising_step_list)}, using {actual} steps"
                )

        # At least one prompt source must be configured.
        has_inline = bool(self.prompts)
        has_file = bool(self.prompt_file)
        has_eval_path = bool(self.eval_prompt_path)
        use_default_dataset = self.eval_prompt_source == "dataset"
        use_reward_datasets = self.eval_reward and bool(self.reward_dataset_map)
        if not (has_inline or has_file or has_eval_path or use_default_dataset or use_reward_datasets):
            raise ValueError(
                "No prompt source specified. Set prompts/prompt_file/eval_prompt_path, "
                "use eval_prompt_source='dataset', or configure reward_dataset_map for eval."
            )

        if self.eval_prompt_source == "dataset":
            if not str(self.dataset).strip():
                raise ValueError("dataset must be non-empty when eval_prompt_source='dataset'")
            if not str(self.dataset_path).strip():
                raise ValueError("dataset_path must be non-empty when eval_prompt_source='dataset'")

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# RTDMDInference
# ---------------------------------------------------------------------------

class RTDMDInference:
    """Unified inference engine for RTDMD distilled and normal generation."""

    def __init__(self, config: InferenceConfig):
        self.config = config
        self.model_type = str(config.model_type).lower()
        self._is_flux2_klein = self.model_type == "flux2_klein"
        self._is_flux = self.model_type == "flux"
        self.transformer: nn.Module | None = None
        self.vae = None
        self.text_encoders: list = []
        self.tokenizers: list = []
        self.scheduler = None
        self.x0_scheduler: CPSScheduler | None = None
        self._sigmas: list[float] = []
        self._uncond_embeds: tuple[torch.Tensor, torch.Tensor] | None = None
        self._scorer = None
        self._single_reward_scorers: dict[str, Any] = {}
        self._eval_prompts_cache: dict[str, list[str]] = {}
        self._eval_metadata_cache: dict[str, list[dict[str, Any]] | None] = {}

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(self) -> RTDMDInference:
        """Load models, apply LoRA(s), and set up the scheduler."""
        cfg = self.config
        self.model_type = str(cfg.model_type).lower()
        self._is_flux2_klein = self.model_type == "flux2_klein"
        self._is_flux = self.model_type == "flux"

        if cfg.distributed:
            setup_distributed()

        if is_main_process():
            logger.info(f"Setting up inference (model_type={self.model_type}, mode={cfg.mode})")

        # Load all backbone components.
        from rtdmd.config import ModelConfig
        model_cfg = ModelConfig(
            pretrained_path=cfg.pretrained_path,
            model_type=cfg.model_type,
            dtype=cfg.dtype,
            image_resolution=cfg.image_resolution,
        )
        if self._is_flux2_klein:
            components = load_flux2_klein_models(model_cfg)
        elif self._is_flux:
            components = load_flux_models(model_cfg)
        else:
            components = load_sd35_models(model_cfg)
        self.text_encoders = components["text_encoders"]
        self.tokenizers = components["tokenizers"]
        self.scheduler = components["scheduler"]
        self.vae = components["vae"]
        self.vae.requires_grad_(False)
        self.vae.eval()

        self.transformer = self._load_transformer(components["transformer"])
        self.transformer.requires_grad_(False)
        self.transformer.eval()

        # Derive latent geometry.
        self.latent_channels = self.transformer.config.in_channels
        vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        latent_divisor = vae_scale_factor * 2 if self._is_flux2_klein else vae_scale_factor
        if cfg.image_resolution % latent_divisor != 0:
            raise ValueError(
                f"image_resolution={cfg.image_resolution} must be divisible by "
                f"{latent_divisor} for model_type={self.model_type}"
            )
        self.latent_size = cfg.image_resolution // latent_divisor
        if self._is_flux and self.latent_size % 2 != 0:
            raise ValueError(
                f"FLUX latent size must be even (got {self.latent_size}); pick an "
                f"image_resolution that is a multiple of {latent_divisor * 2}."
            )

        # Distilled-mode CPS scheduler + sigma schedule.
        if cfg.mode == "distilled":
            self.x0_scheduler = CPSScheduler.from_config(self.scheduler.config)
            if hasattr(self.x0_scheduler, "set_eta"):
                self.x0_scheduler.set_eta(cfg.cps_eta)
            self._sigmas = self._compute_distilled_sigmas()
            if is_main_process():
                logger.info(
                    f"  CPS scheduler ready (cps_eta={cfg.cps_eta}); "
                    f"denoising_step_list={cfg.denoising_step_list}; sigmas={self._sigmas}"
                )

        for enc in self.text_encoders:
            enc.requires_grad_(False)
            enc.eval()

        if cfg.eval_reward and not bool(cfg.reward_dataset_map):
            self._init_scorer()

        device = torch.device(
            "cuda", torch.cuda.current_device() if cfg.distributed else 0,
        )
        self.vae.to(device)
        for enc in self.text_encoders:
            enc.to(device)

        if cfg.distributed:
            wrap_policy = self._build_transformer_wrap_policy()
            if cfg.strategy == "fsdp":
                self.transformer = fsdp_wrap_model(
                    self.transformer,
                    sharding_strategy="full_shard",
                    fsdp_precision=cfg.dtype,
                    auto_wrap_policy=wrap_policy,
                )
            else:
                self.transformer.to(device)
                self.transformer = ddp_wrap_model(self.transformer)
        else:
            self.transformer.to(device)

        if is_main_process():
            logger.info(
                f"  Latent shape: [{self.latent_channels}, {self.latent_size}, {self.latent_size}] "
                f"(image_res={cfg.image_resolution}, latent_divisor={latent_divisor})"
            )
            logger.info("Setup complete")
        return self

    def _build_transformer_wrap_policy(self):
        """FSDP wrap policy keyed on the model type."""
        if self._is_flux2_klein:
            from diffusers.models.transformers.transformer_flux2 import (
                Flux2SingleTransformerBlock,
                Flux2TransformerBlock,
            )
            return get_transformer_wrap_policy(
                {Flux2TransformerBlock, Flux2SingleTransformerBlock}
            )
        if self._is_flux:
            from diffusers.models.transformers.transformer_flux import (
                FluxSingleTransformerBlock,
                FluxTransformerBlock,
            )
            return get_transformer_wrap_policy(
                {FluxTransformerBlock, FluxSingleTransformerBlock}
            )
        from diffusers.models.transformers.transformer_sd3 import JointTransformerBlock
        return get_transformer_wrap_policy(JointTransformerBlock)

    def _compute_distilled_sigmas(self) -> list[float]:
        """Compute the (possibly shifted) sigma schedule used by distilled mode."""
        cfg = self.config
        scheduler_cfg = self.scheduler.config
        if self._is_flux2_klein:
            return compute_sigmas_flux2_klein(
                denoising_step_list=cfg.denoising_step_list,
                scheduler_config=scheduler_cfg,
                image_seq_len=self.latent_size * self.latent_size,
            )
        if self._is_flux:
            packed_tokens = (self.latent_size // 2) * (self.latent_size // 2)
            return compute_sigmas_flux(
                denoising_step_list=cfg.denoising_step_list,
                scheduler_config=scheduler_cfg,
                image_seq_len=packed_tokens,
            )
        return compute_sigmas_sd35(
            cfg.denoising_step_list,
            num_train_timesteps=scheduler_cfg.num_train_timesteps,
            shift=scheduler_cfg.shift,
        )

    def _load_transformer(self, base_transformer: nn.Module) -> nn.Module:
        """Merge zero, one, or two LoRA adapters into the base transformer."""
        cfg = self.config
        lora_paths = [str(p).strip() for p in cfg.lora_paths if str(p).strip()]
        if not lora_paths:
            return base_transformer

        for i, raw_path in enumerate(lora_paths, start=1):
            resolved_path = _resolve_lora_path(raw_path)
            stats = _merge_lora_into_base_transformer(
                model=base_transformer,
                resolved_lora_path=resolved_path,
                fallback_rank=cfg.lora_rank,
                fallback_alpha=cfg.lora_alpha,
            )
            if is_main_process():
                logger.info(
                    f"[LoRA {i}/{len(lora_paths)}] merged from {resolved_path} "
                    f"(merged={int(stats['merged_params'])}, "
                    f"skipped={int(stats['skipped_pairs'])}, "
                    f"scale={stats['scaling']:.4f})"
                )
        return base_transformer

    # ------------------------------------------------------------------
    # Reward scorer wiring
    # ------------------------------------------------------------------

    def _init_scorer(self) -> None:
        cfg = self.config
        self._configure_reward_ckpt_path()
        from rtdmd.rewards.multi_scorer import MultiScorer
        self._scorer = MultiScorer(
            device=torch.device("cpu"),
            score_dict=cfg.reward_fn,
            allow_unavailable=True,
        )
        if is_main_process():
            active = getattr(self._scorer, "active_reward_names", list(cfg.reward_fn.keys()))
            logger.info(f"Initialized scorer: active={active}")
            unavailable = getattr(self._scorer, "unavailable_rewards", {})
            if unavailable:
                logger.warning(f"Skipped unavailable rewards: {unavailable}")

    def _configure_reward_ckpt_path(self) -> None:
        if self.config.reward_ckpt_path:
            from rtdmd.rewards.reward_ckpt_path import set_ckpt_path
            set_ckpt_path(self.config.reward_ckpt_path)

    def _get_single_reward_scorer(self, reward_name: str):
        scorer = self._single_reward_scorers.get(reward_name)
        if scorer is not None:
            return scorer

        if reward_name not in self.config.reward_fn:
            raise KeyError(
                f"Reward {reward_name!r} not found in reward_fn keys="
                f"{list(self.config.reward_fn.keys())}"
            )

        self._configure_reward_ckpt_path()
        from rtdmd.rewards.multi_scorer import MultiScorer
        scorer = MultiScorer(
            device=torch.device("cpu"),
            score_dict={reward_name: self.config.reward_fn[reward_name]},
            allow_unavailable=True,
        )
        self._single_reward_scorers[reward_name] = scorer

        if is_main_process():
            active = getattr(scorer, "active_reward_names", [])
            if active:
                logger.info(f"Initialized single reward scorer: {reward_name}")
            else:
                unavailable = getattr(scorer, "unavailable_rewards", {})
                logger.warning(
                    f"Skipped unavailable single reward scorer {reward_name!r}: {unavailable}"
                )
        return scorer

    # ------------------------------------------------------------------
    # Prompt loading / reward eval helpers (training-aligned)
    # ------------------------------------------------------------------

    @staticmethod
    def _read_prompt_file_with_metadata(
        path: str,
    ) -> tuple[list[str], list[dict[str, Any]] | None]:
        """Read prompts (+ optional metadata) from a .json / .jsonl / .txt file."""
        ext = os.path.splitext(path)[1].lower()
        if ext == ".json":
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                if data and isinstance(data[0], str):
                    return data, None
                if data and isinstance(data[0], dict):
                    prompts = [item.get("prompt", item.get("text", "")) for item in data]
                    return prompts, data
            raise ValueError(f"Cannot parse prompts from JSON: {path}")
        if ext == ".jsonl":
            prompts: list[str] = []
            metadata: list[dict[str, Any]] = []
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    item = json.loads(line)
                    if isinstance(item, dict):
                        prompts.append(item.get("prompt", item.get("text", "")))
                        metadata.append(item)
                    else:
                        prompts.append(str(item))
                        metadata.append({"text": str(item)})
            return prompts, metadata
        with open(path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()], None

    def _load_eval_prompts_and_metadata(
        self,
        dataset_override: str | None = None,
    ) -> tuple[list[str], list[dict[str, Any]] | None]:
        """Load prompts with optional dataset override (matches training eval)."""
        cfg = self.config
        use_dataset = dataset_override is not None or cfg.eval_prompt_source == "dataset"
        if use_dataset:
            dataset_name = dataset_override if dataset_override is not None else cfg.dataset
            dataset_dir = os.path.join(cfg.dataset_path, dataset_name)
            for fname in (
                "test.txt",
                "test.jsonl",
                "test_metadata.jsonl",
                "prompts.json",
                "prompts.txt",
                "metadata.jsonl",
            ):
                fpath = os.path.join(dataset_dir, fname)
                if os.path.exists(fpath):
                    prompts, metadata = self._read_prompt_file_with_metadata(fpath)
                    return self._maybe_truncate(prompts, metadata)
            raise FileNotFoundError(
                f"No prompt file found in dataset directory: {dataset_dir}"
            )

        if cfg.prompts:
            return self._maybe_truncate(list(cfg.prompts), None)

        prompt_path = cfg.eval_prompt_path or cfg.prompt_file
        if not prompt_path:
            raise ValueError(
                "No prompt file configured for eval_prompt_source='prompt_file'. "
                "Set eval_prompt_path or prompt_file."
            )
        if not os.path.exists(prompt_path):
            raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
        prompts, metadata = self._read_prompt_file_with_metadata(prompt_path)
        return self._maybe_truncate(prompts, metadata)

    def _maybe_truncate(
        self,
        prompts: list[str],
        metadata: list[dict[str, Any]] | None,
    ) -> tuple[list[str], list[dict[str, Any]] | None]:
        """Cap the prompt list to ``num_media_images`` when set."""
        cap = int(self.config.num_media_images or 0)
        if cap <= 0 or len(prompts) <= cap:
            return prompts, metadata
        truncated_meta = metadata[:cap] if metadata is not None else None
        return prompts[:cap], truncated_meta

    @staticmethod
    def _cache_key_for_dataset(dataset_override: str | None = None) -> str:
        return "__default__" if dataset_override is None else f"dataset::{dataset_override}"

    def _get_cached_eval_prompts_and_metadata(
        self,
        dataset_override: str | None = None,
    ) -> tuple[list[str], list[dict[str, Any]] | None]:
        cache_key = self._cache_key_for_dataset(dataset_override)
        if cache_key not in self._eval_prompts_cache:
            prompts, metadata = self._load_eval_prompts_and_metadata(dataset_override=dataset_override)
            self._eval_prompts_cache[cache_key] = prompts
            self._eval_metadata_cache[cache_key] = metadata
            source = (
                f"dataset={dataset_override}"
                if dataset_override is not None
                else self.config.eval_prompt_source
            )
            logger.info(f"[Inference Eval] Loaded {len(prompts)} prompts ({source})")
        return self._eval_prompts_cache[cache_key], self._eval_metadata_cache.get(cache_key)

    def _resolve_eval_dataset_name(self, dataset_override: str | None = None) -> str:
        if dataset_override is not None:
            return dataset_override
        if self.config.eval_prompt_source == "dataset":
            return str(self.config.dataset)
        return "prompt_file"

    def _warmup_reward_eval_resources(self) -> None:
        """Preload prompts/scorers before seeding RNG for deterministic eval."""
        cfg = self.config
        if not cfg.eval_reward:
            return
        reward_dataset_map = cfg.reward_dataset_map or {}
        if not isinstance(reward_dataset_map, dict):
            raise TypeError(
                f"reward_dataset_map must be a dict, got {type(reward_dataset_map)}"
            )
        if not reward_dataset_map:
            self._get_cached_eval_prompts_and_metadata(dataset_override=None)
            if self._scorer is None:
                self._init_scorer()
            return
        for reward_name in cfg.reward_fn.keys():
            ds_raw = reward_dataset_map.get(reward_name, None)
            dataset_override = (
                str(ds_raw).strip() if isinstance(ds_raw, str) and str(ds_raw).strip() else None
            )
            self._get_cached_eval_prompts_and_metadata(dataset_override=dataset_override)
            self._get_single_reward_scorer(reward_name)

    @staticmethod
    def _score_details_to_mean_metrics(score_details: dict[str, Any]) -> dict[str, float]:
        metrics: dict[str, float] = {}
        for metric_name, scores in score_details.items():
            if isinstance(scores, torch.Tensor):
                scores = scores.detach().cpu().numpy()
            elif isinstance(scores, list):
                scores = np.array([float(s) for s in scores])
            else:
                scores = np.array(scores)
            valid = scores[scores != -10]
            if len(valid) > 0:
                metrics[metric_name] = float(np.mean(valid))
        return metrics

    @staticmethod
    def _aggregate_eval_metrics_across_ranks(
        eval_metrics: dict[str, float],
        num_local: int,
    ) -> dict[str, float]:
        world_size = get_world_size()
        if not dist.is_initialized() or world_size <= 1:
            return eval_metrics

        device = torch.device("cuda", torch.cuda.current_device())
        local_keys = list(eval_metrics.keys())
        gathered_keys: list[list[str] | None] = [None] * world_size
        dist.all_gather_object(gathered_keys, local_keys)
        all_keys = sorted({k for keys in gathered_keys if keys is not None for k in keys})

        aggregated: dict[str, float] = {}
        for key in all_keys:
            has_local = key in eval_metrics
            local_num = torch.tensor(
                float(eval_metrics[key] * num_local) if has_local else 0.0,
                device=device,
            )
            local_den = torch.tensor(
                float(num_local) if has_local else 0.0,
                device=device,
            )
            dist.all_reduce(local_num, op=dist.ReduceOp.SUM)
            dist.all_reduce(local_den, op=dist.ReduceOp.SUM)
            if local_den.item() > 0:
                aggregated[key] = local_num.item() / local_den.item()
        return aggregated

    @staticmethod
    def _aggregate_num_samples_across_ranks(num_local: int) -> int:
        if not dist.is_initialized() or get_world_size() <= 1:
            return int(num_local)
        device = torch.device("cuda", torch.cuda.current_device())
        total = torch.tensor(float(num_local), device=device)
        dist.all_reduce(total, op=dist.ReduceOp.SUM)
        return int(total.item())

    @staticmethod
    def _aggregate_scalar_sum_across_ranks(value: float) -> float:
        if not dist.is_initialized() or get_world_size() <= 1:
            return float(value)
        device = torch.device("cuda", torch.cuda.current_device())
        total = torch.tensor(float(value), device=device)
        dist.all_reduce(total, op=dist.ReduceOp.SUM)
        return float(total.item())

    @staticmethod
    def _sanitize_output_token(name: str) -> str:
        token = (name or "").strip()
        if not token:
            return "default"
        token = token.replace("/", "_").replace(" ", "_")
        token = "".join(ch if (ch.isalnum() or ch in ("-", "_", ".")) else "_" for ch in token)
        return token or "default"

    # ------------------------------------------------------------------
    # Text encoding / forward helpers
    # ------------------------------------------------------------------

    def _encode_prompts(self, prompts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        device = torch.device("cuda")
        if self._is_flux2_klein:
            return encode_prompts_flux2_klein(
                prompts=prompts,
                text_encoder=self.text_encoders[0],
                tokenizer=self.tokenizers[0],
                device=device,
            )
        if self._is_flux:
            return encode_prompts_flux(prompts, self.text_encoders, self.tokenizers, device)
        return encode_prompts_sd35(prompts, self.text_encoders, self.tokenizers, device)

    def _get_uncond_embeds(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self._uncond_embeds is None:
            self._uncond_embeds = self._encode_prompts([""])
        uncond_embeds, uncond_pooled = self._uncond_embeds
        if self._is_flux2_klein:
            return (
                uncond_embeds.expand(batch_size, -1, -1),
                uncond_pooled.expand(batch_size, -1, -1),
            )
        return (
            uncond_embeds.expand(batch_size, -1, -1),
            uncond_pooled.expand(batch_size, -1),
        )

    def _num_train_timesteps(self) -> int:
        if self.scheduler is not None and hasattr(self.scheduler, "config"):
            return int(getattr(self.scheduler.config, "num_train_timesteps", 1000))
        return 1000

    def _predict_noise(
        self,
        noisy_latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
        timesteps: torch.Tensor,
        pooled_embeds: torch.Tensor,
        guidance_scale: float = 1.0,
        uncond_text_embeddings: torch.Tensor | None = None,
        uncond_pooled_prompt_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self._is_flux2_klein:
            return predict_noise_flux2_klein(
                model=self.transformer,
                noisy_latents=noisy_latents,
                prompt_embeds=prompt_embeds,
                timesteps=timesteps,
                text_ids=pooled_embeds,
                guidance_scale=guidance_scale,
                uncond_prompt_embeds=uncond_text_embeddings,
                uncond_text_ids=uncond_pooled_prompt_embeds,
                num_train_timesteps=self._num_train_timesteps(),
            )
        if self._is_flux:
            return predict_noise_flux(
                model=self.transformer,
                noisy_latents=noisy_latents,
                prompt_embeds=prompt_embeds,
                timesteps=timesteps,
                pooled_embeds=pooled_embeds,
                guidance_scale=guidance_scale,
                uncond_text_embeddings=uncond_text_embeddings,
                uncond_pooled_prompt_embeds=uncond_pooled_prompt_embeds,
                num_train_timesteps=self._num_train_timesteps(),
            )
        return predict_noise_sd35(
            self.transformer,
            noisy_latents,
            prompt_embeds,
            timesteps,
            pooled_embeds,
            guidance_scale=guidance_scale,
            uncond_text_embeddings=uncond_text_embeddings,
            uncond_pooled_prompt_embeds=uncond_pooled_prompt_embeds,
        )

    @torch.no_grad()
    def _decode_latents_to_tensor(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latent tensors to a [0, 1] image tensor."""
        if not self._is_flux2_klein:
            return decode_latents_to_tensor(self.vae, latents)

        device = latents.device
        if next(self.vae.parameters()).device != device:
            self.vae.to(device)
        bn_mean = self.vae.bn.running_mean.view(1, -1, 1, 1).to(device, latents.dtype)
        bn_std = torch.sqrt(
            self.vae.bn.running_var.view(1, -1, 1, 1) + self.vae.config.batch_norm_eps
        ).to(device, latents.dtype)
        latents = latents * bn_std + bn_mean
        latents = unpatchify_flux2_latents(latents)
        images = self.vae.decode(latents.to(self.vae.dtype)).sample
        return (images / 2 + 0.5).clamp(0, 1)

    def _autocast(self):
        dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16}
        dtype = dtype_map.get(self.config.dtype)
        return torch.autocast("cuda", dtype=dtype) if dtype is not None else nullcontext()

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    @staticmethod
    def _tensor_to_pil(images: torch.Tensor) -> list[Image.Image]:
        """Convert an NCHW float tensor in [0, 1] to a list of PIL images."""
        images_uint8 = (
            (images * 255)
            .round()
            .clamp(0, 255)
            .to(torch.uint8)
            .cpu()
            .permute(0, 2, 3, 1)
            .numpy()
        )
        return [Image.fromarray(img) for img in images_uint8]

    @torch.no_grad()
    def _generate_batch(
        self,
        prompts: list[str],
    ) -> tuple[list[Image.Image], torch.Tensor]:
        """Generate one batch and return PIL images + raw float tensors for scoring."""
        cfg = self.config
        prompt_embeds, pooled_embeds = self._encode_prompts(prompts)
        if cfg.mode == "distilled":
            latents = self._generate_distilled(prompt_embeds, pooled_embeds)
        else:
            uncond_embeds, uncond_pooled = self._get_uncond_embeds(len(prompts))
            latents = self._generate_normal(
                prompt_embeds, pooled_embeds, uncond_embeds, uncond_pooled,
            )
        image_tensors = self._decode_latents_to_tensor(latents)
        return self._tensor_to_pil(image_tensors), image_tensors

    @torch.no_grad()
    def _generate_images_for_prompts(
        self,
        all_prompts: list[str],
        all_metadata: list[dict[str, Any]] | None,
        *,
        save_prefix: str = "",
    ) -> tuple[torch.Tensor | None, list[str], list[dict[str, Any]] | None, int, float]:
        """Distributed-safe symmetric-batched generation. Saves PNGs to disk."""
        cfg = self.config
        rank = get_rank()
        world_size = get_world_size() if cfg.distributed else 1

        if not all_prompts:
            return None, [], None, 0, 0.0

        rank_prompts = all_prompts[rank::world_size]
        rank_metadata = all_metadata[rank::world_size] if isinstance(all_metadata, list) else None

        max_prompts_per_rank = (len(all_prompts) + world_size - 1) // world_size
        max_batches = (max_prompts_per_rank + cfg.batch_size - 1) // cfg.batch_size

        local_images: list[torch.Tensor] = []
        local_prompts: list[str] = []
        local_metadata: list[dict[str, Any]] = []
        total_time = 0.0

        for batch_idx in range(max_batches):
            start = batch_idx * cfg.batch_size
            end = min(start + cfg.batch_size, len(rank_prompts))

            if start < len(rank_prompts):
                batch_prompts = rank_prompts[start:end]
                if len(batch_prompts) < cfg.batch_size:
                    batch_prompts = batch_prompts + [batch_prompts[-1]] * (
                        cfg.batch_size - len(batch_prompts)
                    )
                actual_count = end - start
            else:
                fallback_prompt = rank_prompts[0] if rank_prompts else all_prompts[0]
                batch_prompts = [fallback_prompt] * cfg.batch_size
                actual_count = 0

            t0 = time.time()
            batch_images, batch_tensors = self._generate_batch(batch_prompts)
            total_time += time.time() - t0

            if actual_count <= 0:
                continue

            batch_images = batch_images[:actual_count]
            batch_tensors = batch_tensors[:actual_count]
            local_images.append(batch_tensors)
            local_prompts.extend(rank_prompts[start:end])
            if rank_metadata is not None:
                local_metadata.extend(rank_metadata[start:end])

            for i, img in enumerate(batch_images):
                local_idx = start + i
                global_idx = rank + local_idx * world_size
                fname = f"{save_prefix}{global_idx:06d}.png"
                img.save(os.path.join(cfg.output_dir, fname))

        if not local_images:
            metadata_out = [] if rank_metadata is not None else None
            return None, [], metadata_out, 0, total_time

        images_tensor = torch.cat(local_images, dim=0)
        metadata_out = local_metadata if rank_metadata is not None else None
        return images_tensor, local_prompts, metadata_out, len(local_prompts), total_time

    def _generate_distilled(
        self,
        prompt_embeds: torch.Tensor,
        pooled_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """Few-step distilled generation (mirror of training-time eval)."""
        cfg = self.config
        batch_size = prompt_embeds.shape[0]
        device = prompt_embeds.device
        dtype = prompt_embeds.dtype

        sigmas = self._sigmas
        num_steps = min(cfg.num_steps, len(sigmas))

        # Reset scheduler (identical to training).
        self.x0_scheduler.set_timesteps(sigmas=sigmas)
        self.x0_scheduler._step_index = 0

        # x_t and the CPS scheduler's internal noise must draw from different
        # RNG positions to avoid double-counting at sigma=1 (see the long note
        # in dmd_trainer._generate_latents).
        x_t = torch.randn(
            batch_size, self.latent_channels, self.latent_size, self.latent_size,
            device=device, dtype=dtype,
        )
        x0 = x_t

        for idx in range(num_steps):
            t = self.x0_scheduler.timesteps[idx]
            t_batch = t.expand(batch_size).to(device=device, dtype=torch.float32)
            with self._autocast():
                v_pred = self._predict_noise(
                    noisy_latents=x_t,
                    prompt_embeds=prompt_embeds,
                    timesteps=t_batch,
                    pooled_embeds=pooled_embeds,
                    guidance_scale=1.0,
                )
            step_output = self.x0_scheduler.step(v_pred, t, x_t, return_dict=True)
            sigma = self.x0_scheduler.sigmas[idx]
            x0 = (x_t.float() - sigma * v_pred.float()).to(dtype)
            x_t = step_output.prev_sample

        return x0

    def _generate_normal(
        self,
        prompt_embeds: torch.Tensor,
        pooled_embeds: torch.Tensor,
        uncond_embeds: torch.Tensor,
        uncond_pooled: torch.Tensor,
    ) -> torch.Tensor:
        """Standard multi-step inference (FlowMatchEulerDiscreteScheduler + CFG)."""
        cfg = self.config
        batch_size = prompt_embeds.shape[0]
        device = prompt_embeds.device
        dtype = prompt_embeds.dtype

        scheduler = copy.deepcopy(self.scheduler)
        use_dynamic_shifting = bool(getattr(scheduler.config, "use_dynamic_shifting", False))
        if self._is_flux2_klein and use_dynamic_shifting:
            mu = compute_empirical_mu(
                image_seq_len=self.latent_size * self.latent_size,
                num_steps=int(cfg.num_steps),
            )
            scheduler.set_timesteps(
                num_inference_steps=cfg.num_steps, device=device, mu=mu,
            )
        elif self._is_flux and use_dynamic_shifting:
            packed_tokens = (self.latent_size // 2) * (self.latent_size // 2)
            mu = compute_mu_flux(
                image_seq_len=packed_tokens,
                base_image_seq_len=int(getattr(scheduler.config, "base_image_seq_len", 256)),
                max_image_seq_len=int(getattr(scheduler.config, "max_image_seq_len", 4096)),
                base_shift=float(getattr(scheduler.config, "base_shift", 0.5)),
                max_shift=float(getattr(scheduler.config, "max_shift", 1.15)),
            )
            scheduler.set_timesteps(
                num_inference_steps=cfg.num_steps, device=device, mu=mu,
            )
        else:
            scheduler.set_timesteps(cfg.num_steps, device=device)

        x_t = torch.randn(
            batch_size, self.latent_channels, self.latent_size, self.latent_size,
            device=device, dtype=dtype,
        )

        for t in scheduler.timesteps:
            t_batch = t.expand(batch_size).to(device=device, dtype=torch.float32)
            with self._autocast():
                v_pred = self._predict_noise(
                    noisy_latents=x_t,
                    prompt_embeds=prompt_embeds,
                    timesteps=t_batch,
                    pooled_embeds=pooled_embeds,
                    guidance_scale=cfg.guidance_scale,
                    uncond_text_embeddings=uncond_embeds if cfg.guidance_scale > 1.0 else None,
                    uncond_pooled_prompt_embeds=uncond_pooled if cfg.guidance_scale > 1.0 else None,
                )
            x_t = scheduler.step(v_pred, t, x_t).prev_sample

        return x_t

    # ------------------------------------------------------------------
    # Full run
    # ------------------------------------------------------------------

    def run(self) -> dict[str, Any]:
        """Full pipeline: generate images, optionally run offline reward eval."""
        cfg = self.config
        rank = get_rank()
        os.makedirs(cfg.output_dir, exist_ok=True)

        if cfg.eval_reward and cfg.preload_reward_models:
            if is_main_process():
                logger.info("[Inference Eval] Preloading reward resources before generation")
            self._warmup_reward_eval_resources()
        elif cfg.eval_reward and is_main_process():
            logger.info(
                "[Inference Eval] preload_reward_models=false; rewards loaded lazily"
            )

        random.seed(cfg.seed + rank)
        np.random.seed(cfg.seed + rank)
        torch.manual_seed(cfg.seed + rank)
        torch.cuda.manual_seed_all(cfg.seed + rank)

        reward_dataset_map = cfg.reward_dataset_map or {}
        use_reward_specific_datasets = cfg.eval_reward and bool(reward_dataset_map)

        results: dict[str, Any] = {}
        total_time = 0.0
        num_local_generated = 0

        if use_reward_specific_datasets:
            results, total_time, num_local_generated = self._run_per_reward_dataset(
                reward_dataset_map=reward_dataset_map,
            )
        else:
            results, total_time, num_local_generated = self._run_single_dataset()

        num_generated_global = self._aggregate_num_samples_across_ranks(num_local_generated)
        total_time_global = self._aggregate_scalar_sum_across_ranks(total_time)
        if num_generated_global > 0 and is_main_process():
            logger.info(
                f"Generated {num_generated_global} images in {total_time_global:.1f}s "
                f"({total_time_global / num_generated_global:.2f}s/img, summed across ranks)"
            )

        if is_main_process():
            meta = {
                "config": cfg.to_dict(),
                "num_images": num_generated_global,
                "time_seconds_sum": total_time_global,
            }
            if "scores" in results:
                meta["scores"] = results["scores"]
            meta_path = os.path.join(cfg.output_dir, "metadata.json")
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2, default=str)
            logger.info(f"Metadata saved to {meta_path}")

        if cfg.distributed:
            barrier()
            cleanup_distributed()

        return results

    def _run_single_dataset(self) -> tuple[dict[str, Any], float, int]:
        """Generate + (optionally) evaluate using a single shared dataset."""
        cfg = self.config
        prompts, metadata = self._get_cached_eval_prompts_and_metadata(dataset_override=None)
        if is_main_process():
            logger.info(f"Total prompts: {len(prompts)}")

        images_tensor, prompts_local, metadata_local, num_local, elapsed = (
            self._generate_images_for_prompts(prompts, metadata, save_prefix="")
        )

        results: dict[str, Any] = {}
        if cfg.eval_reward:
            if self._scorer is None:
                self._init_scorer()
            self._scorer.to(torch.device("cuda", torch.cuda.current_device()))
            if images_tensor is not None and num_local > 0:
                score_details, _ = self._scorer(
                    images_tensor, prompts_local, metadata=metadata_local, only_strict=False,
                )
            else:
                score_details = {}
            local_metrics = self._score_details_to_mean_metrics(score_details)
            results["scores"] = self._aggregate_eval_metrics_across_ranks(local_metrics, num_local)
            self._scorer.to(torch.device("cpu"))
            torch.cuda.empty_cache()

        return results, elapsed, num_local

    def _run_per_reward_dataset(
        self,
        reward_dataset_map: dict[str, str],
    ) -> tuple[dict[str, Any], float, int]:
        """Evaluate rewards that come from different datasets (training-aligned)."""
        cfg = self.config
        if is_main_process():
            logger.info(
                "[Inference Eval] reward_dataset_map enabled: grouping rewards by dataset"
            )

        eval_metrics: dict[str, float] = {}
        weighted_mean = 0.0
        has_weighted = False
        dataset_to_rewards: dict[str, list[str]] = {}
        for reward_name in cfg.reward_fn.keys():
            ds_raw = reward_dataset_map.get(reward_name, None)
            dataset_override = (
                str(ds_raw).strip() if isinstance(ds_raw, str) and str(ds_raw).strip() else None
            )
            dataset_name = self._resolve_eval_dataset_name(dataset_override=dataset_override)
            dataset_to_rewards.setdefault(dataset_name, []).append(reward_name)

        total_time = 0.0
        num_local_generated = 0
        gpu_device = torch.device("cuda", torch.cuda.current_device())
        cpu_device = torch.device("cpu")

        for dataset_name, reward_names in dataset_to_rewards.items():
            prompt_dataset_override = None if dataset_name == "prompt_file" else dataset_name
            prompts, metadata = self._get_cached_eval_prompts_and_metadata(
                dataset_override=prompt_dataset_override,
            )
            if is_main_process():
                logger.info(
                    f"[Inference Eval] Dataset {dataset_name!r}: total_prompts={len(prompts)}, "
                    f"rewards={reward_names}"
                )

            save_prefix = f"{self._sanitize_output_token(dataset_name)}_"
            images_tensor, prompts_local, metadata_local, num_local, elapsed = (
                self._generate_images_for_prompts(prompts, metadata, save_prefix=save_prefix)
            )
            total_time += elapsed
            num_local_generated += num_local

            for reward_name in reward_names:
                scorer = self._get_single_reward_scorer(reward_name)
                active = getattr(scorer, "active_reward_names", [])
                if reward_name not in active:
                    if is_main_process():
                        logger.warning(
                            f"[Inference Eval] Skip unavailable reward {reward_name!r} "
                            f"for dataset {dataset_name!r}"
                        )
                    continue

                scorer.to(gpu_device)
                if images_tensor is not None and num_local > 0:
                    score_details, _ = scorer(
                        images_tensor, prompts_local, metadata=metadata_local, only_strict=False,
                    )
                else:
                    score_details = {}

                local_metrics = self._score_details_to_mean_metrics(score_details)
                aggregated = self._aggregate_eval_metrics_across_ranks(local_metrics, num_local)

                reward_value = aggregated.get(reward_name)
                if reward_value is not None:
                    weighted_mean += float(cfg.reward_fn[reward_name]) * float(reward_value)
                    has_weighted = True
                elif is_main_process():
                    logger.warning(
                        f"[Inference Eval] Reward {reward_name!r} has no valid metric; skipped."
                    )

                for key, value in aggregated.items():
                    if key == "mean":
                        continue
                    out_key = key
                    if out_key in eval_metrics and out_key != reward_name:
                        out_key = f"{reward_name}_{key}"
                    eval_metrics[out_key] = value

                scorer.to(cpu_device)
                torch.cuda.empty_cache()

        if has_weighted:
            eval_metrics["mean"] = weighted_mean
        return {"scores": eval_metrics}, total_time, num_local_generated
