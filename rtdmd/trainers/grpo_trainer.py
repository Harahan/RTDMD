"""
GRPO (Group Relative Policy Optimization) Trainer for RTDMD.

Pure RL fine-tuning of flow-matching models via SDE sampling + PPO policy gradients.
Completely independent of DMD — no teacher or fake score network required.

Training loop per epoch:
    1. [Sampling] SDE multi-step generation with log-probability tracking
    2. [Reward]   Score generated images via reward models (async)
    3. [Advantage] Per-prompt normalized advantages with cross-rank gathering
    4. [Training]  PPO clipped policy gradient with optional KL regularization

Class hierarchy:
    BaseTrainer → GRPOTrainer

Reference: grpo (https://arxiv.org/abs/2505.05470)
"""

from __future__ import annotations

import copy
import contextlib
import hashlib
import logging
import os
import random
import time
from collections import defaultdict
from concurrent import futures
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from diffusers import FlowMatchEulerDiscreteScheduler
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from rtdmd.config import RTDMDConfig
from rtdmd.diffusers_patch.sde_with_logprob import sde_step_with_logprob
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
)
from rtdmd.parallel.utils import (
    fsdp_wrap_model,
    get_transformer_wrap_policy,
    get_rank,
    get_world_size,
    is_main_process,
    barrier,
)
from rtdmd.trainers.base_trainer import BaseTrainer
from rtdmd.utils.checkpoint import (
    _is_lora_checkpoint,
    is_peft_lora_dir,
)
from rtdmd.utils.fast_init import fast_init
from rtdmd.utils.logging import get_gpu_memory_gb
from rtdmd.trainers.grpo_utils import DistributedKRepeatSampler, PerPromptStatTracker
from rtdmd.utils.ema import EMAModuleWrapper
from rtdmd.utils.lora import setup_lora

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _create_deterministic_generators(
    prompts: list[str], base_seed: int,
) -> list[torch.Generator]:
    """Create per-prompt deterministic noise generators via SHA256(prompt)."""
    generators = []
    for prompt in prompts:
        digest = hashlib.sha256(prompt.encode()).digest()
        prompt_hash = int.from_bytes(digest[:4], "big")
        seed = (base_seed + prompt_hash) % (2 ** 31)
        gen = torch.Generator().manual_seed(seed)
        generators.append(gen)
    return generators


# ---------------------------------------------------------------------------
# GRPOTrainer
# ---------------------------------------------------------------------------

class GRPOTrainer(BaseTrainer):
    """GRPO trainer: online RL fine-tuning via SDE sampling + PPO.

    Inherits BaseTrainer for distributed setup, checkpoint, logging, and EMA.
    Overrides _train_loop() with grpo's epoch-based structure:
    sampling → advantage → PPO training.

    Does NOT use teacher, fake_score_net, or DMD loss.
    """

    def __init__(self, config: RTDMDConfig):
        super().__init__(config)
        self.model_type = str(self.config.model.model_type).lower()
        self._is_flux2_klein = self.model_type == "flux2_klein"
        self._is_flux = self.model_type == "flux"
        if self.model_type not in {"sd35", "flux2_klein", "flux"}:
            raise ValueError(
                "GRPOTrainer supports model_type in ['sd35', 'flux2_klein', 'flux'], "
                f"got: {self.model_type}"
            )

        self.generator: nn.Module | None = None
        self.reference_model: nn.Module | None = None
        self.text_encoders: list[nn.Module] = []
        self.tokenizers: list = []
        self.scheduler: FlowMatchEulerDiscreteScheduler | None = None
        self.sde_scheduler: FlowMatchEulerDiscreteScheduler | None = None

        # GRPO state
        self._grpo_epoch: int = 0
        self._grpo_global_step: int = 0
        # Whether the pipeline should cache per-step noise_preds/prev_means/std_devs
        # (e.g. for fake-score x0 reconstruction in subclasses). Subclasses flip
        # this to True in setup when they need policy outputs.
        self._store_policy_outputs: bool = False
        # --- EMA Note ---
        # GRPO uses PARAMETER-LEVEL EMA (EMAModuleWrapper) rather than
        # MODEL-LEVEL EMA (deepcopy) used in DMD trainer.  This saves GPU memory
        # (especially with LoRA — only shadow-copies trainable params), but means
        # there is NO separate "generator_ema" model in self.models.
        #
        # Checkpoint structure (when use_ema=True) — mirrors DMD trainer layout:
        #   checkpoint-{step}/
        #   ├── transformer/                ← EMA weights in PEFT/diffusers format (for inference)
        #   ├── generator.pt                ← non-EMA training weights (for resume)
        #   ├── generator_ema.pt            ← EMA weights state_dict (for inference)
        #   ├── optimizer_generator.pt
        #   ├── meta.pt                     ← includes ema_state (for resume)
        #   └── grpo_state_rank{N}.pt
        #
        # - _get_extra_state_dicts() gathers an EMA snapshot via in-place
        #   swap-gather-swap and hands it to save_checkpoint(), which writes
        #   both generator_ema.pt and the inference-ready transformer/ folder.
        # - _save_final_transformer() saves EMA to {output_dir}/transformer/.
        # - Evaluation: temporarily swaps EMA weights in, runs eval, restores.
        # - Resume: loads training weights from generator.pt + ema_state from
        #   meta.pt. generator_ema.pt is for inference, NOT used for resume.
        #
        # This matches DMD trainer's on-disk layout without paying the
        # model-level deepcopy memory cost (EMA is parameter-level only).
        self._ema: EMAModuleWrapper | None = None
        self._trainable_params: list[nn.Parameter] = []
        self._stat_tracker: PerPromptStatTracker | None = None
        self._scorer = None
        self._eval_scorer_grpo = None
        self._grpo_sampler: DistributedKRepeatSampler | None = None
        self._grpo_dataloader: DataLoader | None = None
        self._grpo_sample_iter = None
        self._grpo_reward_requires_metadata: bool = False
        self._grpo_prompt_returns_metadata: bool = False

        # Latent shape (set in setup_models)
        self.latent_channels: int = 0
        self.latent_size: int = 0

        # Cached unconditional embeddings
        self._uncond_embeds: tuple | None = None

        # Lora mode flag
        self._lora_enabled: bool = False

        # Cache for backend-specific denoising sigma schedules.
        self._denoising_sigmas_cache: dict[tuple[int, ...], list[float]] = {}

    # ------------------------------------------------------------------
    # Model setup
    # ------------------------------------------------------------------
    def _load_model_init_from_path(
        self,
        model: nn.Module,
        init_path: str,
        role_name: str,
    ) -> None:
        """Load role-specific base-weight initialization checkpoint into a model."""
        if not init_path:
            return

        init_path = str(init_path).strip()
        if not init_path:
            return

        if is_peft_lora_dir(init_path):
            suffix = f"{role_name}_lora.load_path"
            raise ValueError(
                f"{role_name}_init_path expects base-weight checkpoints, but got peft LoRA dir: {init_path}. "
                f"Please use {suffix} for LoRA adapter initialization."
            )

        # .pt/.pth state_dict checkpoint
        if os.path.isfile(init_path):
            state_dict = torch.load(init_path, map_location="cpu", weights_only=False)
            is_lora = _is_lora_checkpoint(state_dict)
            if is_lora:
                suffix = f"{role_name}_lora.load_path"
                raise ValueError(
                    f"{role_name}_init_path expects base-weight checkpoints, but got LoRA-only state_dict: {init_path}. "
                    f"Please use {suffix} for LoRA adapter initialization."
                )
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            logger.info(
                f"  {role_name}: loaded init checkpoint {init_path} "
                f"(keys={len(state_dict)}, missing={len(missing)}, unexpected={len(unexpected)})"
            )
            return

        # Either a local diffusers-style dir or a HuggingFace Hub repo id.
        # For local paths we still inspect the on-disk layout so we can pick
        # between a bare transformer dir and a checkpoint dir containing a
        # `transformer/` subfolder. For hub ids we delegate to
        # `from_pretrained`, which downloads + caches transparently and
        # raises `RepositoryNotFoundError` on truly invalid ids.
        model_dir = init_path
        subfolder: str | None = None
        if os.path.isdir(init_path):
            if not os.path.exists(os.path.join(model_dir, "config.json")):
                alt_dir = os.path.join(model_dir, "transformer")
                if os.path.exists(os.path.join(alt_dir, "config.json")):
                    model_dir = alt_dir
                else:
                    raise FileNotFoundError(
                        f"{role_name}_init_path has no config.json at {init_path} or {alt_dir}"
                    )
        else:
            # HuggingFace Hub repo id: full-pipeline repos keep the transformer
            # under a `transformer/` subfolder; bare-transformer repos do not.
            # Try the subfolder layout first since that's what the Stability /
            # Black-Forest-Labs releases use; fall back to bare on failure.
            subfolder = "transformer"

        model_cls = model.__class__
        with fast_init(torch.device("cpu")):
            try:
                loaded_model = model_cls.from_pretrained(
                    model_dir,
                    subfolder=subfolder,
                    torch_dtype=next(model.parameters()).dtype,
                )
            except (OSError, EnvironmentError, ValueError):
                if subfolder is None:
                    raise
                loaded_model = model_cls.from_pretrained(
                    model_dir,
                    torch_dtype=next(model.parameters()).dtype,
                )
        state_dict = loaded_model.state_dict()
        del loaded_model

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        logger.info(
            f"  {role_name}: loaded init transformer {model_dir}"
            + (f" (subfolder={subfolder})" if subfolder else "")
            + f" (keys={len(state_dict)}, missing={len(missing)}, unexpected={len(unexpected)})"
        )

    def setup_models(self) -> None:
        """Load backend models for GRPO training.

        Creates only: generator (+ optional reference_model for KL when full-weight).
        Does NOT create teacher or fake_score_net.
        """
        model_cfg = self.config.model
        dist_cfg = self.config.distributed
        grpo_cfg = self.config.grpo

        logger.info("=" * 60)
        logger.info("Setting up GRPO models")
        logger.info("=" * 60)

        # Load base model components by backend.
        if self._is_flux2_klein:
            components = load_flux2_klein_models(model_cfg)
        elif self._is_flux:
            components = load_flux_models(model_cfg)
        else:
            components = load_sd35_models(model_cfg)
        base_transformer = components["transformer"]
        self.text_encoders = components["text_encoders"]
        self.tokenizers = components["tokenizers"]
        self.scheduler = components["scheduler"]
        self.vae = components["vae"]
        self._uncond_embeds = None
        self._denoising_sigmas_cache.clear()
        self.vae.requires_grad_(False)
        self.vae.eval()
        self.vae.to(torch.device("cuda"))

        # Latent shape
        vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        if self._is_flux2_klein:
            # Flux2 uses 2x2 latent patchification before transformer tokens, so
            # transformer.in_channels already accounts for packed channels.
            self.latent_channels = base_transformer.config.in_channels
            latent_divisor = vae_scale_factor * 2
        elif self._is_flux:
            # FLUX.1: transformer.in_channels=64 is the *packed* channel count
            # (16 VAE channels x 2x2 packing). The VAE-latent-domain channel
            # count we operate on in latents tensors is 16, derived from
            # vae.config.latent_channels. Packing happens inside
            # predict_noise_flux. Latent H/W = image_res / vae_scale_factor.
            self.latent_channels = int(self.vae.config.latent_channels)
            latent_divisor = vae_scale_factor
        else:
            self.latent_channels = base_transformer.config.in_channels
            latent_divisor = vae_scale_factor
        if model_cfg.image_resolution % latent_divisor != 0:
            raise ValueError(
                f"image_resolution={model_cfg.image_resolution} must be divisible by {latent_divisor} "
                f"for model_type={self.model_type}"
            )
        self.latent_size = model_cfg.image_resolution // latent_divisor
        if self._is_flux and self.latent_size % 2 != 0:
            raise ValueError(
                f"FLUX requires latent_size (={self.latent_size}) divisible by 2 "
                f"for 2x2 packing; choose image_resolution divisible by "
                f"{vae_scale_factor * 2}."
            )
        logger.info(
            f"  Latent shape: [{self.latent_channels}, {self.latent_size}, {self.latent_size}] "
            f"(image_res={model_cfg.image_resolution}, latent_divisor={latent_divisor})"
        )

        # Enable gradient checkpointing
        if model_cfg.gradient_checkpointing:
            base_transformer.enable_gradient_checkpointing()

        # --- Generator ---
        self._lora_enabled = model_cfg.generator_lora.enabled
        self.generator = copy.deepcopy(base_transformer)
        # Role-specific base init must happen before LoRA injection / merge.
        if self._is_redundant_pretrained_init_path(model_cfg.generator_init_path):
            logger.info(
                "  generator_init_path matches pretrained_path; "
                "skip redundant initialization reload."
            )
        else:
            self._load_model_init_from_path(
                self.generator,
                model_cfg.generator_init_path,
                "generator",
            )

        if self._lora_enabled:
            setup_lora(self.generator, model_cfg.generator_lora)
            logger.info(f"  Generator: LoRA injected (rank={model_cfg.generator_lora.rank})")
        else:
            self.generator.train()
            logger.info("  Generator: full-weight training")

        # --- Reference model (for KL regularization) ---
        self.reference_model = None
        if grpo_cfg.beta > 0 and not self._lora_enabled:
            # Full-weight: frozen copy of base_transformer
            self.reference_model = base_transformer
            self.reference_model.requires_grad_(False)
            self.reference_model.eval()
            logger.info("  Reference: frozen base transformer")
        else:
            if grpo_cfg.beta > 0:
                logger.info("  Reference: via disable_adapter() (LoRA mode, no extra memory)")
            else:
                logger.info("  Reference: disabled (beta=0)")
            # Free base_transformer — generator already has its own deepcopy,
            # and LoRA KL uses disable_adapter() on the generator itself.
            del base_transformer
            import gc; gc.collect()
            torch.cuda.empty_cache()

        # Freeze text encoders
        for enc in self.text_encoders:
            enc.requires_grad_(False)
            enc.eval()

        # --- SDE scheduler ---
        self.sde_scheduler = FlowMatchEulerDiscreteScheduler.from_config(
            self.scheduler.config
        )

        # --- Pre-wrap hook (opt-in SD3-Medium PatchEmbed fix) ---
        self._pre_wrap_models()

        # --- FSDP/DDP wrapping ---
        self._wrap_models(dist_cfg)

        # Register models for checkpointing (reference_model excluded — it's frozen pretrained,
        # reloaded from pretrained_path on resume, no need to checkpoint)
        self.models = {"generator": self.generator}

        # Move text encoders to GPU
        for enc in self.text_encoders:
            enc.to(torch.device("cuda"))

        # --- Trainable parameters & EMA ---
        self._trainable_params = list(
            filter(lambda p: p.requires_grad, self.generator.parameters())
        )
        if grpo_cfg.use_ema:
            self._ema = EMAModuleWrapper(
                self._trainable_params,
                decay=grpo_cfg.ema_decay,
                update_step_interval=grpo_cfg.ema_update_interval,
                device=torch.device("cuda"),
            )
            logger.info(
                f"  EMA: parameter-level (decay={grpo_cfg.ema_decay}, "
                f"interval={grpo_cfg.ema_update_interval})"
            )

        # --- Stat tracker ---
        self._stat_tracker = PerPromptStatTracker(
            global_std=grpo_cfg.per_prompt_global_std
        )

        # --- K-repeat dataloader ---
        self._setup_grpo_dataloader()

        # --- Validate batch constraints ---
        world_size = get_world_size()
        total_samples_per_epoch = grpo_cfg.num_batches_per_epoch * grpo_cfg.sample_batch_size * world_size
        K = grpo_cfg.num_image_per_prompt
        if total_samples_per_epoch % K != 0:
            raise ValueError(
                f"num_batches_per_epoch * sample_batch_size * world_size ({total_samples_per_epoch}) "
                f"must be divisible by num_image_per_prompt ({K})"
            )
        if (grpo_cfg.sample_batch_size * world_size) % K != 0:
            raise ValueError(
                f"sample_batch_size * world_size ({grpo_cfg.sample_batch_size * world_size}) "
                f"must be divisible by num_image_per_prompt ({K})"
            )

        logger.info(
            f"  GRPO: num_steps={grpo_cfg.num_steps}, noise_level={grpo_cfg.noise_level}, "
            f"sde_type={grpo_cfg.sde_type}, guidance_scale={grpo_cfg.guidance_scale}, "
            f"beta={grpo_cfg.beta}, clip_range={grpo_cfg.clip_range}, "
            f"K={grpo_cfg.num_image_per_prompt}, num_batches_per_epoch={grpo_cfg.num_batches_per_epoch}"
        )
        logger.info("Model setup complete")
        logger.info("=" * 60)

    def _build_transformer_wrap_policy(self):
        """Create FSDP auto-wrap policy for the active transformer backend."""
        try:
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
        except ImportError:
            return None

    def _load_pretrained_transformer(self, torch_dtype: torch.dtype) -> nn.Module:
        """Load transformer model matching config.model.model_type."""
        if self._is_flux2_klein:
            from diffusers import Flux2Transformer2DModel

            return Flux2Transformer2DModel.from_pretrained(
                self.config.model.pretrained_path,
                subfolder="transformer",
                torch_dtype=torch_dtype,
            )

        if self._is_flux:
            from diffusers import FluxTransformer2DModel

            return FluxTransformer2DModel.from_pretrained(
                self.config.model.pretrained_path,
                subfolder="transformer",
                torch_dtype=torch_dtype,
            )

        from diffusers import SD3Transformer2DModel

        return SD3Transformer2DModel.from_pretrained(
            self.config.model.pretrained_path,
            subfolder="transformer",
            torch_dtype=torch_dtype,
        )

    def _maybe_replace_patch_embed_with_linear(self, model: nn.Module | None) -> None:
        """Apply the SD3-Medium PatchEmbed Conv2d -> Linear workaround to ONE model.

        Gated by ``config.model.fsdp_patch_embed_linear``. No-op when the flag
        is False, when ``model is None``, or when the model has no
        Conv2d-based ``pos_embed.proj`` (e.g., Flux2-Klein). Idempotent — safe
        to call twice on the same model.

        Subclasses that load additional transformers AFTER ``super().setup_models()``
        (e.g., RTDMDTrainer's teacher / fake_score_net) must call this
        helper on each newly-created model BEFORE FSDP wrapping, otherwise
        those models keep their Conv2d PatchEmbed and trigger the FSDP
        ``use_orig_params=True`` writeback bug.
        """
        if model is None:
            return
        if not self.config.model.fsdp_patch_embed_linear:
            return
        from rtdmd.trainers.dmd_trainer import _replace_patch_embed_proj_with_linear
        _replace_patch_embed_proj_with_linear(model)

    def _pre_wrap_models(self) -> None:
        """Hook called after models are created but BEFORE FSDP/DDP wrapping.

        When config.model.fsdp_patch_embed_linear is True, replaces any
        Conv2d-based PatchEmbed.proj with an equivalent Linear projection
        (SD3-Medium FSDP compatibility fix). Default False keeps SD3.5 /
        Flux2 / etc. behavior and saved checkpoints unchanged. Mirrors
        DMDTrainer._pre_wrap_models() so the same flag works here.
        """
        self._maybe_replace_patch_embed_with_linear(self.generator)
        self._maybe_replace_patch_embed_with_linear(self.reference_model)

    def _wrap_models(self, dist_cfg) -> None:
        """FSDP/DDP wrap trainable models."""
        fsdp_precision = self.config.model.dtype
        if dist_cfg.strategy == "fsdp":
            wrap_policy = self._build_transformer_wrap_policy()

            self.generator = fsdp_wrap_model(
                self.generator,
                sharding_strategy=dist_cfg.fsdp_sharding,
                fsdp_precision=fsdp_precision,
                auto_wrap_policy=wrap_policy,
            )
            if self.reference_model is not None:
                self.reference_model = fsdp_wrap_model(
                    self.reference_model,
                    sharding_strategy=dist_cfg.fsdp_sharding,
                    fsdp_precision=fsdp_precision,
                    auto_wrap_policy=wrap_policy,
                )
            logger.info(f"  Models wrapped with FSDP (sharding={dist_cfg.fsdp_sharding})")
        else:
            from rtdmd.parallel.utils import ddp_wrap_model
            device = torch.device("cuda")
            has_lora = self.config.model.generator_lora.enabled
            self.generator = self.generator.to(device)
            if dist.is_initialized() and get_world_size() > 1:
                self.generator = ddp_wrap_model(
                    self.generator,
                    find_unused_parameters=has_lora,
                    broadcast_buffers=False,
                )
            if self.reference_model is not None:
                self.reference_model = self.reference_model.to(device)
            logger.info("  Models wrapped with DDP" if dist.is_initialized() and get_world_size() > 1 else "  Models on GPU (single process)")

    def _setup_grpo_dataloader(self) -> None:
        """Create K-repeat dataloader for GRPO sampling."""
        from rtdmd.data.prompt_dataset import PromptDataset

        grpo_cfg = self.config.grpo
        reward_names = set(grpo_cfg.reward_fn.keys())
        self._grpo_reward_requires_metadata = "geneval" in reward_names
        dataset = PromptDataset(
            self.config.prompt_path,
            return_metadata=self._grpo_reward_requires_metadata,
        )
        self._grpo_prompt_returns_metadata = self._grpo_reward_requires_metadata

        if self._grpo_reward_requires_metadata and len(dataset) > 0:
            first = dataset[0]
            if not isinstance(first, dict):
                raise ValueError(
                    "geneval is enabled in grpo.reward_fn, but prompt dataset "
                    "does not return metadata entries."
                )
            first_meta = first.get("metadata", {})
            if not isinstance(first_meta, dict):
                raise ValueError(
                    "geneval reward expects dict metadata per prompt, but got "
                    f"{type(first_meta).__name__}."
                )
            # Geneval needs structured fields beyond prompt text itself.
            if set(first_meta.keys()).issubset({"prompt", "text"}):
                raise ValueError(
                    "geneval is enabled in grpo.reward_fn, but prompt metadata "
                    "is missing Geneval attributes. Use metadata jsonl "
                    "(e.g., dataset/geneval/train_metadata.jsonl) as prompt source."
                )

        self._grpo_sampler = DistributedKRepeatSampler(
            dataset_size=len(dataset),
            batch_size=grpo_cfg.sample_batch_size,
            k=grpo_cfg.num_image_per_prompt,
            num_replicas=get_world_size(),
            rank=get_rank(),
            seed=self.config.train.seed,
        )
        self._grpo_dataloader = DataLoader(
            dataset,
            batch_sampler=self._grpo_sampler,
            num_workers=0,
            collate_fn=lambda batch: batch,
        )
        self._grpo_sampler.set_epoch(self._grpo_epoch)
        self._grpo_sample_iter = iter(self._grpo_dataloader)
        logger.info(
            f"  GRPO dataloader: dataset_size={len(dataset)}, "
            f"K={grpo_cfg.num_image_per_prompt}, sample_bs={grpo_cfg.sample_batch_size}, "
            f"metadata={'on' if self._grpo_prompt_returns_metadata else 'off'}"
        )

    # ------------------------------------------------------------------
    # Optimizer setup
    # ------------------------------------------------------------------

    def setup_optimizers(self) -> None:
        """Create optimizer for generator only (no fake_score)."""
        grpo_cfg = self.config.grpo

        self.optimizers["generator"] = torch.optim.AdamW(
            self._trainable_params,
            lr=grpo_cfg.learning_rate,
            betas=(grpo_cfg.adam_beta1, grpo_cfg.adam_beta2),
            eps=grpo_cfg.adam_epsilon,
            weight_decay=grpo_cfg.adam_weight_decay,
        )

        # No LR scheduler (grpo uses constant LR)
        self.schedulers["generator"] = self.create_warmup_constant_scheduler(
            self.optimizers["generator"], warmup_steps=0,
        )

        logger.info(
            f"Optimizer created: generator (lr={grpo_cfg.learning_rate}, "
            f"params={len(self._trainable_params)})"
        )

    # ------------------------------------------------------------------
    # Dataloader (required by BaseTrainer)
    # ------------------------------------------------------------------

    def _create_dataloader(self):
        """Not used — GRPOTrainer overrides _train_loop() entirely.

        Returns (None, None) to satisfy BaseTrainer's abstract interface.
        The actual data loading is handled by _setup_grpo_dataloader().
        """
        return None, None

    # ------------------------------------------------------------------
    # Train step (not used — we override _train_loop)
    # ------------------------------------------------------------------

    def train_step(self, batch: Any) -> dict[str, dict[str, float]]:
        """Not used — GRPOTrainer overrides _train_loop() directly."""
        raise NotImplementedError("GRPOTrainer uses _train_loop() override")

    # ------------------------------------------------------------------
    # Step selection helper
    # ------------------------------------------------------------------

    def _resolve_step_selection(self, grpo_cfg) -> dict | None:
        """Validate and resolve step_selection config.

        Returns a validated dict with keys ``range_start``, ``range_end``,
        ``num_train_steps`` if all conditions are met, or ``None`` to fall
        back to ``timestep_fraction``.

        Conditions for activation:
        - ``step_selection.enabled`` is True
        - ``sampling_mode == "dmd"`` and ``sde_type == "cps"``
        - Range and count are valid for the current ``denoising_step_list``
        """
        sel = grpo_cfg.step_selection
        if not sel or not sel.get("enabled", False):
            return None
        if grpo_cfg.sampling_mode != "dmd" or grpo_cfg.sde_type != "cps":
            logger.warning(
                "step_selection requires sampling_mode='dmd' and sde_type='cps', "
                "falling back to timestep_fraction"
            )
            return None
        total_steps = len(grpo_cfg.denoising_step_list)
        range_start = sel.get("range_start", 0)
        range_end = sel.get("range_end", total_steps - 1)
        num_train = sel.get("num_train_steps", 1)
        if range_end >= total_steps or range_start < 0 or range_start > range_end:
            logger.warning(
                f"step_selection range [{range_start}, {range_end}] invalid "
                f"for {total_steps} steps, falling back to timestep_fraction"
            )
            return None
        if num_train > (range_end - range_start + 1):
            logger.warning(
                f"step_selection num_train_steps={num_train} > range size="
                f"{range_end - range_start + 1}, falling back to timestep_fraction"
            )
            return None
        return {
            "range_start": range_start,
            "range_end": range_end,
            "num_train_steps": num_train,
        }

    def _sample_selected_steps(
        self,
        step_sel_config: dict,
        *,
        epoch_seed: int,
    ) -> list[int]:
        """Deterministically sample step indices without distributed collectives.

        Uses an epoch-scoped local RNG so every rank samples the same steps
        without relying on ``dist.broadcast``. This avoids rank desync deadlocks
        when one rank lags or diverges before entering a collective.
        """
        candidates = list(range(
            int(step_sel_config["range_start"]),
            int(step_sel_config["range_end"]) + 1,
        ))
        num_train_steps = int(step_sel_config["num_train_steps"])
        # Keep sampling deterministic across ranks while decoupling from global
        # RNG state consumed by rollout/eval paths.
        base_seed = int(getattr(self.config.train, "seed", 0))
        rng = random.Random(base_seed + int(epoch_seed) + 104729)
        return sorted(rng.sample(candidates, num_train_steps))
    def _get_denoising_sigmas(self, denoising_step_list: list[int]) -> list[float]:
        """Return backend-specific shifted sigma list for custom DMD sampling."""
        key = tuple(int(x) for x in denoising_step_list)
        if key in self._denoising_sigmas_cache:
            return self._denoising_sigmas_cache[key]

        if self._is_flux2_klein:
            sigmas = compute_sigmas_flux2_klein(
                denoising_step_list=list(denoising_step_list),
                scheduler_config=self.scheduler.config,
                image_seq_len=self.latent_size * self.latent_size,
                num_steps=len(denoising_step_list),
            )
        elif self._is_flux:
            # FLUX image_seq_len = number of packed image tokens =
            # (latent_H / 2) * (latent_W / 2). For 512x512 -> 1024 tokens,
            # 1024x1024 -> 4096 tokens.
            flux_image_seq_len = (self.latent_size // 2) * (self.latent_size // 2)
            sigmas = compute_sigmas_flux(
                denoising_step_list=list(denoising_step_list),
                scheduler_config=self.scheduler.config,
                image_seq_len=flux_image_seq_len,
                num_steps=len(denoising_step_list),
            )
        else:
            sigmas = compute_sigmas_sd35(
                list(denoising_step_list),
                num_train_timesteps=int(self._num_train_timesteps()),
                shift=self.scheduler.config.shift,
            )
        self._denoising_sigmas_cache[key] = sigmas
        return sigmas

    def _predict_noise(
        self,
        model: nn.Module,
        noisy_latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
        timesteps: torch.Tensor,
        pooled_embeds: torch.Tensor,
        guidance_scale: float = 1.0,
        uncond_text_embeddings: torch.Tensor | None = None,
        uncond_pooled_prompt_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Backend-dispatched velocity prediction with optional CFG.

        FLUX.1 guidance semantics differ from SD3.5:
        - SD3.5: ``guidance_scale > 1.0`` triggers traditional CFG (uncond+cond
          doubling).
        - FLUX.1-dev (``guidance_embeds=True``): ``guidance_scale`` is fed to
          the model as a scalar input for distilled CFG (single forward pass);
          uncond_* arguments are ignored.
        - FLUX.1-schnell / wo-guidance-embed (``guidance_embeds=False``):
          same as SD3.5 — traditional CFG when ``guidance_scale > 1.0`` and
          uncond_* embeddings are provided.
        """
        if self._is_flux2_klein:
            return predict_noise_flux2_klein(
                model=model,
                noisy_latents=noisy_latents,
                prompt_embeds=prompt_embeds,
                timesteps=timesteps,
                text_ids=pooled_embeds,
                guidance_scale=guidance_scale,
                uncond_prompt_embeds=uncond_text_embeddings,
                uncond_text_ids=uncond_pooled_prompt_embeds,
                num_train_timesteps=int(self._num_train_timesteps()),
            )

        if self._is_flux:
            return predict_noise_flux(
                model=model,
                noisy_latents=noisy_latents,
                prompt_embeds=prompt_embeds,
                timesteps=timesteps,
                pooled_embeds=pooled_embeds,
                guidance_scale=guidance_scale,
                uncond_text_embeddings=uncond_text_embeddings,
                uncond_pooled_prompt_embeds=uncond_pooled_prompt_embeds,
                num_train_timesteps=int(self._num_train_timesteps()),
            )

        return predict_noise_sd35(
            model,
            noisy_latents,
            prompt_embeds,
            timesteps,
            pooled_embeds,
            guidance_scale=guidance_scale,
            uncond_text_embeddings=uncond_text_embeddings,
            uncond_pooled_prompt_embeds=uncond_pooled_prompt_embeds,
        )

    # ------------------------------------------------------------------
    # Main training loop override
    # ------------------------------------------------------------------

    def _train_loop(self) -> None:
        """GRPO epoch-based training loop."""
        grpo_cfg = self.config.grpo

        # Autocast: LoRA + fp32 model → nullcontext (matches original grpo);
        # LoRA + bf16/fp16 model → need autocast to cast fp32 inputs to model dtype.
        if self._lora_enabled and self.config.model.dtype in ("fp32", "no"):
            autocast_ctx = contextlib.nullcontext
        else:
            autocast_ctx = self._autocast

        # Resume
        self._restore_rng_state()

        # Reward scorer
        self._init_scorer()

        reward_workers_raw = os.environ.get("RTDMD_GRPO_REWARD_WORKERS", "1")
        try:
            reward_workers = max(1, int(reward_workers_raw))
        except ValueError:
            logger.warning(
                "Invalid RTDMD_GRPO_REWARD_WORKERS=%r, fallback to 1",
                reward_workers_raw,
            )
            reward_workers = 1
        if reward_workers > 1:
            logger.warning(
                "Using RTDMD_GRPO_REWARD_WORKERS=%d. Multiple CUDA worker threads "
                "can increase ProcessGroupNCCL watchdog/GIL deadlock risk.",
                reward_workers,
            )
        executor = futures.ThreadPoolExecutor(max_workers=reward_workers)

        # Pre-compute negative embeddings
        neg_embeds, neg_pooled = self._get_uncond_embeds(grpo_cfg.sample_batch_size)
        train_neg_embeds, train_neg_pooled = self._get_uncond_embeds(grpo_cfg.train_batch_size)

        # Resolve step selection (DMD+CPS random step training)
        step_sel_config = self._resolve_step_selection(grpo_cfg)

        if step_sel_config is not None:
            # step_selection active: expand advantages to full pipeline T,
            # timestep_fraction is ignored
            num_train_timesteps = len(grpo_cfg.denoising_step_list)
            logger.info(
                f"  Step selection enabled: range=[{step_sel_config['range_start']}, "
                f"{step_sel_config['range_end']}], num_train_steps="
                f"{step_sel_config['num_train_steps']}, "
                f"num_train_timesteps(adv)={num_train_timesteps}"
            )
        else:
            num_train_timesteps = int(grpo_cfg.num_steps * grpo_cfg.timestep_fraction)

        # effective_grad_accum is computed per-epoch in _run_grpo_epoch
        # (depends on actual training step count which may vary with step_selection)

        clip_lt = grpo_cfg.clip_range_lt if grpo_cfg.clip_range_lt > 0 else grpo_cfg.clip_range
        clip_gt = grpo_cfg.clip_range_gt if grpo_cfg.clip_range_gt > 0 else grpo_cfg.clip_range

        logger.info(
            f"Starting GRPO: epochs={grpo_cfg.num_epochs}, "
            f"num_train_timesteps={num_train_timesteps}"
        )

        # Auto-disable per-prompt tracking when K=1 (std=0 → all advantages=0)
        if grpo_cfg.num_image_per_prompt == 1:
            grpo_cfg.per_prompt_stat_tracking = False
            logger.warning("  per_prompt_stat_tracking disabled: num_image_per_prompt=1")

        while self._grpo_epoch < grpo_cfg.num_epochs:
            # Eval at epoch boundary
            if grpo_cfg.eval_freq > 0 and self._grpo_epoch % grpo_cfg.eval_freq == 0:
                with self.timer.measure("eval"):
                    self._evaluate_grpo()

            # Save at epoch boundary
            if grpo_cfg.save_freq > 0 and self._grpo_epoch % grpo_cfg.save_freq == 0 and self._grpo_epoch > 0:
                self._save_checkpoint()

            with self.timer.measure("epoch"):
                epoch_metrics = self._run_grpo_epoch(
                    grpo_cfg, autocast_ctx, executor,
                    neg_embeds, neg_pooled,
                    train_neg_embeds, train_neg_pooled,
                    num_train_timesteps,
                    clip_lt, clip_gt,
                    step_sel_config=step_sel_config,
                )

            # Separate wandb sections: grpo_epoch/ for rewards, profile/ for timing+mem
            # Split epoch metrics into wandb sections
            reward_metrics = {k: v for k, v in epoch_metrics.items() if k.startswith("reward_")}
            adv_epoch_metrics = {k: v for k, v in epoch_metrics.items() if k.startswith("advantage_")}
            stage_time_metrics = {k: v for k, v in epoch_metrics.items() if k.startswith("time_")}
            epoch_misc = {k: v for k, v in epoch_metrics.items()
                          if (
                              not k.startswith("reward_")
                              and not k.startswith("advantage_")
                              and not k.startswith("time_")
                          )}
            epoch_misc["lr"] = self.schedulers["generator"].get_last_lr()[0]

            profile_metrics = {"gpu_mem_gb": get_gpu_memory_gb()}
            profile_metrics.update(self.timer.get_metrics())
            # Per-epoch stage timings (rollout/fake/training/total) are logged
            # under profile/ to keep all runtime diagnostics in one panel.
            profile_metrics.update(stage_time_metrics)

            if self.wandb_logger:
                self.wandb_logger.log_multi_section({
                    "grpo_reward": reward_metrics,
                    "grpo_advantage": adv_epoch_metrics,
                    "grpo_epoch": epoch_misc,
                    "profile": profile_metrics,
                }, step=self._grpo_global_step)

            if is_main_process():
                all_metrics = {**epoch_metrics, **profile_metrics}
                all_metrics["lr"] = epoch_misc["lr"]
                parts = [f"[Epoch {self._grpo_epoch}]"]
                for k, v in all_metrics.items():
                    parts.append(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}")
                logger.info(" | ".join(parts))

            self._grpo_epoch += 1

        logger.info(f"Training complete: {self._grpo_epoch} epochs, {self._grpo_global_step} steps")
        self._save_checkpoint()
        self._save_final_transformer()

    # ------------------------------------------------------------------
    # Single GRPO epoch
    # ------------------------------------------------------------------

    @staticmethod
    def _sync_time() -> float:
        """Return wall-clock time after draining CUDA queues."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return time.perf_counter()

    @staticmethod
    def _build_stage_time_metrics(
        rollout_s: float,
        training_s: float,
        fake_update_s: float = 0.0,
    ) -> dict[str, float]:
        """Build standardized per-epoch stage timing metrics."""
        rollout_s = float(rollout_s)
        training_s = float(training_s)
        fake_update_s = float(fake_update_s)
        return {
            "time_fake_update_s": fake_update_s,
            "time_rollout_s": rollout_s,
            "time_training_s": training_s,
            "time_total_s": rollout_s + training_s + fake_update_s,
        }
    def _split_prompt_batch(
        batch: list[Any],
    ) -> tuple[list[str], list[dict] | None]:
        """Normalize dataloader batch to (prompts, optional metadata list)."""
        if not batch:
            return [], None

        first = batch[0]
        if isinstance(first, str):
            return [str(x) for x in batch], None

        if isinstance(first, dict):
            prompts: list[str] = []
            metadatas: list[dict] = []
            for item in batch:
                if not isinstance(item, dict):
                    raise TypeError(
                        "Mixed prompt batch types detected; expected all dict entries "
                        "when metadata mode is enabled."
                    )
                prompt = item.get("prompt", item.get("text", ""))
                prompts.append(str(prompt))
                metadata = item.get("metadata", {})
                if metadata is None:
                    metadata = {}
                if not isinstance(metadata, dict):
                    raise TypeError(
                        "Prompt metadata must be a dict, "
                        f"got {type(metadata).__name__}."
                    )
                metadatas.append(metadata)
            return prompts, metadatas

        raise TypeError(
            "Unsupported prompt batch entry type: "
            f"{type(first).__name__}. Expected str or dict."
        )

    def _grpo_sampling_stage(
        self,
        grpo_cfg,
        autocast_ctx,
        executor,
        neg_embeds,
        neg_pooled,
        selected_steps: list[int] | None = None,
        epoch_seed: int = 0,
    ) -> tuple[dict, list[str], list]:
        """SDE/DMD sampling with async reward scoring.

        Runs ``num_batches_per_epoch`` sampling iterations via K-repeat dataloader.
        Each iteration: encode prompts → SDE/DMD pipeline → async reward submission.
        Waits for all rewards, then concatenates into a single samples dict.

        Args:
            grpo_cfg: GRPOConfig instance.
            autocast_ctx: Autocast context manager.
            executor: ThreadPoolExecutor for async reward computation.
            neg_embeds: Unconditional text embeddings [sample_bs, seq, dim].
            neg_pooled: Unconditional pooled embeddings [sample_bs, dim].
            selected_steps: Pipeline step indices selected for training (or None).
                When provided, non-selected steps use prompt-deterministic shared
                CPS noise to reduce advantage variance.
            epoch_seed: Seed offset for deterministic noise generation in non-trained
                steps (varies per epoch).

        Returns:
            samples: Dict of concatenated CPU tensors with keys: ``prompt_embeds``
                [N, seq, dim], ``pooled_prompt_embeds`` [N, dim], ``timesteps`` [N, T],
                ``latents`` [N, T, C, H, W], ``next_latents`` [N, T, C, H, W],
                ``log_probs`` [N, T], ``rewards`` {name: [N]}.
            all_prompts_local: Flat list of all prompts on this rank (len=N).
            all_images_np: List of uint8 numpy arrays for image logging (empty if disabled).
        """
        self.generator.eval()
        samples_list = []
        all_prompts_local = []
        all_images_np = []

        scorer_device = (
            torch.device("cuda", torch.cuda.current_device())
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        self._scorer.to(scorer_device)
        try:
            for i in range(grpo_cfg.num_batches_per_epoch):
                sampler_epoch = self._grpo_epoch * grpo_cfg.num_batches_per_epoch + i
                self._grpo_sampler.set_epoch(sampler_epoch)
                try:
                    batch_items = next(self._grpo_sample_iter)
                except StopIteration:
                    self._grpo_sample_iter = iter(self._grpo_dataloader)
                    batch_items = next(self._grpo_sample_iter)

                prompts, metadatas = self._split_prompt_batch(batch_items)
                if not prompts:
                    raise RuntimeError("Encountered empty prompt batch during GRPO sampling.")

                bs = len(prompts)
                all_prompts_local.extend(prompts)

                with torch.no_grad():
                    prompt_embeds, pooled_embeds = self._encode_prompts(prompts)

                generator = None
                if grpo_cfg.same_latent:
                    generator = _create_deterministic_generators(
                        prompts, base_seed=self._grpo_epoch * 10000 + i
                    )

                with autocast_ctx():
                    with torch.no_grad():
                        images, all_latents, all_log_probs, policy_outputs = (
                            self._pipeline_with_logprob(
                                prompt_embeds, pooled_embeds,
                                neg_embeds[:bs], neg_pooled[:bs],
                                grpo_cfg, generator=generator,
                                prompts=prompts,
                                selected_train_steps=(
                                    set(selected_steps) if selected_steps else None
                                ),
                                epoch_seed=epoch_seed,
                            )
                        )

                _should_log_images = (
                    grpo_cfg.log_num_groups > 0
                    and (grpo_cfg.log_image_freq == 0 or self._grpo_epoch % grpo_cfg.log_image_freq == 0)
                )
                if _should_log_images:
                    for img in images:
                        all_images_np.append(
                            (img.permute(1, 2, 0) * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
                        )

                rewards_future = executor.submit(
                    self._compute_rewards,
                    images,
                    prompts,
                    metadatas,
                )
                time.sleep(0)

                timesteps = self.sde_scheduler.timesteps.repeat(bs, 1).to(prompt_embeds.device)

                sample_entry = {
                    "prompt_embeds": prompt_embeds.cpu(),
                    "pooled_prompt_embeds": pooled_embeds.cpu(),
                    "timesteps": timesteps.cpu(),
                    "latents": all_latents[:, :-1].cpu(),
                    "next_latents": all_latents[:, 1:].cpu(),
                    "log_probs": all_log_probs.cpu(),
                    "rewards": rewards_future,
                    "prompts": prompts,
                }
                if policy_outputs is not None:
                    for pk, pv in policy_outputs.items():
                        sample_entry[pk] = pv.cpu()
                samples_list.append(sample_entry)

            for sample in samples_list:
                rewards, _ = sample["rewards"].result()
                sample["rewards"] = {
                    k: torch.as_tensor(v, dtype=torch.float32)
                    for k, v in rewards.items()
                }
        finally:
            self._scorer.to(torch.device("cpu"))
            torch.cuda.empty_cache()

        # Concat all batches
        tensor_keys = ["prompt_embeds", "pooled_prompt_embeds", "timesteps",
                       "latents", "next_latents", "log_probs"]
        policy_keys = [k for k in ("noise_preds", "prev_means", "std_devs")
                       if k in samples_list[0]]
        samples = {k: torch.cat([s[k] for s in samples_list], dim=0)
                   for k in tensor_keys + policy_keys}

        reward_keys = list(samples_list[0]["rewards"].keys())
        samples["rewards"] = {
            rk: torch.cat([s["rewards"][rk] for s in samples_list], dim=0)
            for rk in reward_keys
        }

        return samples, all_prompts_local, all_images_np

    # ------------------------------------------------------------------
    # GRPO Stage 2: Advantage computation
    # ------------------------------------------------------------------

    def _grpo_compute_advantages(
        self,
        grpo_cfg,
        samples: dict,
        all_prompts_local: list[str],
        all_images_np: list,
        num_train_timesteps: int,
    ) -> dict[str, float]:
        """Cross-rank advantage computation, image logging, and zero-advantage filtering.

        Gathers rewards and prompts across all ranks, computes per-prompt normalized
        advantages, logs K-repeat group images, filters zero-advantage samples, and
        pads to ensure divisibility by ``num_batches_per_epoch``.

        Modifies ``samples`` in-place: adds ``advantages`` key, removes ``rewards``
        key, and filters out zero-advantage rows from all tensor keys.

        Args:
            grpo_cfg: GRPOConfig instance.
            samples: Dict of CPU tensors from sampling stage (modified in-place).
            all_prompts_local: Flat prompt list on this rank.
            all_images_np: uint8 numpy images for group logging (may be empty).
            num_train_timesteps: Timestep dimension for advantage expansion.

        Returns:
            epoch_metrics: Dict with ``reward_{key}`` (global mean per scorer)
                and ``actual_batch_size`` (filtered batch size per sub-batch).
        """
        device = torch.device("cuda")
        reward_keys = list(samples["rewards"].keys())

        # Expand rewards to timestep dimension
        avg_key = "mean" if "mean" in samples["rewards"] else reward_keys[0]
        samples["rewards"]["ori_mean"] = samples["rewards"][avg_key].clone()
        samples["rewards"]["mean"] = samples["rewards"][avg_key].unsqueeze(1).repeat(1, num_train_timesteps)

        # Gather rewards across ranks
        gathered_rewards = {}
        for k, v in samples["rewards"].items():
            if dist.is_initialized() and get_world_size() > 1:
                v_gpu = v.to(device)
                gathered_list = [torch.zeros_like(v_gpu) for _ in range(get_world_size())]
                dist.all_gather(gathered_list, v_gpu)
                gathered_rewards[k] = torch.cat(gathered_list, dim=0).cpu().numpy()
            else:
                gathered_rewards[k] = v.cpu().numpy()

        # Gather prompts across ranks
        if dist.is_initialized() and get_world_size() > 1:
            all_prompts_gathered = [None] * get_world_size()
            dist.all_gather_object(all_prompts_gathered, all_prompts_local)
            all_prompts_global = [p for rank_prompts in all_prompts_gathered for p in rank_prompts]
        else:
            all_prompts_global = all_prompts_local

        # Epoch metrics: reward stats
        # "mean"/"ori_mean" are internal aggregate keys — skip ori_mean entirely
        # (identical to mean), only log mean once. Per-scorer keys get full stats.
        epoch_metrics = {}
        skip_keys = {"ori_mean", "ori_avg", "_accuracy"}
        for k, v in gathered_rewards.items():
            if any(s in k for s in skip_keys):
                continue
            epoch_metrics[f"reward/{k}"] = float(v.mean())
            if k != "mean":
                epoch_metrics[f"reward/{k}_std"] = float(v.std())
                epoch_metrics[f"reward/{k}_max"] = float(v.max())
                epoch_metrics[f"reward/{k}_min"] = float(v.min())
        epoch_metrics["num_samples"] = float(len(all_prompts_global))

        # Per-prompt stat tracking → advantages
        if grpo_cfg.weight_advantages:
            reward_weights = grpo_cfg.reward_fn
            per_reward_advantages = {}
            for rk, w in reward_weights.items():
                if rk not in gathered_rewards:
                    continue
                rk_rewards = gathered_rewards[rk]
                if rk_rewards.ndim == 1:
                    rk_rewards = np.tile(rk_rewards[:, None], (1, num_train_timesteps))
                if grpo_cfg.per_prompt_stat_tracking:
                    tmp_tracker = PerPromptStatTracker(global_std=grpo_cfg.per_prompt_global_std)
                    rk_adv = tmp_tracker.update(all_prompts_global, rk_rewards)
                else:
                    rk_adv = (rk_rewards - rk_rewards.mean()) / (rk_rewards.std() + 1e-4)
                per_reward_advantages[rk] = (rk_adv, w)

            advantages = sum(w * adv for adv, w in per_reward_advantages.values())
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-4)

            # Update main tracker for group_size / trained_prompt_num logging
            if grpo_cfg.per_prompt_stat_tracking:
                self._stat_tracker.update(all_prompts_global, gathered_rewards["mean"])
                self._stat_tracker.clear()
        elif grpo_cfg.per_prompt_stat_tracking:
            advantages = self._stat_tracker.update(all_prompts_global, gathered_rewards["mean"])
            self._stat_tracker.clear()
        else:
            avg = gathered_rewards["mean"]
            advantages = (avg - avg.mean()) / (avg.std() + 1e-4)

        advantages = torch.as_tensor(advantages, dtype=torch.float32)
        epoch_metrics["advantage_mean"] = float(advantages.mean())
        epoch_metrics["advantage_std"] = float(advantages.std())
        epoch_metrics["advantage_max"] = float(advantages.max())
        epoch_metrics["advantage_min"] = float(advantages.min())
        epoch_metrics["advantage_nonzero_frac"] = float((advantages.abs().sum(dim=-1) != 0).float().mean())

        # Per-prompt group stats (from original grpo)
        if grpo_cfg.per_prompt_stat_tracking:
            group_size, trained_prompt_num = self._stat_tracker.get_stats()
            epoch_metrics["group_size"] = float(group_size)
            epoch_metrics["trained_prompt_num"] = float(trained_prompt_num)

        # Zero-std ratio: fraction of unique prompts whose K-repeat rewards
        # have zero variance (indicates reward model is not discriminating)
        prompts_arr = np.array(all_prompts_global)
        unique_prompts_arr = np.unique(prompts_arr)
        reward_key = "mean" if "mean" in gathered_rewards else list(gathered_rewards.keys())[0]
        group_stds = []
        for p in unique_prompts_arr:
            group_rewards = gathered_rewards[reward_key][prompts_arr == p]
            if len(group_rewards) > 1:
                group_stds.append(float(np.std(group_rewards)))
        if group_stds:
            group_stds_arr = np.array(group_stds)
            epoch_metrics["zero_std_ratio"] = float(np.mean(group_stds_arr == 0))
            epoch_metrics["reward_std_mean"] = float(group_stds_arr.mean())
            epoch_metrics["reward_std_max"] = float(group_stds_arr.max())
            epoch_metrics["reward_std_min"] = float(group_stds_arr.min())
        epoch_metrics["num_unique_prompts"] = float(len(unique_prompts_arr))

        # Ungather to local rank
        if dist.is_initialized() and get_world_size() > 1:
            advantages = (
                advantages.reshape(get_world_size(), -1, advantages.shape[-1])
                [get_rank()]
            )
        samples["advantages"] = advantages

        # --- Log K-repeat group images ---
        self._log_grpo_group_images(grpo_cfg, all_prompts_local, all_images_np)
        del all_images_np

        # --- Zero advantage filtering ---
        tensor_keys = ["prompt_embeds", "pooled_prompt_embeds", "timesteps",
                       "latents", "next_latents", "log_probs"]
        policy_filter_keys = [k for k in ("noise_preds", "prev_means", "std_devs")
                              if k in samples]
        mask = samples["advantages"].abs().sum(dim=1) != 0
        num_batches = grpo_cfg.num_batches_per_epoch
        true_count = int(mask.sum().item())

        # Avoid empty-batch training when rewards collapse to identical values.
        # Keep one full local sub-batch so downstream PPO/aux paths don't run with B=0.
        if true_count == 0 and mask.numel() > 0:
            keep_count = min(int(mask.numel()), int(num_batches))
            _rng = torch.Generator()
            _rng.manual_seed(int(self._grpo_epoch) * 1_000_003 + int(get_rank()))
            keep_idx = torch.randperm(int(mask.numel()), generator=_rng)[:keep_count]
            mask[keep_idx] = True
            true_count = int(mask.sum().item())

        if true_count % num_batches != 0:
            false_indices = torch.where(~mask)[0]
            num_to_change = int(num_batches - (true_count % num_batches))
            if len(false_indices) >= num_to_change:
                _rng = torch.Generator()
                _rng.manual_seed(int(self._grpo_epoch) * 1_000_003 + int(get_rank()) + 17)
                random_indices = torch.randperm(len(false_indices), generator=_rng)[:num_to_change]
                mask[false_indices[random_indices]] = True
                true_count = int(mask.sum().item())

        epoch_metrics["num_samples_before_filter"] = float(mask.numel())
        epoch_metrics["num_samples_after_filter"] = float(true_count)
        epoch_metrics["num_samples_dropped"] = float(mask.numel() - true_count)
        epoch_metrics["actual_batch_size"] = float(true_count // num_batches)

        # Filter all rank-local tensor fields aligned on sample dimension.
        # This keeps optional trainer-specific tensor keys (e.g. extra token ids)
        # in sync with the zero-advantage mask without requiring parent edits.
        filter_keys = []
        for k, v in samples.items():
            if k == "rewards":
                continue
            if isinstance(v, torch.Tensor) and v.ndim > 0 and v.shape[0] == mask.shape[0]:
                filter_keys.append(k)

        required_keys = set(tensor_keys + policy_filter_keys + ["advantages"])
        missing_required = [k for k in required_keys if k not in filter_keys]
        if missing_required:
            raise RuntimeError(
                f"Zero-advantage filtering missing required sample keys: {missing_required}"
            )

        for k in filter_keys:
            samples[k] = samples[k][mask]
        del samples["rewards"]

        return epoch_metrics

    # ------------------------------------------------------------------
    # GRPO Stage 3: PPO training
    # ------------------------------------------------------------------

    def _log_grpo_group_images(
        self,
        grpo_cfg,
        all_prompts_local: list[str],
        all_images_np: list,
    ) -> None:
        """Log random K-repeat group images to wandb.

        Selects ``log_num_groups`` random unique prompts (same selection across all
        ranks via broadcast), gathers each prompt's K images across ranks, and logs
        them as grouped wandb images at ``media/grpo_group_{k}``.

        Args:
            grpo_cfg: GRPOConfig instance (reads ``log_num_groups``).
            all_prompts_local: Flat prompt list on this rank.
            all_images_np: uint8 numpy images on this rank (same order as prompts).
        """
        log_k = grpo_cfg.log_num_groups
        if log_k <= 0 or not self.wandb_logger or not all_images_np:
            return

        if dist.is_initialized() and get_world_size() > 1:
            _all_unique_gathered: list = [None] * get_world_size()
            unique_prompts = list(dict.fromkeys(all_prompts_local))
            dist.all_gather_object(_all_unique_gathered, unique_prompts)
            global_unique_prompts = list(dict.fromkeys(
                p for rank_prompts in _all_unique_gathered for p in rank_prompts
            ))
        else:
            global_unique_prompts = list(dict.fromkeys(all_prompts_local))

        _K_log = min(log_k, len(global_unique_prompts))

        if is_main_process():
            _rng = random.Random(self._grpo_global_step)
            selected_prompts = _rng.sample(global_unique_prompts, _K_log)
        else:
            selected_prompts = [None] * _K_log
        if dist.is_initialized() and get_world_size() > 1:
            dist.broadcast_object_list(selected_prompts, src=0)

        _local_group_images: dict[str, list] = {}
        for _prompt in selected_prompts:
            _indices = [i for i, p in enumerate(all_prompts_local) if p == _prompt]
            _local_group_images[_prompt] = [all_images_np[idx] for idx in _indices]

        if dist.is_initialized() and get_world_size() > 1:
            _gathered: list = [None] * get_world_size()
            dist.all_gather_object(_gathered, _local_group_images)
        else:
            _gathered = [_local_group_images]

        if is_main_process():
            from PIL import Image
            for _k, _prompt in enumerate(selected_prompts):
                _all_np = []
                for _rank_data in _gathered:
                    _all_np.extend(_rank_data.get(_prompt, []))
                if _all_np:
                    _pil_imgs = [Image.fromarray(arr) for arr in _all_np]
                    _captions = [_prompt] * len(_pil_imgs)
                    self.wandb_logger.log_images(
                        images=_pil_imgs, captions=_captions,
                        step=self._grpo_global_step, key=f"media/grpo_group_{_k}",
                    )

    # ------------------------------------------------------------------
    # Pipeline with log-probability
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _pipeline_with_logprob(
        self,
        prompt_embeds: torch.Tensor,
        pooled_embeds: torch.Tensor,
        neg_embeds: torch.Tensor,
        neg_pooled: torch.Tensor,
        grpo_cfg,
        generator: list[torch.Generator] | None = None,
        prompts: list[str] | None = None,
        selected_train_steps: set[int] | None = None,
        epoch_seed: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """SDE/DMD multi-step sampling with log-probability tracking.

        Supports two sigma schedule modes via ``grpo_cfg.sampling_mode``:
        - ``"sde"``: evenly spaced timesteps from ``set_timesteps(num_steps)``.
        - ``"dmd"``: custom sigmas from ``denoising_step_list`` via backend-specific
          shifted-sigma computation.

        When ``selected_train_steps`` is provided (step selection mode), pipeline
        steps NOT in the set use prompt-deterministic shared CPS noise. This
        ensures the same prompt gets identical noise across K-repeat group members
        (even across different ranks), reducing advantage variance from non-trained
        steps.

        Args:
            prompt_embeds: Text embeddings [B, seq_len, dim] on GPU.
            pooled_embeds: Pooled CLIP embeddings [B, dim] on GPU.
            neg_embeds: Unconditional text embeddings [B, seq_len, dim].
            neg_pooled: Unconditional pooled embeddings [B, dim].
            grpo_cfg: Config-like object with ``num_steps``, ``sampling_mode``,
                ``denoising_step_list``, ``guidance_scale``, ``noise_level``,
                ``sde_type``, ``vae_decode_batch_size``, ``same_latent``.
            generator: Per-sample torch.Generators for deterministic noise (or None).
            prompts: Original prompt texts [B] for deterministic noise seeding.
                Required when ``selected_train_steps`` is not None.
            selected_train_steps: Set of pipeline step indices selected for training.
                Steps not in this set use prompt-seeded shared noise.
            epoch_seed: Seed offset for deterministic noise (varies per epoch).

        Returns:
            images: VAE-decoded images [B, C, H, W] in [0, 1] on GPU.
            all_latents: Stacked latents at each step [B, T+1, C, H, W] on GPU.
            all_log_probs: Log-probabilities at each step [B, T] on GPU.
        """
        B = prompt_embeds.shape[0]
        device = prompt_embeds.device

        # Initialize noise
        if generator is not None and len(generator) == B:
            # Per-sample deterministic generators
            latent_list = [
                torch.randn(
                    1, self.latent_channels, self.latent_size, self.latent_size,
                    device=device, dtype=torch.float32, generator=g,
                )
                for g in generator
            ]
            latents = torch.cat(latent_list, dim=0)
        else:
            latents = torch.randn(
                B, self.latent_channels, self.latent_size, self.latent_size,
                device=device, dtype=torch.float32,
            )

        # Set SDE scheduler timesteps
        if getattr(grpo_cfg, "sampling_mode", "sde") == "dmd":
            sigmas = self._get_denoising_sigmas(grpo_cfg.denoising_step_list)
            sigmas_tensor = torch.tensor(sigmas + [0.0], dtype=torch.float32)
            self.sde_scheduler.timesteps = (
                sigmas_tensor[:-1] * self._num_train_timesteps()
            ).to(device=device)
            self.sde_scheduler.sigmas = sigmas_tensor.to(device=device)
            self.sde_scheduler.num_inference_steps = len(sigmas)
            self.sde_scheduler._step_index = None
        else:
            # Dynamic-shifted schedulers (FLUX.1 / FLUX.2) need mu for
            # FlowMatchEulerDiscreteScheduler.set_timesteps; otherwise it
            # raises "mu must be passed when use_dynamic_shifting is True".
            mu: float | None = None
            if bool(getattr(self.sde_scheduler.config, "use_dynamic_shifting", False)):
                if self._is_flux2_klein:
                    mu = compute_empirical_mu(
                        image_seq_len=self.latent_size * self.latent_size,
                        num_steps=int(grpo_cfg.num_steps),
                    )
                elif self._is_flux:
                    # FLUX.1: linear mu in image_seq_len = (latent_H/2)^2 packed tokens.
                    flux_image_seq_len = (self.latent_size // 2) * (self.latent_size // 2)
                    mu = compute_mu_flux(
                        image_seq_len=flux_image_seq_len,
                        base_image_seq_len=int(getattr(
                            self.sde_scheduler.config, "base_image_seq_len", 256,
                        )),
                        max_image_seq_len=int(getattr(
                            self.sde_scheduler.config, "max_image_seq_len", 4096,
                        )),
                        base_shift=float(getattr(
                            self.sde_scheduler.config, "base_shift", 0.5,
                        )),
                        max_shift=float(getattr(
                            self.sde_scheduler.config, "max_shift", 1.15,
                        )),
                    )

            if mu is not None:
                self.sde_scheduler.set_timesteps(
                    num_inference_steps=grpo_cfg.num_steps,
                    device=device,
                    mu=mu,
                )
            else:
                self.sde_scheduler.set_timesteps(
                    num_inference_steps=grpo_cfg.num_steps,
                    device=device,
                )

        # Prepare CFG embeddings
        do_cfg = grpo_cfg.guidance_scale > 1.0
        if do_cfg:
            all_embeds = torch.cat([neg_embeds[:B], prompt_embeds], dim=0)
            all_pooled = torch.cat([neg_pooled[:B], pooled_embeds], dim=0)

        all_latents = [latents]
        all_log_probs = []

        store_policy = self._store_policy_outputs
        all_noise_preds = [] if store_policy else None
        all_prev_means = [] if store_policy else None
        all_std_devs = [] if store_policy else None

        for i, t in enumerate(self.sde_scheduler.timesteps):
            # Forward pass
            if do_cfg:
                latent_input = torch.cat([latents] * 2, dim=0)
                timestep = t.expand(latent_input.shape[0])
                noise_pred = self._forward_transformer(
                    self.generator, latent_input, timestep, all_embeds, all_pooled,
                )
                uncond, cond = noise_pred.chunk(2)
                noise_pred = uncond + grpo_cfg.guidance_scale * (cond - uncond)
            else:
                timestep = t.expand(B)
                noise_pred = self._forward_transformer(
                    self.generator, latents, timestep, prompt_embeds, pooled_embeds,
                )

            latents, log_prob, prev_mean, std_dev = sde_step_with_logprob(
                self.sde_scheduler,
                noise_pred.float(),
                t.unsqueeze(0),
                latents.float(),
                noise_level=grpo_cfg.noise_level,
                sde_type=grpo_cfg.sde_type,
                variance_noise=self._make_shared_noise(
                    i, selected_train_steps, prompts, epoch_seed,
                    latents.shape, latents.dtype, device,
                ),
            )

            all_latents.append(latents)
            all_log_probs.append(log_prob)

            if store_policy:
                all_noise_preds.append(noise_pred.detach())
                all_prev_means.append(prev_mean.detach())
                all_std_devs.append(std_dev.detach().expand(B, -1, -1, -1))

        # VAE decode
        decode_latents = latents
        batch_images = []
        for start in range(0, B, grpo_cfg.vae_decode_batch_size):
            end = min(start + grpo_cfg.vae_decode_batch_size, B)
            imgs = self._decode_latents_to_tensor(decode_latents[start:end])
            batch_images.append(imgs)
        images = torch.cat(batch_images, dim=0)

        policy_outputs = None
        if store_policy:
            policy_outputs = {
                "noise_preds": torch.stack(all_noise_preds, dim=1),   # [B, T, C, H, W]
                "prev_means": torch.stack(all_prev_means, dim=1),     # [B, T, C, H, W]
                "std_devs": torch.stack(all_std_devs, dim=1),         # [B, T, 1, 1, 1]
            }

        return (
            images,
            torch.stack(all_latents, dim=1),    # [B, T+1, C, H, W]
            torch.stack(all_log_probs, dim=1),  # [B, T]
            policy_outputs,
        )

    # ------------------------------------------------------------------
    # Shared noise helper for step selection
    # ------------------------------------------------------------------

    @staticmethod
    def _make_shared_noise(
        step_idx: int,
        selected_train_steps: set[int] | None,
        prompts: list[str] | None,
        epoch_seed: int,
        latent_shape: torch.Size,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor | None:
        """Generate prompt-deterministic shared noise for non-trained steps.

        For step selection mode, pipeline steps NOT selected for training use
        noise seeded by ``SHA256(prompt) + step_idx + epoch_seed``. This ensures
        the same prompt gets identical CPS noise across all K-repeat group
        members, even when they are scattered across different ranks by the
        ``DistributedKRepeatSampler``.

        Uses CPU generators for cross-device reproducibility (CUDA generators
        are not guaranteed to be reproducible across different devices).

        Args:
            step_idx: Current pipeline step index.
            selected_train_steps: Set of step indices selected for training,
                or None if step selection is disabled.
            prompts: Original prompt texts [B].
            epoch_seed: Epoch-level seed offset.
            latent_shape: Shape of the latent tensor [B, C, H, W].
            dtype: Desired noise dtype.
            device: Target device.

        Returns:
            Deterministic noise [B, C, H, W] on ``device``, or None if this
            step is selected for training or step selection is disabled.
        """
        if selected_train_steps is None or step_idx in selected_train_steps:
            return None
        if prompts is None:
            return None

        B = latent_shape[0]
        single_shape = latent_shape[1:]  # (C, H, W)
        noise_list = []
        for sample_idx in range(B):
            prompt = prompts[sample_idx]
            digest = hashlib.sha256(prompt.encode()).digest()
            prompt_hash = int.from_bytes(digest[:4], "big")
            seed = (prompt_hash + step_idx * 10007 + epoch_seed) % (2 ** 31)
            gen = torch.Generator().manual_seed(seed)  # CPU generator
            noise_list.append(
                torch.randn(single_shape, dtype=dtype, generator=gen)
            )
        return torch.stack(noise_list, dim=0).to(device)

    # ------------------------------------------------------------------
    # Forward through transformer (direct call, not predict_noise_sd35)
    # ------------------------------------------------------------------

    def _forward_transformer(
        self,
        model: nn.Module,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        pooled_projections: torch.Tensor,
    ) -> torch.Tensor:
        """Direct transformer forward pass (no internal CFG handling).

        The caller must concatenate latents/embeds for CFG and split the output.
        This is required because grpo's CFG concatenates the latent input
        (not just embeddings), unlike backend helper CFG wrappers.

        Args:
            model: Transformer model (may be FSDP/DDP wrapped, callable directly).
            latents: Input hidden states [B, C, H, W] (or [2B, ...] for CFG).
            timestep: Timestep tensor [B] (or [2B] for CFG).
            encoder_hidden_states: Text embeddings [B, seq, dim] (or [2B, ...] for CFG).
            pooled_projections: Pooled embeddings [B, dim] (or [2B, ...] for CFG).

        Returns:
            Velocity prediction [B, C, H, W] (or [2B, ...] for CFG).
        """
        # FLUX.1 and FLUX.2 transformers don't accept the SD3 keyword-style
        # forward (no separate pooled_projections / hidden_states API). Route
        # them through _predict_noise which handles packing/img_ids/txt_ids
        # and (for FLUX.1-dev) the distilled-guidance scalar input.
        if self._is_flux2_klein or self._is_flux:
            return self._predict_noise(
                model=model,
                noisy_latents=latents,
                prompt_embeds=encoder_hidden_states,
                timesteps=timestep,
                pooled_embeds=pooled_projections,
                guidance_scale=1.0,
            )

        return model(
            hidden_states=latents,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            pooled_projections=pooled_projections,
            return_dict=False,
        )[0]

    # ------------------------------------------------------------------
    # Reference model predictions for KL
    # ------------------------------------------------------------------

    def _compute_reference_mean_lora(
        self, batch, j, embeds, pooled, grpo_cfg, autocast_ctx,
    ) -> torch.Tensor:
        """Get reference model mean prediction via disable_adapter() (LoRA mode).

        Temporarily disables LoRA adapters on the generator to forward-pass with
        base model weights, then computes the SDE transition mean ``prev_mean_ref``.

        Args:
            batch: Training batch dict with ``latents``, ``timesteps``, ``next_latents``.
            j: Timestep index within the trajectory.
            embeds: Text embeddings (possibly CFG-concatenated [2B, seq, dim]).
            pooled: Pooled embeddings (possibly CFG-concatenated [2B, dim]).
            grpo_cfg: GRPOConfig instance.
            autocast_ctx: Autocast context manager.

        Returns:
            prev_mean_ref: Deterministic mean of the SDE transition [B, C, H, W].
        """
        model = self.generator
        if isinstance(model, (FSDP, DDP)):
            underlying = model.module
        else:
            underlying = model

        underlying.disable_adapters()
        try:
            with autocast_ctx():
                if grpo_cfg.cfg_in_training:
                    noise_pred_ref = self._forward_transformer(
                        model,
                        torch.cat([batch["latents"][:, j]] * 2, dim=0),
                        torch.cat([batch["timesteps"][:, j]] * 2, dim=0),
                        embeds, pooled,
                    )
                    uncond, cond = noise_pred_ref.chunk(2)
                    noise_pred_ref = uncond + grpo_cfg.guidance_scale * (cond - uncond)
                else:
                    noise_pred_ref = self._forward_transformer(
                        model, batch["latents"][:, j],
                        batch["timesteps"][:, j], embeds, pooled,
                    )
        finally:
            underlying.enable_adapters()

        _, _, prev_mean_ref, _ = sde_step_with_logprob(
            self.sde_scheduler,
            noise_pred_ref.float(),
            batch["timesteps"][:, j],
            batch["latents"][:, j].float(),
            prev_sample=batch["next_latents"][:, j].float(),
            noise_level=grpo_cfg.noise_level,
            sde_type=grpo_cfg.sde_type,
        )
        return prev_mean_ref

    def _compute_reference_mean_fullweight(
        self, batch, j, embeds, pooled, grpo_cfg, autocast_ctx,
    ) -> torch.Tensor:
        """Get reference model mean prediction using frozen reference_model.

        Forwards through ``self.reference_model`` (frozen base transformer deepcopy)
        and computes the SDE transition mean ``prev_mean_ref``.

        Args:
            batch: Training batch dict with ``latents``, ``timesteps``, ``next_latents``.
            j: Timestep index within the trajectory.
            embeds: Text embeddings (possibly CFG-concatenated [2B, seq, dim]).
            pooled: Pooled embeddings (possibly CFG-concatenated [2B, dim]).
            grpo_cfg: GRPOConfig instance.
            autocast_ctx: Autocast context manager.

        Returns:
            prev_mean_ref: Deterministic mean of the SDE transition [B, C, H, W].
        """
        with autocast_ctx():
            if grpo_cfg.cfg_in_training:
                noise_pred_ref = self._forward_transformer(
                    self.reference_model,
                    torch.cat([batch["latents"][:, j]] * 2, dim=0),
                    torch.cat([batch["timesteps"][:, j]] * 2, dim=0),
                    embeds, pooled,
                )
                uncond, cond = noise_pred_ref.chunk(2)
                noise_pred_ref = uncond + grpo_cfg.guidance_scale * (cond - uncond)
            else:
                noise_pred_ref = self._forward_transformer(
                    self.reference_model, batch["latents"][:, j],
                    batch["timesteps"][:, j], embeds, pooled,
                )

            _, _, prev_mean_ref, _ = sde_step_with_logprob(
                self.sde_scheduler,
                noise_pred_ref.float(),
                batch["timesteps"][:, j],
                batch["latents"][:, j].float(),
                prev_sample=batch["next_latents"][:, j].float(),
                noise_level=grpo_cfg.noise_level,
                sde_type=grpo_cfg.sde_type,
            )
        return prev_mean_ref

    # ------------------------------------------------------------------
    # Reward computation
    # ------------------------------------------------------------------

    def _init_scorer(self) -> None:
        """Initialize reward scorer."""
        grpo_cfg = self.config.grpo
        if self._scorer is not None:
            return
        reward_names = set(grpo_cfg.reward_fn.keys())
        if "geneval" in reward_names and not self._grpo_prompt_returns_metadata:
            raise ValueError(
                "grpo.reward_fn contains 'geneval', but prompt batches are not "
                "metadata-enabled. Use a metadata jsonl prompt source "
                "(e.g., dataset/geneval/train_metadata.jsonl) or switch training reward "
                "to metadata-free metrics (e.g., geneval2, hpsv2, clipscore)."
            )
        from rtdmd.rewards.reward_ckpt_path import set_ckpt_path
        if grpo_cfg.reward_ckpt_path:
            set_ckpt_path(grpo_cfg.reward_ckpt_path)
        from rtdmd.rewards.multi_scorer import MultiScorer
        self._scorer = MultiScorer(
            device=torch.device("cpu"),
            score_dict=grpo_cfg.reward_fn,
        )
        logger.info(f"  GRPO scorer initialized: {list(grpo_cfg.reward_fn.keys())}")

    def _compute_rewards(
        self,
        images: torch.Tensor,
        prompts: list[str],
        metadata: list[dict] | None = None,
    ) -> tuple[dict, Any]:
        """Compute rewards for generated images."""
        score_details, scorer_metadata = self._scorer(
            images,
            prompts,
            metadata=metadata,
        )
        return score_details, scorer_metadata

    # ------------------------------------------------------------------
    # Text encoding
    # ------------------------------------------------------------------

    def _encode_prompts(
        self, prompts: list[str],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode text prompts using active backend text encoder stack."""
        device = torch.device("cuda")
        if self._is_flux2_klein:
            return encode_prompts_flux2_klein(
                prompts=prompts,
                text_encoder=self.text_encoders[0],
                tokenizer=self.tokenizers[0],
                device=device,
            )
        if self._is_flux:
            return encode_prompts_flux(
                prompts=prompts,
                text_encoders=self.text_encoders,
                tokenizers=self.tokenizers,
                device=device,
            )
        return encode_prompts_sd35(
            prompts, self.text_encoders, self.tokenizers, device,
        )

    @torch.no_grad()
    def _get_uncond_embeds(
        self, batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Get unconditional embeddings for backend conditioning / CFG."""
        if self._uncond_embeds is None:
            device = torch.device("cuda")
            if self._is_flux2_klein:
                self._uncond_embeds = encode_prompts_flux2_klein(
                    prompts=[""],
                    text_encoder=self.text_encoders[0],
                    tokenizer=self.tokenizers[0],
                    device=device,
                )
            elif self._is_flux:
                self._uncond_embeds = encode_prompts_flux(
                    prompts=[""],
                    text_encoders=self.text_encoders,
                    tokenizers=self.tokenizers,
                    device=device,
                )
            else:
                self._uncond_embeds = encode_prompts_sd35(
                    [""], self.text_encoders, self.tokenizers, device,
                )
        uncond_embeds, uncond_pooled = self._uncond_embeds
        if self._is_flux2_klein:
            # FLUX.2 stores text_ids in the "pooled" slot, shape [1, L, 4].
            return (
                uncond_embeds.expand(batch_size, -1, -1),
                uncond_pooled.expand(batch_size, -1, -1),
            )
        # SD3.5 and FLUX.1 share the same shape convention: pooled is [1, D].
        return (
            uncond_embeds.expand(batch_size, -1, -1),
            uncond_pooled.expand(batch_size, -1),
        )

    @torch.no_grad()
    def _decode_latents_to_tensor(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latent tensors to [0,1] image tensors, including Flux2 layout."""
        if not self._is_flux2_klein:
            return super()._decode_latents_to_tensor(latents)

        device = latents.device
        if next(self.vae.parameters()).device != device:
            self.vae.to(device)

        latents_bn_mean = self.vae.bn.running_mean.view(1, -1, 1, 1).to(
            latents.device,
            latents.dtype,
        )
        latents_bn_std = torch.sqrt(
            self.vae.bn.running_var.view(1, -1, 1, 1) + self.vae.config.batch_norm_eps
        ).to(latents.device, latents.dtype)
        latents = latents * latents_bn_std + latents_bn_mean
        latents = unpatchify_flux2_latents(latents)

        images = self.vae.decode(latents.to(self.vae.dtype), return_dict=False)[0]
        return (images / 2 + 0.5).clamp(0, 1)

    # ------------------------------------------------------------------
    # Evaluation override
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _evaluate_grpo(self) -> None:
        """Evaluate generator using ODE sampling (noise_level=0).

        Called at epoch boundaries. Uses eval_num_steps and eval_guidance_scale.
        If EMA is enabled, switches to EMA weights for evaluation.
        """
        grpo_cfg = self.config.grpo
        eval_cfg = self.config.eval

        if not eval_cfg.enabled:
            return

        logger.info(f"[Eval] Starting evaluation at epoch {self._grpo_epoch}")

        # Save training RNG
        rng_state = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.random.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state(),
        }

        try:
            # Switch to EMA weights
            if self._ema is not None:
                self._ema.copy_ema_to(self._trainable_params, store_temp=True)

            # Keep legacy ordering for default eval behavior:
            # - default (no reward_dataset_map): seed first, lazy init later
            # - per-reward datasets enabled: preload before seed for deterministic RNG
            reward_dataset_map = getattr(eval_cfg, "reward_dataset_map", {}) or {}
            if reward_dataset_map:
                self._warmup_eval_reward_resources(scorer_attr="_eval_scorer_grpo")

            # Set eval seed
            eval_seed = eval_cfg.eval_seed + get_rank()
            random.seed(eval_seed)
            np.random.seed(eval_seed)
            torch.manual_seed(eval_seed)
            torch.cuda.manual_seed_all(eval_seed)

            self.generator.eval()

            from types import SimpleNamespace
            eval_grpo = SimpleNamespace(
                sampling_mode=grpo_cfg.sampling_mode,
                denoising_step_list=grpo_cfg.denoising_step_list,
                num_steps=grpo_cfg.eval_num_steps,
                guidance_scale=grpo_cfg.eval_guidance_scale,
                noise_level=grpo_cfg.eval_noise_level,
                sde_type=grpo_cfg.eval_sde_type,
                vae_decode_batch_size=grpo_cfg.vae_decode_batch_size,
                same_latent=False,
            )

            def _batch_generate(batch_prompts: list[str]) -> torch.Tensor:
                prompt_embeds, pooled_embeds = self._encode_prompts(batch_prompts)
                neg_embeds, neg_pooled = self._get_uncond_embeds(len(batch_prompts))

                with self._autocast():
                    images, _, _, _ = self._pipeline_with_logprob(
                        prompt_embeds, pooled_embeds,
                        neg_embeds, neg_pooled,
                        eval_grpo,
                    )
                return images

            eval_metrics, all_images_tensor, all_prompts_local = self._run_eval_for_reward_sets(
                batch_size=grpo_cfg.test_batch_size,
                batch_generate_fn=_batch_generate,
                scorer_attr="_eval_scorer_grpo",
            )

            self._offload_eval_scorers(scorer_attr="_eval_scorer_grpo")

            if self.wandb_logger:
                self.wandb_logger.log(eval_metrics, step=self._grpo_global_step, section="eval")

                self._log_eval_media_groups(
                    step=self._grpo_global_step,
                    base_key="media/eval",
                    fallback_images=all_images_tensor,
                    fallback_prompts=all_prompts_local,
                )

            if is_main_process():
                parts = [f"[Eval] epoch={self._grpo_epoch}"]
                for k, v in eval_metrics.items():
                    parts.append(f"{k}={v:.4g}")
                logger.info(" | ".join(parts))

        finally:
            # Restore EMA
            if self._ema is not None:
                self._ema.copy_temp_to(self._trainable_params)

            # Restore RNG
            random.setstate(rng_state["python"])
            np.random.set_state(rng_state["numpy"])
            torch.random.set_rng_state(rng_state["torch_cpu"])
            torch.cuda.set_rng_state(rng_state["torch_cuda"])

            if dist.is_initialized():
                barrier()

    # ------------------------------------------------------------------
    # Gradient clipping
    # ------------------------------------------------------------------

    @staticmethod
    def _clip_grad_norm(model: nn.Module, max_norm: float) -> None:
        """FSDP-aware gradient clipping."""
        if isinstance(model, FSDP):
            model.clip_grad_norm_(max_norm)
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

    # ------------------------------------------------------------------
    # Checkpoint hooks
    # ------------------------------------------------------------------

    def _get_extra_checkpoint_state(self) -> dict:
        """Add GRPO-specific state to meta.pt."""
        state = {
            "grpo_epoch": self._grpo_epoch,
            "grpo_global_step": self._grpo_global_step,
        }
        if self._ema is not None:
            state["ema_state"] = self._ema.state_dict()
        return state

    def _restore_extra_checkpoint_state(self, extra: dict) -> None:
        """Restore GRPO-specific state from meta.pt."""
        self._grpo_epoch = extra.get("grpo_epoch", 0)
        self._grpo_global_step = extra.get("grpo_global_step", 0)
        self.global_step = self._grpo_global_step

        # Restore EMA
        if self._ema is not None and "ema_state" in extra:
            self._ema.load_state_dict(extra["ema_state"])

        # Reset sampler for deterministic resume
        if self._grpo_sampler is not None:
            self._grpo_sampler.set_epoch(
                self._grpo_epoch * self.config.grpo.num_batches_per_epoch
            )
            self._grpo_sample_iter = iter(self._grpo_dataloader)

        logger.info(
            f"  GRPO state restored: epoch={self._grpo_epoch}, "
            f"global_step={self._grpo_global_step}"
        )

    def _save_final_transformer(self) -> None:
        """Save the final generator (EMA when available) to ``transformer/``.

        Mirrors DMD trainer's final layout: the inference-ready
        ``{output_dir}/transformer/`` folder contains EMA weights when EMA is
        enabled, and falls back to training weights otherwise.

        Format:
        - LoRA mode: PEFT (``adapter_config.json`` + ``adapter_model.safetensors``).
        - Full-weight mode: diffusers (``save_pretrained``).
        """
        if self._ema is None:
            return super()._save_final_transformer()

        from rtdmd.utils.checkpoint import (
            _gather_fsdp_state_dict,
            _save_diffusers_transformer,
            _resolve_lora_config,
            save_lora_peft_format,
        )

        lora_models = self._get_lora_model_names()
        lora_configs = self._get_lora_configs()
        generator = self.models["generator"]

        # Swap EMA weights into the FSDP model, gather to rank 0, restore.
        self._ema.copy_ema_to(self._trainable_params, store_temp=True)
        try:
            state_dict = _gather_fsdp_state_dict(generator)
            if is_main_process():
                transformer_dir = os.path.join(self.config.output_dir, "transformer")
                if "generator" in lora_models:
                    # _resolve_lora_config aliases generator_ema -> generator
                    lora_cfg = _resolve_lora_config("generator_ema", lora_configs)
                    if lora_cfg is not None:
                        save_lora_peft_format(state_dict, lora_cfg, transformer_dir)
                        logger.info(
                            f"Final generator (EMA) LoRA saved in peft format: "
                            f"{transformer_dir}"
                        )
                    else:
                        logger.warning(
                            "Final EMA save skipped: no LoRAConfig resolved for "
                            "generator_ema."
                        )
                else:
                    _save_diffusers_transformer(
                        state_dict=state_dict,
                        config_path=self.config.model.pretrained_path,
                        save_dir=transformer_dir,
                    )
                    logger.info(f"Final generator (EMA) saved: {transformer_dir}")
        finally:
            self._ema.copy_temp_to(self._trainable_params)

        if dist.is_initialized():
            barrier()

    def _get_extra_state_dicts(self) -> dict[str, dict]:
        """Pre-gather the EMA snapshot for the periodic checkpoint.

        When EMA is enabled, this swaps EMA weights into the FSDP-managed
        generator, gathers a full state dict to rank 0, and swaps the training
        weights back in. The returned dict is fed to ``save_checkpoint`` so
        the EMA copy is written as ``generator_ema.pt`` and used for the
        ``transformer/`` inference folder.

        All ranks must call this because ``_gather_fsdp_state_dict`` issues
        collective FSDP ops. Only rank 0 returns a populated state dict;
        other ranks return an empty placeholder (which is harmless because
        the merge inside ``save_checkpoint`` runs inside ``is_main_process``).
        """
        if self._ema is None:
            return {}

        from rtdmd.utils.checkpoint import _gather_fsdp_state_dict

        self._ema.copy_ema_to(self._trainable_params, store_temp=True)
        try:
            ema_state = _gather_fsdp_state_dict(self.models["generator"])
        finally:
            self._ema.copy_temp_to(self._trainable_params)
        return {"generator_ema": ema_state}

    def _post_save_checkpoint(self) -> None:
        """Save per-rank GRPO state (e.g., the per-prompt stat tracker).

        The EMA snapshot is produced by :py:meth:`_get_extra_state_dicts` and
        written by :func:`save_checkpoint`, so this hook only handles state
        that must be persisted per rank.
        """
        ckpt_dir = os.path.join(self.config.output_dir, f"checkpoint-{self.global_step}")
        rank = get_rank()

        grpo_state = {}
        if self._stat_tracker is not None:
            grpo_state["stat_tracker"] = self._stat_tracker.state_dict()
        if grpo_state:
            os.makedirs(ckpt_dir, exist_ok=True)
            path = os.path.join(ckpt_dir, f"grpo_state_rank{rank}.pt")
            torch.save(grpo_state, path)

    def _post_load_checkpoint(self, ckpt_dir: str, extra: dict) -> None:
        """Restore per-rank GRPO state."""
        rank = get_rank()
        path = os.path.join(ckpt_dir, f"grpo_state_rank{rank}.pt")
        if os.path.exists(path):
            state = torch.load(path, map_location="cpu", weights_only=False)
            if self._stat_tracker is not None and "stat_tracker" in state:
                self._stat_tracker.load_state_dict(state["stat_tracker"])
            logger.info(f"  Restored per-rank GRPO state from {path}")

    def _get_lora_model_names(self) -> set[str]:
        """Return LoRA model names for checkpoint saving.

        Includes ``generator_ema`` when EMA is enabled so the pre-gathered
        EMA snapshot returned by :py:meth:`_get_extra_state_dicts` gets the
        LoRA-only filter applied in :func:`save_checkpoint` (the EMA snapshot
        shares the generator's LoRA config; see
        :func:`rtdmd.utils.checkpoint._resolve_lora_config`).
        """
        lora_models = set()
        if self.config.model.generator_lora.enabled:
            lora_models.add("generator")
            if self._ema is not None:
                lora_models.add("generator_ema")
        return lora_models

    def _get_lora_configs(self) -> dict[str, Any]:
        """Return LoRA configs for peft format saving."""
        lora_configs = {}
        if self.config.model.generator_lora.enabled:
            lora_configs["generator"] = self.config.model.generator_lora
        return lora_configs
