"""
DMD (Distribution Matching Distillation) Trainer for RTDMD.

Implements the core DMD algorithm for distilling flow-matching models into
few-step generators. Manages three model instances:
- Generator: trainable student model, produces clean latents from noise
- Fake score network: trainable estimator of the generator's output distribution
- Teacher (real score): frozen pretrained model, provides the target distribution

The training alternates between:
1. Fake score updates: train fake_score_net on generator outputs via flow matching
2. Generator updates: minimize DMD loss (real_score - fake_score gradient)

Reference papers:
- DMD: "One-step Diffusion with Distribution Matching Distillation" (CVPR 2024)
- DMD2: "Improved Distribution Matching Distillation" (Yin et al., 2024)
"""

from __future__ import annotations

import copy
import logging
import os
from contextlib import nullcontext

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.training_utils import (
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
)

from rtdmd.config import RTDMDConfig
from rtdmd.schedulers import get_scheduler
from rtdmd.data.prompt_dataset import create_prompt_dataloader
from rtdmd.models.flux import (
    compute_sigmas_flux,
    encode_prompts_flux,
    load_flux_models,
    predict_noise_flux,
)
from rtdmd.models.flux2_klein import (
    compute_sigmas_flux2_klein,
    encode_prompts_flux2_klein,
    load_flux2_klein_models,
    predict_noise_flux2_klein,
    unpatchify_flux2_latents,
)
from rtdmd.models.sd35 import (
    compute_sigmas_sd35,
    load_sd35_models,
    encode_prompts_sd35,
    extract_into_tensor,
    predict_noise_sd35,
)
from rtdmd.parallel.utils import (
    fsdp_wrap_model,
    get_transformer_wrap_policy,
    get_world_size,
)
from rtdmd.trainers.base_trainer import BaseTrainer
from rtdmd.utils.checkpoint import (
    _is_lora_checkpoint,
    is_peft_lora_dir,
)
from rtdmd.utils.fast_init import fast_init
from rtdmd.utils.lora import setup_lora

logger = logging.getLogger(__name__)


def _replace_patch_embed_proj_with_linear(transformer: nn.Module) -> None:
    """Replace PatchEmbed.proj Conv2d with an equivalent nn.Linear.

    FSDP use_orig_params=True has a known shape-writeback bug when a model
    contains Conv2d layers whose weights have 4-D shape (out_ch, in_ch, kH, kW).
    During unsharding, FSDP tries to write back a flat 1-D tensor into the
    original parameter, but the stored shape is 4-D, causing:

        RuntimeError: Cannot writeback when the parameter shape changes.
        Expects torch.Size([98304]) but got torch.Size([1536, 16, 2, 2])

    SD3-medium's PatchEmbed.proj is a Conv2d(16, 1536, kernel_size=2, stride=2)
    that acts as a pure linear projection (no spatial mixing because
    kernel_size == stride == patch_size). We can safely replace it with a
    weight-equivalent nn.Linear and patch the forward method of PatchEmbed to
    reshape the input appropriately.

    The replacement is done in-place before FSDP wrapping so no ignored_modules
    hack is needed.
    """
    if not hasattr(transformer, "pos_embed"):
        return
    patch_embed = transformer.pos_embed
    if not hasattr(patch_embed, "proj"):
        return
    conv = patch_embed.proj
    if not isinstance(conv, nn.Conv2d):
        return

    out_ch, in_ch, kH, kW = conv.weight.shape  # e.g. (1536, 16, 2, 2)
    in_features = in_ch * kH * kW              # 64 for SD3 medium

    linear = nn.Linear(
        in_features, out_ch,
        bias=(conv.bias is not None),
        dtype=conv.weight.dtype,
    )
    linear.weight.data.copy_(conv.weight.data.view(out_ch, -1))
    if conv.bias is not None:
        linear.bias.data.copy_(conv.bias.data)

    # Store patch_size so the patched forward knows how to reshape.
    linear._patch_size = kH  # kH == kW == stride for standard PatchEmbed

    patch_embed.proj = linear

    # Monkey-patch PatchEmbed.forward to reshape input before calling linear.
    # PatchEmbed.forward calls self.proj(latent) where latent is [B, C, H, W].
    # After the swap self.proj is an nn.Linear, so we must fold the spatial dims:
    #   [B, C, H, W]  ->  [B, H/p * W/p, C*p*p]  ->  linear  ->  [B, N, out_ch]
    # then reshape back to [B, out_ch, H/p, W/p] so the rest of PatchEmbed.forward
    # (flatten, layer_norm, pos_embed addition) still works correctly.
    original_forward = patch_embed.__class__.forward

    def _patched_forward(self, latent):
        # If proj is still a Conv2d (e.g. another instance) fall through.
        if isinstance(self.proj, nn.Conv2d):
            return original_forward(self, latent)

        p = self.proj._patch_size
        B, C, H, W = latent.shape
        Hp, Wp = H // p, W // p
        # Unfold into patches: [B, C*p*p, N] -> [B, N, C*p*p]
        patches = latent.unfold(2, p, p).unfold(3, p, p)  # [B,C,Hp,Wp,p,p]
        patches = patches.contiguous().view(B, C, Hp, Wp, p * p)
        patches = patches.permute(0, 2, 3, 1, 4).contiguous()  # [B,Hp,Wp,C,p*p]
        patches = patches.view(B, Hp * Wp, C * p * p)          # [B,N,C*p*p]

        # Linear projection
        out = self.proj(patches)   # [B, N, out_ch]

        # Reshape back to BCHW for the rest of PatchEmbed.forward
        out = out.view(B, Hp, Wp, -1).permute(0, 3, 1, 2)  # [B, out_ch, Hp, Wp]

        # Now run the remainder of the original forward manually, skipping
        # the self.proj(latent) call.
        if self.pos_embed_max_size is not None:
            height, width = latent.shape[-2:]
        else:
            height, width = Hp, Wp
        if self.flatten:
            out = out.flatten(2).transpose(1, 2)  # BCHW -> BNC
        if self.layer_norm:
            out = self.norm(out)
        if self.pos_embed is None:
            return out.to(out.dtype)
        if self.pos_embed_max_size:
            pos_embed = self.cropped_pos_embed(height, width)
        else:
            from diffusers.models.embeddings import get_2d_sincos_pos_embed
            if self.height != height or self.width != width:
                pos_embed = get_2d_sincos_pos_embed(
                    embed_dim=self.pos_embed.shape[-1],
                    grid_size=(height, width),
                    base_size=self.base_size,
                    interpolation_scale=self.interpolation_scale,
                    device=out.device,
                    output_type="pt",
                )
                pos_embed = pos_embed.float().unsqueeze(0)
            else:
                pos_embed = self.pos_embed
        return (out + pos_embed).to(out.dtype)

    # Bind the patched forward to this specific instance (not the class).
    import types
    patch_embed.forward = types.MethodType(_patched_forward, patch_embed)

    logger.debug(
        f"Replaced PatchEmbed.proj Conv2d({in_ch},{out_ch},{kH},{kW}) "
        f"with Linear({in_features},{out_ch}) for FSDP compatibility"
    )


def dmd_loss(
    latents: torch.Tensor,
    pred_fake_x0: torch.Tensor,
    pred_real_x0: torch.Tensor,
    normalize: bool = True,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute the Distribution Matching Distillation loss.

    The DMD gradient pushes the generator's output distribution toward the
    teacher's distribution by computing: grad = (fake_x0 - real_x0), normalized
    by the teacher's prediction magnitude.

    Formulation:
        grad = (p_real - p_fake) / |p_real|.mean()
    where p_real = latents - pred_real_x0, p_fake = latents - pred_fake_x0

    Equivalently: grad = pred_fake_x0 - pred_real_x0 (before normalization by p_real).

    Args:
        latents: Generator's clean output latents [B, C, H, W].
        pred_fake_x0: Fake score network's x0 prediction.
        pred_real_x0: Teacher's x0 prediction.
        normalize: Whether to normalize gradient by |p_real|.mean().

    Returns:
        Tuple of (loss, metrics_dict).
    """
    with torch.no_grad():
        p_real = latents - pred_real_x0
        p_fake = latents - pred_fake_x0
        grad = p_real - p_fake

        if normalize:
            normalizer = torch.abs(p_real).mean(dim=[1, 2, 3], keepdim=True)
            grad = grad / normalizer

        grad = torch.nan_to_num(grad)
        gradient_norm = torch.norm(grad).item()

    # The loss gradient w.r.t. generator params effectively applies the DMD gradient
    loss = 0.5 * F.mse_loss(
        latents.double(),
        (latents - grad).detach().double(),
        reduction="mean",
    )

    metrics = {
        "loss_dm": loss.item(),
        "gradient_norm": gradient_norm,
    }
    return loss, metrics


# ---------------------------------------------------------------------------
# DMD Trainer
# ---------------------------------------------------------------------------

class DMDTrainer(BaseTrainer):
    """DMD Trainer for flow-matching text-to-image models.

    Trains a generator to match the teacher's output distribution in few steps
    using the DMD objective. The fake score network learns the generator's
    current output distribution to provide the contrast signal.

    Training loop per step:
    1. [Always] Update fake score network on generator's outputs (flow matching)
    2. [Every fake_update_ratio steps] Update generator via DMD loss
    """

    def __init__(self, config: RTDMDConfig):
        super().__init__(config)
        self.model_type = str(self.config.model.model_type).lower()
        self._is_flux2_klein = self.model_type == "flux2_klein"
        self._is_flux = self.model_type == "flux"
        if self.model_type not in {"sd35", "flux2_klein", "flux"}:
            raise ValueError(
                f"DMDTrainer supports model_type in ['sd35', 'flux2_klein', 'flux'], "
                f"got: {self.model_type}"
            )
        # Model components (populated in setup_models)
        self.generator: nn.Module | None = None
        self.generator_ema: nn.Module | None = None  # EMA copy of generator
        self.fake_score_net: nn.Module | None = None
        self.teacher: nn.Module | None = None
        self.text_encoders: list[nn.Module] = []
        self.tokenizers: list = []
        self.scheduler: FlowMatchEulerDiscreteScheduler | None = None
        self.x0_scheduler = None

        # Precomputed schedule tensors (populated in setup_models)
        self.sigmas: torch.Tensor | None = None

        # Cached unconditional embeddings for CFG (populated lazily)
        self._uncond_embeds: tuple | None = None
        # Cache denoising sigma schedules by step count for fast repeated generation.
        self._denoising_sigmas_cache: dict[int, list[float]] = {}

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
        """Load backend models and create the 3-model DMD setup.

        Creates:
        - generator: copy of pretrained transformer (trainable)
        - fake_score_net: copy of pretrained transformer (trainable)
        - teacher: original pretrained transformer (frozen)
        - text_encoders: frozen, shared across all three
        """
        model_cfg = self.config.model
        dist_cfg = self.config.distributed

        logger.info("=" * 60)
        logger.info("Setting up DMD models")
        logger.info("=" * 60)

        # Load backend components based on model_type.
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

        # Enable gradient checkpointing if configured
        if model_cfg.gradient_checkpointing:
            base_transformer.enable_gradient_checkpointing()

        # Create the three transformer instances
        # Generator / fake: trainable copies (full weight or LoRA)
        self.generator = copy.deepcopy(base_transformer)
        self.fake_score_net = copy.deepcopy(base_transformer)

        # Teacher (real score): frozen original
        self.teacher = base_transformer

        # Apply role-specific initialization ckpts BEFORE freezing/wrapping.
        # This allows generator/fake/teacher to start from different checkpoints.
        self._load_model_init_from_path(
            self.generator,
            model_cfg.generator_init_path,
            "generator",
        )
        self._load_model_init_from_path(
            self.fake_score_net,
            model_cfg.fake_score_init_path,
            "fake_score",
        )
        self._load_model_init_from_path(
            self.teacher,
            model_cfg.teacher_init_path,
            "teacher",
        )

        # Inject LoRA after role-specific base initialization.
        self.generator_lora_params = []
        if model_cfg.generator_lora.enabled:
            self.generator_lora_params = setup_lora(self.generator, model_cfg.generator_lora)
            logger.info(f"  Generator: LoRA injected (rank={model_cfg.generator_lora.rank})")
        else:
            self.generator.train()
            logger.info("  Generator: trainable copy created (full weight)")

        self.fake_score_lora_params = []
        if model_cfg.fake_score_lora.enabled:
            self.fake_score_lora_params = setup_lora(self.fake_score_net, model_cfg.fake_score_lora)
            logger.info(f"  Fake score: LoRA injected (rank={model_cfg.fake_score_lora.rank})")
        else:
            self.fake_score_net.train()
            logger.info("  Fake score network: trainable copy created (full weight)")

        self.teacher.requires_grad_(False)
        self.teacher.eval()
        logger.info("  Teacher: frozen (no grad)")

        # EMA generator: frozen copy updated via exponential moving average
        # (default: ema_decay=0.9999, ema_every=1)
        # When generator uses LoRA, EMA is a copy of the LoRA-injected generator,
        # and ema_update() naturally only touches params that exist in both models
        # (including lora_A/lora_B).
        if self.config.train.use_ema:
            self.generator_ema = copy.deepcopy(self.generator)
            self.generator_ema.requires_grad_(False)
            self.generator_ema.eval()
            logger.info(
                f"  Generator EMA: created (decay={self.config.train.ema_decay}, "
                f"every={self.config.train.ema_every})"
            )
        else:
            self.generator_ema = None

        # Freeze text encoders and VAE (not needed for DMD, but keep for future eval)
        for enc in self.text_encoders:
            enc.requires_grad_(False)
            enc.eval()
        logger.info("  Text encoders: frozen")

        # Save VAE for evaluation (decode latents → images).
        self.vae = components["vae"]
        self.vae.requires_grad_(False)
        self.vae.eval()
        self.vae.to(torch.device("cuda"))
        logger.info(f"  VAE: frozen, stored on GPU ({self.vae.__class__.__name__})")

        # Extract latent shape from model config and VAE (no hard-coding)
        vae_scale_factor = 2 ** (len(components["vae"].config.block_out_channels) - 1)
        if self._is_flux2_klein:
            # Flux2 uses 2x2 latent patchification before transformer tokens, so
            # transformer.in_channels already accounts for the packed channels.
            self.latent_channels = base_transformer.config.in_channels
            latent_divisor = vae_scale_factor * 2
        elif self._is_flux:
            # FLUX.1: transformer.in_channels=64 is the *packed* channel count
            # (16 VAE channels x 2x2 packing). The VAE-latent-domain channel
            # count we operate on in latents/x0/noise tensors is 16, derived
            # from vae.config.latent_channels. Packing is done inside
            # predict_noise_flux. Latent H/W = image_res / vae_scale_factor.
            self.latent_channels = int(components["vae"].config.latent_channels)
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

        # Precompute sigma schedule for flow matching.
        # sigmas[i] gives the noise level at integer timestep i (flipped so
        # sigmas[0]=0, sigmas[T]=1).
        #
        # IMPORTANT: For FLUX.1 / FLUX.2 (use_dynamic_shifting=True), the
        # scheduler intentionally LEAVES `scheduler.sigmas` unshifted at init.
        # FLUX models are trained on this unshifted [1.0, 0.999, ..., 0.001]
        # grid (see diffusers' examples/dreambooth/train_dreambooth_lora_flux.py
        # which uses `noise_scheduler_copy.sigmas` directly without any mu
        # shifting). Dynamic shifting (mu) is applied only at INFERENCE time
        # via FluxPipeline.set_timesteps(mu=...). The backward-simulation
        # generator path in _generate_latents already accounts for this by
        # calling compute_sigmas_flux/flux2_klein with the proper image_seq_len.
        #
        # For SD3.5 (use_dynamic_shifting=False, shift=3.0), the scheduler's
        # __init__ applies the static shift to `sigmas`, so SD3.5 also sees
        # its training distribution here.
        #
        # Either way, `self.sigmas` matches the active model's training
        # distribution and should NOT be re-shifted.
        self.sigmas = torch.flip(self.scheduler.sigmas, dims=[0]).cuda()

        # Create x0-prediction scheduler for backward simulation / inference.
        scheduler_cls = get_scheduler(self.config.dmd.generator_scheduler)
        self.x0_scheduler = scheduler_cls.from_config(self.scheduler.config)
        if hasattr(self.x0_scheduler, "set_eta"):
            self.x0_scheduler.set_eta(self.config.dmd.cps_eta)
        self._denoising_sigmas_cache.clear()
        default_steps = max(1, min(
            int(self.config.dmd.num_inference_steps),
            len(self.config.dmd.denoising_step_list),
        ))
        self._denoising_sigmas = self._get_denoising_sigmas(default_steps)
        self.x0_scheduler.set_timesteps(sigmas=self._denoising_sigmas)
        logger.info(
            f"  X0 scheduler: denoising_step_list={self.config.dmd.denoising_step_list}, "
            f"default_sigmas={self._denoising_sigmas}, "
            f"timesteps={self.x0_scheduler.timesteps.tolist()}"
        )

        # Hook for subclasses to modify models before distributed wrapping
        # (e.g., inject extra LoRA adapters on raw, non-FSDP models).
        self._pre_wrap_models()

        # Wrap models with FSDP or DDP
        self._wrap_models(dist_cfg)

        # Register in self.models for checkpoint saving
        self.models = {
            "generator": self.generator,
            "fake_score_net": self.fake_score_net,
        }
        if self.generator_ema is not None:
            self.models["generator_ema"] = self.generator_ema
        # Teacher is not saved (it's the original pretrained model)

        # Move text encoders to GPU
        device = torch.device("cuda")
        for enc in self.text_encoders:
            enc.to(device)

        logger.info("Model setup complete")
        logger.info("=" * 60)

    def _pre_wrap_models(self) -> None:
        """Hook called after models are created but BEFORE FSDP/DDP wrapping.

        Override in subclasses to modify models that require raw (non-wrapped)
        access, e.g., LoRA injection, deepcopy for reference models.
        Subclasses MUST call super()._pre_wrap_models() to preserve the opt-in
        SD3-Medium PatchEmbed Conv2d -> Linear fix below.

        When config.model.fsdp_patch_embed_linear is True, replaces any
        Conv2d-based PatchEmbed.proj with an equivalent Linear projection so
        FSDP use_orig_params=True (which has a known shape-writeback bug for
        4-D Conv2d weights) can shard the whole model. Disabled by default to
        keep SD3.5 / Flux2 / etc. behavior and saved checkpoints unchanged.
        """
        if not self.config.model.fsdp_patch_embed_linear:
            return
        for m in [self.generator, self.fake_score_net, self.teacher, self.generator_ema]:
            if m is not None:
                _replace_patch_embed_proj_with_linear(m)

    def _wrap_models(self, dist_cfg) -> None:
        """Wrap trainable models with FSDP or DDP."""
        strategy = dist_cfg.strategy
        # FSDP param/reduce/buffer dtype follows model.dtype (normally bf16).
        # train.autocast_dtype independently controls forward-pass autocast.
        fsdp_precision = self.config.model.dtype

        if strategy == "fsdp":
            # Use transformer block-level wrapping for better FSDP granularity
            # SD3 and Flux2 use different block classes.
            try:
                if self._is_flux2_klein:
                    from diffusers.models.transformers.transformer_flux2 import (
                        Flux2SingleTransformerBlock,
                        Flux2TransformerBlock,
                    )

                    wrap_policy = get_transformer_wrap_policy(
                        {Flux2TransformerBlock, Flux2SingleTransformerBlock}
                    )
                elif self._is_flux:
                    from diffusers.models.transformers.transformer_flux import (
                        FluxSingleTransformerBlock,
                        FluxTransformerBlock,
                    )

                    wrap_policy = get_transformer_wrap_policy(
                        {FluxTransformerBlock, FluxSingleTransformerBlock}
                    )
                else:
                    from diffusers.models.transformers.transformer_sd3 import JointTransformerBlock

                    wrap_policy = get_transformer_wrap_policy(JointTransformerBlock)
            except ImportError:
                wrap_policy = None
                logger.warning(
                    "Could not import transformer block class for FSDP wrap policy, "
                    "falling back to size-based wrapping"
                )

            self.generator = fsdp_wrap_model(
                self.generator,
                sharding_strategy=dist_cfg.fsdp_sharding,
                fsdp_precision=fsdp_precision,
                auto_wrap_policy=wrap_policy,
            )
            self.fake_score_net = fsdp_wrap_model(
                self.fake_score_net,
                sharding_strategy=dist_cfg.fsdp_sharding,
                fsdp_precision=fsdp_precision,
                auto_wrap_policy=wrap_policy,
            )
            self.teacher = fsdp_wrap_model(
                self.teacher,
                sharding_strategy=dist_cfg.fsdp_sharding,
                fsdp_precision=fsdp_precision,
                auto_wrap_policy=wrap_policy,
            )
            if self.generator_ema is not None:
                self.generator_ema = fsdp_wrap_model(
                    self.generator_ema,
                    sharding_strategy=dist_cfg.fsdp_sharding,
                    fsdp_precision=fsdp_precision,
                    auto_wrap_policy=wrap_policy,
                )
            logger.info(
                f"  Models wrapped with FSDP (sharding={dist_cfg.fsdp_sharding}, "
                f"fsdp_precision={fsdp_precision})"
            )
        elif strategy == "ddp":
            from rtdmd.parallel.utils import ddp_wrap_model
            device = torch.device("cuda")
            self.generator = self.generator.to(device)
            self.fake_score_net = self.fake_score_net.to(device)
            self.teacher = self.teacher.to(device)
            # When LoRA is enabled, many parameters are frozen so DDP must
            # be told to find unused parameters to avoid hanging in backward.
            model_cfg = self.config.model
            has_lora = model_cfg.generator_lora.enabled or model_cfg.fake_score_lora.enabled
            self.generator = ddp_wrap_model(
                self.generator, find_unused_parameters=has_lora,
            )
            self.fake_score_net = ddp_wrap_model(
                self.fake_score_net, find_unused_parameters=has_lora,
            )
            # Teacher and EMA are frozen, no need for DDP
            if self.generator_ema is not None:
                self.generator_ema = self.generator_ema.to(device)
            logger.info("  Trainable models wrapped with DDP")
        else:
            raise ValueError(f"Unknown distributed strategy: {strategy}")

    # ------------------------------------------------------------------
    # Optimizer setup
    # ------------------------------------------------------------------

    def setup_optimizers(self) -> None:
        """Create separate optimizers for generator and fake score network.

        When LoRA is enabled, only LoRA parameters (requires_grad=True) are
        passed to the optimizer, reducing optimizer state memory.
        """
        gen_cfg = self.config.solver.generator
        fake_cfg = self.config.solver.fake_score

        # Filter to only trainable parameters (LoRA params when LoRA enabled)
        gen_params = [p for p in self.generator.parameters() if p.requires_grad]
        fake_params = [p for p in self.fake_score_net.parameters() if p.requires_grad]

        self.optimizers["generator"] = torch.optim.AdamW(
            gen_params,
            lr=gen_cfg.lr,
            betas=(gen_cfg.beta1, gen_cfg.beta2),
            eps=gen_cfg.eps,
            weight_decay=gen_cfg.weight_decay,
        )
        self.optimizers["fake_score"] = torch.optim.AdamW(
            fake_params,
            lr=fake_cfg.lr,
            betas=(fake_cfg.beta1, fake_cfg.beta2),
            eps=fake_cfg.eps,
            weight_decay=fake_cfg.weight_decay,
        )

        # LR schedulers: linear warmup then constant
        self.schedulers["generator"] = self.create_warmup_constant_scheduler(
            self.optimizers["generator"], gen_cfg.warmup_steps,
        )
        self.schedulers["fake_score"] = self.create_warmup_constant_scheduler(
            self.optimizers["fake_score"], fake_cfg.warmup_steps,
        )

        logger.info(
            f"Optimizers created: generator (lr={gen_cfg.lr}, params={len(gen_params)}), "
            f"fake_score (lr={fake_cfg.lr}, params={len(fake_params)})"
        )

    # ------------------------------------------------------------------
    # Dataloader
    # ------------------------------------------------------------------

    def _create_dataloader(self):
        """Create prompt dataloader."""
        return create_prompt_dataloader(
            prompt_path=self.config.prompt_path,
            batch_size=self.config.train.batch_size,
            distributed=(get_world_size() > 1),
            seed=self.config.train.seed,
        )

    # ------------------------------------------------------------------
    # Core training step
    # ------------------------------------------------------------------

    def train_step(self, batch: list[str]) -> dict[str, dict[str, float]]:
        """Execute one DMD training iteration.

        Alternates between fake score updates and generator updates based on
        fake_update_ratio. For example, with ratio=5:
        steps 1-4: only fake score update
        step 5: fake score update + generator update

        Args:
            batch: List of text prompts.

        Returns:
            Dict of section -> metrics for logging.
        """
        dmd_cfg = self.config.dmd
        metrics: dict[str, dict[str, float]] = {}

        # Encode text prompts (cached computation)
        with self.timer.measure("text_encoding"):
            prompt_embeds, pooled_embeds = self._encode_prompts(batch)
            uncond_embeds, uncond_pooled = self._get_uncond_embeds(len(batch))

        # === Stage 1: Fake Score Network Update (every step) ===
        with self.timer.measure("fake_score_update"):
            fake_metrics = self._update_fake_score(
                prompt_embeds, pooled_embeds, uncond_embeds, uncond_pooled,
            )
            metrics["fake_score"] = fake_metrics

        # === Stage 2: Generator Update (every fake_update_ratio steps) ===
        is_generator_step = (self.global_step + 1) % dmd_cfg.fake_update_ratio == 0
        if is_generator_step:
            with self.timer.measure("generator_update"):
                gen_metrics = self._update_generator(
                    prompt_embeds, pooled_embeds, uncond_embeds, uncond_pooled,
                )
                metrics["dmd"] = gen_metrics

        # === Stage 3: EMA Update (every ema_every steps) ===
        train_cfg = self.config.train
        if self.generator_ema is not None and (self.global_step + 1) % train_cfg.ema_every == 0:
            self.ema_update(self.generator, self.generator_ema, train_cfg.ema_decay)

        return metrics

    # ------------------------------------------------------------------
    # DMD real score model selection
    # ------------------------------------------------------------------

    @property
    def dmd_real_score_model(self) -> nn.Module:
        """Model used as real score (teacher) in DMD loss. Subclasses may override."""
        return self.teacher

    def _get_denoising_sigmas(self, num_steps: int) -> list[float]:
        """Return denoising sigmas for generation, cached by step count."""
        max_steps = len(self.config.dmd.denoising_step_list)
        clamped_steps = max(1, min(int(num_steps), max_steps))
        if clamped_steps in self._denoising_sigmas_cache:
            return self._denoising_sigmas_cache[clamped_steps]

        if self._is_flux2_klein:
            sigmas = compute_sigmas_flux2_klein(
                denoising_step_list=self.config.dmd.denoising_step_list,
                scheduler_config=self.scheduler.config,
                image_seq_len=self.latent_size * self.latent_size,
                num_steps=clamped_steps,
            )
        elif self._is_flux:
            # FLUX image_seq_len = number of packed image tokens =
            # (latent_H / 2) * (latent_W / 2). For 512x512 -> 1024 tokens,
            # 1024x1024 -> 4096 tokens.
            flux_image_seq_len = (self.latent_size // 2) * (self.latent_size // 2)
            sigmas = compute_sigmas_flux(
                denoising_step_list=self.config.dmd.denoising_step_list,
                scheduler_config=self.scheduler.config,
                image_seq_len=flux_image_seq_len,
                num_steps=clamped_steps,
            )
        else:
            sigmas = compute_sigmas_sd35(
                self.config.dmd.denoising_step_list[:clamped_steps],
                num_train_timesteps=self.scheduler.config.num_train_timesteps,
                shift=self.scheduler.config.shift,
            )

        self._denoising_sigmas_cache[clamped_steps] = sigmas
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
    # Fake score network update
    # ------------------------------------------------------------------

    def _update_fake_score(
        self,
        prompt_embeds: torch.Tensor,
        pooled_embeds: torch.Tensor,
        uncond_embeds: torch.Tensor,
        uncond_pooled: torch.Tensor,
    ) -> dict[str, float]:
        """Train the fake score network on the generator's current output distribution.

        The fake score network learns to estimate the score (velocity field) of the
        generator's output distribution using a standard flow matching objective:
        1. Generate clean latents x0 from the generator (no grad)
        2. Add noise: x_t = (1 - sigma_t) * x0 + sigma_t * noise
        3. Predict velocity: v_pred = fake_score_net(x_t, t)
        4. Loss = weighted MSE(v_pred, noise - x0)

        Returns:
            Dict of training metrics.
        """
        dmd_cfg = self.config.dmd
        batch_size = prompt_embeds.shape[0]

        # 1. Generate fake data from generator (detached, no grad through generator)
        with torch.no_grad():
            x0_fake, actual_steps = self._generate_latents(
                prompt_embeds, pooled_embeds, uncond_embeds, uncond_pooled,
                num_steps=dmd_cfg.num_inference_steps,
                random_stop=dmd_cfg.random_stop_idx,
            )
            x0_fake = x0_fake.detach()

        # 2. Sample timesteps for fake score training
        indices = self._sample_timesteps(
            batch_size=batch_size,
            min_step=dmd_cfg.fake_min_step,
            max_step=dmd_cfg.fake_max_step,
            sampling=dmd_cfg.fake_timestep_sampling,
            logit_mean=dmd_cfg.logit_mean,
            logit_std=dmd_cfg.logit_std,
            device=x0_fake.device,
        )
        timesteps = self.scheduler.timesteps[indices.cpu()].to(device=x0_fake.device)

        # 3. Compute sigmas and create noisy latents: x_t = (1 - sigma) * x0 + sigma * noise
        sigmas = self._get_sigmas(timesteps, n_dim=x0_fake.ndim, dtype=x0_fake.dtype)
        noise = torch.randn_like(x0_fake)
        noisy_latents = (1.0 - sigmas) * x0_fake + sigmas * noise

        # 4. Compute flow matching loss weight
        weighting = compute_loss_weighting_for_sd3(
            weighting_scheme=dmd_cfg.fake_timestep_sampling, sigmas=sigmas,
        )

        # 5. Forward pass through fake score network
        fake_train_cfg = dmd_cfg.fake_train_guidance_scale
        with self._autocast():
            v_pred = self._predict_noise(
                self.fake_score_net,
                noisy_latents,
                prompt_embeds,
                timesteps,
                pooled_embeds,
                guidance_scale=fake_train_cfg,
                uncond_text_embeddings=uncond_embeds if fake_train_cfg > 1.0 else None,
                uncond_pooled_prompt_embeds=uncond_pooled if fake_train_cfg > 1.0 else None,
            )

        # 6. Flow matching target: v = noise - x0
        target = noise - x0_fake
        loss_fake = torch.mean(
            (weighting.float() * (v_pred.float() - target.float()) ** 2).reshape(
                batch_size, -1
            ),
            dim=1,
        ).mean()

        # 7. Backward and optimizer step
        self.optimizers["fake_score"].zero_grad(set_to_none=True)
        loss_fake.backward()
        fake_grad_norm = self._compute_grad_norm(self.fake_score_net)
        if self.config.solver.fake_score.max_grad_norm > 0:
            self._clip_grad_norm(
                self.fake_score_net,
                self.config.solver.fake_score.max_grad_norm,
            )
        self.optimizers["fake_score"].step()
        self.schedulers["fake_score"].step()

        return {
            "loss_fake": loss_fake.item(),
            "grad_norm": fake_grad_norm,
            "lr": self.schedulers["fake_score"].get_last_lr()[0],
            "gen_steps": actual_steps,
        }

    # ------------------------------------------------------------------
    # Generator update
    # ------------------------------------------------------------------

    def _update_generator(
        self,
        prompt_embeds: torch.Tensor,
        pooled_embeds: torch.Tensor,
        uncond_embeds: torch.Tensor,
        uncond_pooled: torch.Tensor,
    ) -> dict[str, float]:
        """Update the generator using the DMD loss.

        DMD loss computation:
        1. Generate clean latents x0 from the generator (WITH grad)
        2. Add noise: x_t = sigma_t * noise + (1 - sigma_t) * x0
        3. Get teacher prediction: pred_real = teacher(x_t, t) [frozen, with CFG]
        4. Get fake score prediction: pred_fake = fake_score(x_t, t) [frozen for this step]
        5. Compute DMD gradient and loss

        Returns:
            Dict of training metrics.
        """
        dmd_cfg = self.config.dmd
        batch_size = prompt_embeds.shape[0]

        # 1. Generate clean latents from generator (WITH gradient)
        x0_gen, actual_steps = self._generate_latents(
            prompt_embeds, pooled_embeds, uncond_embeds, uncond_pooled,
            num_steps=dmd_cfg.num_inference_steps,
            random_stop=dmd_cfg.random_stop_idx,
        )

        # 2. Sample timesteps for DMD loss
        timesteps = self._sample_timesteps(
            batch_size=batch_size,
            min_step=dmd_cfg.min_step,
            max_step=dmd_cfg.max_step,
            sampling=dmd_cfg.dmd_timestep_sampling,
            logit_mean=dmd_cfg.dmd_logit_mean,
            logit_std=dmd_cfg.dmd_logit_std,
            device=x0_gen.device,
        )

        # 3. Add noise to generator output
        current_sigmas = extract_into_tensor(self.sigmas, timesteps, x0_gen.shape)
        noise = torch.randn_like(x0_gen)
        noisy_latents = current_sigmas * noise + (1.0 - current_sigmas) * x0_gen

        # Compute actual timestep values for the transformer
        current_timesteps = self._sigma_to_model_timestep(current_sigmas).view(-1)

        # 4. Get teacher (real score) prediction — frozen, with CFG
        with torch.no_grad():
            # Teacher CFG scale: sample from [min, max] (set min==max for fixed)
            # Use torch RNG + broadcast to ensure all ranks agree (Python RNG
            # differs per rank since seeds are offset by rank for data diversity).
            cfg_tensor = torch.empty(1, device=x0_gen.device).uniform_(
                dmd_cfg.real_guidance_scale_min, dmd_cfg.real_guidance_scale_max,
            )
            if dist.is_initialized():
                dist.broadcast(cfg_tensor, src=0)
            teacher_cfg = cfg_tensor.item()
            # NOTE: autocast under no_grad may behave unexpectedly in some PyTorch
            # versions. If teacher predictions look wrong or produce NaN,
            # switch to manual dtype casting (e.g., explicit .to(bf16)).
            with self._autocast():
                pred_real_noise = self._predict_noise(
                    self.dmd_real_score_model,
                    noisy_latents,
                    prompt_embeds,
                    current_timesteps,
                    pooled_embeds,
                    guidance_scale=teacher_cfg,
                    uncond_text_embeddings=uncond_embeds,
                    uncond_pooled_prompt_embeds=uncond_pooled,
                )
            # Convert noise prediction to x0 prediction: x0 = x_t - sigma * v
            pred_real_x0 = (noisy_latents - current_sigmas * pred_real_noise).to(noisy_latents.dtype)

        # 5. Get fake score prediction — frozen for this step
        with torch.no_grad():
            with self._autocast():
                pred_fake_noise = self._predict_noise(
                    self.fake_score_net,
                    noisy_latents,
                    prompt_embeds,
                    current_timesteps,
                    pooled_embeds,
                    guidance_scale=dmd_cfg.fake_guidance_scale,
                    uncond_text_embeddings=uncond_embeds if dmd_cfg.fake_guidance_scale > 1.0 else None,
                    uncond_pooled_prompt_embeds=uncond_pooled if dmd_cfg.fake_guidance_scale > 1.0 else None,
                )
            pred_fake_x0 = (noisy_latents - current_sigmas * pred_fake_noise).to(noisy_latents.dtype)

        # 6. Compute DMD loss
        loss_dm, dm_metrics = dmd_loss(
            x0_gen, pred_fake_x0, pred_real_x0,
            normalize=dmd_cfg.gradient_normalization,
        )

        # 7. Backward and optimizer step
        self.optimizers["generator"].zero_grad(set_to_none=True)
        loss_dm.backward()
        generator_grad_norm = self._compute_grad_norm(self.generator)
        if self.config.solver.generator.max_grad_norm > 0:
            self._clip_grad_norm(
                self.generator,
                self.config.solver.generator.max_grad_norm,
            )
        self.optimizers["generator"].step()
        self.schedulers["generator"].step()

        dm_metrics["lr"] = self.schedulers["generator"].get_last_lr()[0]
        dm_metrics["teacher_cfg"] = teacher_cfg
        dm_metrics["gen_steps"] = actual_steps
        dm_metrics["model_grad_norm"] = generator_grad_norm
        return dm_metrics

    # ------------------------------------------------------------------
    # Sample generation
    # ------------------------------------------------------------------

    def _generate_latents(
        self,
        prompt_embeds: torch.Tensor,
        pooled_embeds: torch.Tensor,
        uncond_embeds: torch.Tensor,  # noqa: ARG002 - reserved for future CFG-based generation
        uncond_pooled: torch.Tensor,  # noqa: ARG002 - reserved for future CFG-based generation
        num_steps: int = 4,
        gradient_truncation: bool = True,
        random_stop: bool = False,
        model: nn.Module | None = None,
        return_last_step_data: bool = False,
    ) -> tuple[torch.Tensor, int] | tuple[torch.Tensor, int, dict]:
        """Generate latents using CPSScheduler.

        Uses the x0-prediction scheduler with stochastic backward simulation:
        1. Start from pure noise x_t (at sigma=1.0, x_t = noise)
        2. Model predicts velocity v
        3. Scheduler.step() predicts x0 = x_t - sigma*v, then re-noises
           to next sigma level: prev = (1-sigma_next)*x0 + sigma_next*noise
        4. Return x0 (clean prediction) from the last executed step

        Gradient truncation:
        - When enabled, only the LAST executed denoising step retains gradients.

        Random stop:
        - When enabled, randomly samples a stop index from [1, num_steps].

        Args:
            prompt_embeds: Text embeddings [B, seq_len, dim].
            pooled_embeds: Pooled CLIP embeddings [B, pooled_dim].
            uncond_embeds: Unconditional text embeddings for CFG.
            uncond_pooled: Unconditional pooled embeddings for CFG.
            num_steps: Maximum number of denoising steps.
            gradient_truncation: If True, only the last step has gradients.
            random_stop: If True, randomly stop at step [1, num_steps].
            model: Model to use for denoising. Defaults to self.generator.
                   Pass self.generator_ema for evaluation.
            return_last_step_data: If True, also return a dict with data from
                the last denoising step (x_t before step, x_next after step,
                v_pred, sigma, sigma_next, timestep_index) needed for ISG or
                AC-DMD training computation.

        Returns:
            When return_last_step_data=False:
                Tuple of (latent tensors [B, C, H, W], actual steps executed).
            When return_last_step_data=True:
                Tuple of (latents, actual_steps, last_step_data dict).
        """
        if model is None:
            model = self.generator
        batch_size = prompt_embeds.shape[0]
        device = prompt_embeds.device
        dtype = prompt_embeds.dtype

        # Sigma schedule derived from denoising_step_list.
        # For Flux2-Klein this includes resolution-dependent dynamic shifting.
        sigmas = self._get_denoising_sigmas(num_steps)
        num_steps = len(sigmas)

        # Determine how many steps to actually execute
        if random_stop:
            # Random stop index in [1, num_steps].
            # Must be the same across all ranks for FSDP collective ops to work.
            stop_index = torch.randint(
                low=1, high=num_steps + 1, size=(1,),
                device=device, dtype=torch.long,
            )
            if dist.is_initialized():
                dist.broadcast(stop_index, src=0)
            stop_index = int(stop_index.item())
        else:
            stop_index = num_steps

        logger.debug(
            f"Backward simulation: {stop_index}/{num_steps} steps, "
            f"sigmas={sigmas[:stop_index]}"
        )

        # Reset scheduler state for this generation pass
        self.x0_scheduler.set_timesteps(sigmas=sigmas)
        self.x0_scheduler._step_index = 0

        # Start from pure noise (at sigma[0]=1.0, this IS the noisy sample)
        x_t = torch.randn(
            batch_size, self.latent_channels, self.latent_size, self.latent_size,
            device=device, dtype=dtype,
        )
        x0 = x_t  # will be overwritten

        # Stochastic multi-step denoising loop via scheduler
        last_step_data = None
        for idx in range(stop_index):
            is_last_step = (idx + 1 == stop_index)
            use_grad = not gradient_truncation or is_last_step
            ctx = nullcontext() if use_grad else torch.no_grad()

            with ctx:
                # Timestep for transformer: sigma * num_train_timesteps
                t = self.x0_scheduler.timesteps[idx]
                t_batch = t.expand(batch_size).to(device=device, dtype=torch.float32)

                # Capture x_t before the last step for ISG
                if return_last_step_data and is_last_step:
                    x_t_before_last = x_t

                with self._autocast():
                    v_pred = self._predict_noise(
                        model,
                        x_t,
                        prompt_embeds,
                        t_batch,
                        pooled_embeds,
                        guidance_scale=1.0,
                    )

                # Scheduler step: predict x0, re-noise to next sigma level
                step_output = self.x0_scheduler.step(v_pred, t, x_t, return_dict=True)

                # Extract x0 prediction for return (scheduler returns re-noised prev_sample,
                # but we need the clean x0 when random_stop truncates early)
                sigma = self.x0_scheduler.sigmas[idx]
                x0 = (x_t.float() - sigma * v_pred.float()).to(dtype)

                # Build last step data for AC-DMD training
                # (sigma_next=0 if final denoising step).
                if return_last_step_data and is_last_step:
                    sigma_next_val = (
                        self.x0_scheduler.sigmas[idx + 1].item()
                        if idx + 1 < len(self.x0_scheduler.sigmas)
                        else 0.0
                    )
                    # Store the raw timestep from the scheduler for precise ISG
                    # computation. Using round(sigma * 1000) is approximate;
                    # the scheduler's timestep is exact.
                    last_step_data = {
                        "x_t": x_t_before_last,
                        # Exact post-step sample at sigma_next from scheduler.step().
                        # This is important for trainers that define a sub-interval
                        # start as the next sigma level.
                        "x_next": step_output.prev_sample,
                        "v_pred": v_pred,
                        "sigma": sigma.item(),
                        "sigma_next": sigma_next_val,
                        "timestep": t.item(),  # exact scheduler timestep
                        "timestep_index": idx,
                    }

                x_t = step_output.prev_sample

        if return_last_step_data:
            return x0, stop_index, last_step_data
        return x0, stop_index

    # ------------------------------------------------------------------
    # Text encoding helpers
    # ------------------------------------------------------------------

    def _encode_prompts(
        self, prompts: list[str],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode a batch of text prompts using the active model backend.

        Returns:
            Tuple of (prompt_embeds, pooled_or_text_ids).
        """
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
        """Get unconditional (empty string) embeddings for CFG/backend conditioning.

        Caches the result and expands to match batch size.

        Returns:
            Tuple of (uncond_embeds, uncond_pooled_or_text_ids).
        """
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
        """Decode latent tensors to image tensors in [0, 1]."""
        if not self._is_flux2_klein:
            return super()._decode_latents_to_tensor(latents)

        device = latents.device
        if next(self.vae.parameters()).device != device:
            self.vae.to(device)

        latents_bn_mean = self.vae.bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
        latents_bn_std = torch.sqrt(self.vae.bn.running_var.view(1, -1, 1, 1) + self.vae.config.batch_norm_eps).to(
            latents.device,
            latents.dtype,
        )
        latents = latents * latents_bn_std + latents_bn_mean
        latents = unpatchify_flux2_latents(latents)
        images = self.vae.decode(latents.to(self.vae.dtype)).sample
        return (images / 2 + 0.5).clamp(0, 1)

    # ------------------------------------------------------------------
    # Gradient clipping helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _clip_grad_norm(model: nn.Module, max_norm: float) -> float:
        """Clip gradient norm, FSDP-aware.

        For FSDP models, the standard torch.nn.utils.clip_grad_norm_ only
        computes the norm of the local shard, which is incorrect.
        FSDP.clip_grad_norm_ gathers the global norm across all ranks.

        Returns:
            Global gradient norm before clipping.
        """
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        if isinstance(model, FSDP):
            norm = model.clip_grad_norm_(max_norm)
            return float(norm.item() if isinstance(norm, torch.Tensor) else norm)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        return float(norm.item() if isinstance(norm, torch.Tensor) else norm)

    @staticmethod
    def _compute_grad_norm(model: nn.Module) -> float:
        """Compute global gradient norm without modifying gradients."""
        return DMDTrainer._clip_grad_norm(model, float("inf"))

    # ------------------------------------------------------------------
    # Timestep sampling helpers
    # ------------------------------------------------------------------

    def _sample_timesteps(
        self,
        batch_size: int,
        min_step: int,
        max_step: int,
        sampling: str,
        logit_mean: float,
        logit_std: float,
        device: torch.device,
    ) -> torch.Tensor:
        """Sample integer timestep indices with configurable strategy.

        Args:
            batch_size: Number of timesteps to sample.
            min_step: Minimum timestep index (inclusive).
            max_step: Maximum timestep index (inclusive).
            sampling: "uniform" or "logit_normal".
            logit_mean: Logit-normal mean (only used when sampling == "logit_normal").
            logit_std: Logit-normal std (only used when sampling == "logit_normal").
            device: Target device.

        Returns:
            Integer timestep indices [batch_size] in [min_step, max_step].
        """
        max_step = min(max_step, 999)
        step_range = max_step - min_step + 1

        if sampling == "logit_normal":
            u = compute_density_for_timestep_sampling(
                weighting_scheme="logit_normal",
                batch_size=batch_size,
                logit_mean=logit_mean,
                logit_std=logit_std,
            )
            indices = (u * step_range).long().clamp(0, step_range - 1) + min_step
            return indices.to(device)
        else:
            return torch.randint(
                min_step, max_step + 1, (batch_size,),
                device=device, dtype=torch.long,
            )

    # ------------------------------------------------------------------
    # Sigma / timestep conversion helpers
    # ------------------------------------------------------------------

    def _num_train_timesteps(self) -> float:
        """Return scheduler num_train_timesteps as float.

        In this codebase, the transformer-facing timestep convention is:
            timestep_model = sigma * num_train_timesteps
        where sigma is already the schedule-space sigma used by FlowMatch/CPS
        (including any SD3.5 shift that has already been absorbed when sigma is
        constructed from scheduler tables).
        """
        if self.scheduler is not None and hasattr(self.scheduler, "config"):
            return float(
                getattr(
                    self.scheduler.config,
                    "num_train_timesteps",
                    self.config.dmd.num_train_timesteps,
                )
            )
        return float(self.config.dmd.num_train_timesteps)

    def _sigma_to_model_timestep(
        self,
        sigma: torch.Tensor | float,
    ) -> torch.Tensor | float:
        """Convert schedule-space sigma to model timestep."""
        num_ts = self._num_train_timesteps()
        if isinstance(sigma, torch.Tensor):
            return sigma.float() * num_ts
        return float(sigma) * num_ts

    def _model_timestep_to_sigma(
        self,
        timestep: torch.Tensor | float,
    ) -> torch.Tensor | float:
        """Convert model timestep to schedule-space sigma."""
        num_ts = self._num_train_timesteps()
        if isinstance(timestep, torch.Tensor):
            return timestep.float() / num_ts
        return float(timestep) / num_ts

    # ------------------------------------------------------------------
    # Scheduler helpers
    # ------------------------------------------------------------------

    def _get_sigmas(
        self,
        timesteps: torch.Tensor,
        n_dim: int = 4,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Look up sigma values for given timesteps from the scheduler.

        Args:
            timesteps: Timestep indices [B].
            n_dim: Number of dimensions to reshape to (for broadcasting).
            dtype: Output dtype.

        Returns:
            Sigma values [B, 1, 1, ...] with n_dim dimensions.
        """
        sigmas = self.scheduler.sigmas.to(device="cuda", dtype=dtype)
        schedule_timesteps = self.scheduler.timesteps.cuda()

        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma
