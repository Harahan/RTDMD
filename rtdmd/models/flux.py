"""
FLUX.1 Model Loading Factory.

Handles loading all components of the FLUX.1 pipeline:
- FluxTransformer2DModel (the DiT backbone)
- AutoencoderKL (VAE for latent encoding/decoding)
- Text encoders: CLIP (pooled, dim=768) + T5 (sequence, dim=4096, seq_len=512)
- Tokenizers
- FlowMatchEulerDiscreteScheduler (with dynamic shifting for FLUX.1-dev)

This factory follows the same backend interface as sd35.py and flux2_klein.py:
    load_flux_models(config) -> dict of named components

Guidance handling (the key FLUX.1-specific behavior):
- FLUX.1-dev has `guidance_embeds=True`: CFG is *distilled* into the model via
  a scalar guidance input (passed as `guidance` argument). A single forward
  pass replaces the traditional classifier-free guidance uncond+cond doubling.
- FLUX.1-schnell has `guidance_embeds=False`: no guidance input layer.
  `guidance=None` is passed in forward.

For DMD training with FLUX.1-dev:
- Teacher: pass guidance_scale=3.5 (the FLUX.1-dev default distilled CFG value)
- Generator / fake_score: pass guidance_scale=1.0 as a "no-CFG" dummy value
  (FLUX.1-dev training distribution included guidance ∈ [1.5, 5], so 1.0 is the
   weakest legal guidance signal). MUST pass a scalar (not None) because the
   model architecture has a guidance_in layer that expects an input.

Loads FLUX.1 transformer + VAE + text encoders with packed-latent helpers.
helpers, and diffusers FluxPipeline for the canonical implementations.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import torch
import torch.nn as nn
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    FluxTransformer2DModel,
)
from transformers import (
    CLIPTextModel,
    CLIPTokenizer,
    T5EncoderModel,
    T5TokenizerFast,
)

from rtdmd.config import ModelConfig
from rtdmd.parallel.utils import is_main_process
from rtdmd.utils.fast_init import fast_init

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dtype mapping
# ---------------------------------------------------------------------------

_DTYPE_MAP = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}


def _get_torch_dtype(dtype_str: str) -> torch.dtype:
    if dtype_str not in _DTYPE_MAP:
        raise ValueError(f"Unknown dtype: {dtype_str}. Choose from: {list(_DTYPE_MAP.keys())}")
    return _DTYPE_MAP[dtype_str]


# ---------------------------------------------------------------------------
# FSDP / DDP wrapping helpers (used by trainer to resolve config / dtype
# through arbitrary wrapping layers).
# ---------------------------------------------------------------------------

def _resolve_model_config(model: nn.Module) -> Any | None:
    """Best-effort retrieval of model config from wrapped modules."""
    current = model
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        cfg = getattr(current, "config", None)
        if cfg is not None:
            return cfg
        next_model = None
        for attr in ("module", "_orig_mod", "_fsdp_wrapped_module"):
            candidate = getattr(current, attr, None)
            if candidate is not None:
                next_model = candidate
                break
        current = next_model
    return None


def _resolve_model_dtype(model: nn.Module, fallback: torch.dtype) -> torch.dtype:
    """Best-effort retrieval of model dtype from wrapped modules."""
    current = model
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        model_dtype = getattr(current, "dtype", None)
        if isinstance(model_dtype, torch.dtype):
            return model_dtype
        try:
            return next(current.parameters()).dtype
        except (StopIteration, AttributeError):
            pass
        next_model = None
        for attr in ("module", "_orig_mod", "_fsdp_wrapped_module"):
            candidate = getattr(current, attr, None)
            if candidate is not None:
                next_model = candidate
                break
        current = next_model
    return fallback


# ---------------------------------------------------------------------------
# FLUX latent pack/unpack helpers (from diffusers FluxPipeline)
# ---------------------------------------------------------------------------

def pack_flux_latents(latents: torch.Tensor) -> torch.Tensor:
    """Pack [B, C, H, W] -> [B, (H/2)*(W/2), C*4] for FLUX transformer input.

    The 2x2 patchification turns 16-channel VAE latents into 64-channel tokens,
    which matches FluxTransformer2DModel.config.in_channels=64.
    """
    batch_size, num_channels, height, width = latents.shape
    if height % 2 != 0 or width % 2 != 0:
        raise ValueError(
            f"FLUX latent H/W must be divisible by 2; got H={height}, W={width}"
        )
    latents = latents.view(batch_size, num_channels, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    return latents.reshape(
        batch_size, (height // 2) * (width // 2), num_channels * 4
    ).contiguous()


def unpack_flux_latents(
    tokens: torch.Tensor, latent_height: int, latent_width: int
) -> torch.Tensor:
    """Unpack [B, (H/2)*(W/2), C*4] -> [B, C, H, W] for FLUX latent space.

    `latent_height` and `latent_width` are the VAE-latent-domain dims (before
    packing). For a 512x512 image with 8x VAE downsampling, latent dims are 64.
    """
    batch_size, num_tokens, channels_packed = tokens.shape
    if latent_height % 2 != 0 or latent_width % 2 != 0:
        raise ValueError(
            f"FLUX latent H/W must be divisible by 2; got H={latent_height}, W={latent_width}"
        )
    h_packed = latent_height // 2
    w_packed = latent_width // 2
    expected_tokens = h_packed * w_packed
    if num_tokens != expected_tokens:
        raise ValueError(
            f"Token count mismatch when unpacking FLUX latents: got {num_tokens}, "
            f"expected {expected_tokens} (latent H={latent_height}, W={latent_width})"
        )
    if channels_packed % 4 != 0:
        raise ValueError(
            f"Packed channels must be divisible by 4; got {channels_packed}"
        )
    out = tokens.view(batch_size, h_packed, w_packed, channels_packed // 4, 2, 2)
    out = out.permute(0, 3, 1, 4, 2, 5)
    return out.reshape(
        batch_size, channels_packed // 4, latent_height, latent_width
    ).contiguous()


def prepare_flux_latent_image_ids(
    latent_height: int,
    latent_width: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build FLUX latent token coordinates with shape [(H/2)*(W/2), 3].

    diffusers FluxTransformer2DModel.forward expects 2D ids (no batch dim) in
    recent versions; the older 3D form is deprecated.
    """
    h_packed = latent_height // 2
    w_packed = latent_width // 2
    latent_image_ids = torch.zeros(h_packed, w_packed, 3)
    latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(h_packed)[:, None]
    latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(w_packed)[None, :]
    latent_image_ids = latent_image_ids.reshape(h_packed * w_packed, 3)
    return latent_image_ids.to(device=device, dtype=dtype)


# ---------------------------------------------------------------------------
# Text encoding functions
# ---------------------------------------------------------------------------

@torch.no_grad()
def encode_prompt_clip_flux(
    text_encoder: CLIPTextModel,
    tokenizer: CLIPTokenizer,
    prompts: list[str],
    device: torch.device,
    max_length: int = 77,
) -> torch.Tensor:
    """Encode prompts via CLIP and return pooled output [B, 768].

    FLUX uses ONLY the pooler_output from CLIP (no sequence-level hidden states
    from CLIP, unlike SD3 which concatenates CLIP hidden states with T5).
    """
    text_inputs = tokenizer(
        prompts,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        return_overflowing_tokens=False,
        return_length=False,
        return_tensors="pt",
    )
    outputs = text_encoder(
        text_inputs.input_ids.to(device),
        output_hidden_states=False,
    )
    pooled_output = outputs.pooler_output
    return pooled_output.to(dtype=text_encoder.dtype, device=device)


@torch.no_grad()
def encode_prompt_t5_flux(
    text_encoder: T5EncoderModel,
    tokenizer: T5TokenizerFast,
    prompts: list[str],
    device: torch.device,
    max_length: int = 512,
) -> torch.Tensor:
    """Encode prompts via T5 and return hidden states [B, max_length, 4096]."""
    text_inputs = tokenizer(
        prompts,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        return_length=False,
        return_overflowing_tokens=False,
        return_tensors="pt",
    )
    outputs = text_encoder(text_inputs.input_ids.to(device))
    return outputs[0].to(dtype=text_encoder.dtype, device=device)


@torch.no_grad()
def encode_prompts_flux(
    prompts: list[str],
    text_encoders: list,
    tokenizers: list,
    device: torch.device,
    max_sequence_length: int = 512,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode prompts using the full FLUX text encoder stack (CLIP + T5).

    Returns:
        Tuple of (prompt_embeds, pooled_prompt_embeds):
        - prompt_embeds: [B, max_sequence_length, 4096] from T5
        - pooled_prompt_embeds: [B, 768] from CLIP pooler_output
    """
    pooled_embeds = encode_prompt_clip_flux(
        text_encoders[0], tokenizers[0], prompts, device,
    )
    prompt_embeds = encode_prompt_t5_flux(
        text_encoders[1], tokenizers[1], prompts, device,
        max_length=max_sequence_length,
    )
    return prompt_embeds, pooled_embeds


# ---------------------------------------------------------------------------
# Velocity prediction (the FLUX transformer forward dispatch)
# ---------------------------------------------------------------------------

def _build_guidance_tensor(
    batch_size: int,
    device: torch.device,
    guidance_value: float,
) -> torch.Tensor:
    """Build the distilled-CFG guidance tensor for FLUX.1-dev forward."""
    return torch.full(
        (batch_size,), float(guidance_value), device=device, dtype=torch.float32,
    )


def predict_noise_flux(
    model: nn.Module,
    noisy_latents: torch.Tensor,
    prompt_embeds: torch.Tensor,
    timesteps: torch.Tensor,
    pooled_embeds: torch.Tensor,
    guidance_scale: float = 1.0,
    uncond_text_embeddings: torch.Tensor | None = None,
    uncond_pooled_prompt_embeds: torch.Tensor | None = None,
    num_train_timesteps: int = 1000,
) -> torch.Tensor:
    """Run a forward pass through the FLUX transformer and return [B, C, H, W] velocity.

    Guidance handling (the key FLUX-specific dispatch):

    1) ``model.config.guidance_embeds == True`` (FLUX.1-dev architecture with
       guidance_in layer): the CFG is *distilled* into the model. Pass
       ``guidance = [guidance_scale] * B`` as a scalar input. A SINGLE forward
       pass is performed; uncond_* arguments are IGNORED (no traditional CFG).
       For DMD: teacher gets guidance_scale=3.5; generator/fake should pass 1.0
       as a "no-CFG" dummy (the value 1.0 sits at the edge of the FLUX.1-dev
       training distribution).

    2) ``model.config.guidance_embeds == False`` (FLUX.1-schnell or
       wo-guidance-embed variant): ``guidance=None`` is passed. When
       ``guidance_scale > 1.0`` AND ``uncond_*`` embeddings are provided, a
       traditional CFG (uncond+cond doubling) is applied; otherwise a single
       forward pass is performed.

    Args:
        model: FluxTransformer2DModel instance (possibly FSDP-wrapped).
        noisy_latents: VAE-latent-domain tensor [B, 16, H, W] (NOT packed).
        prompt_embeds: T5 sequence embeddings [B, max_seq_len, 4096].
        timesteps: Diffusion timesteps in the [0, num_train_timesteps] range [B].
        pooled_embeds: CLIP pooled embeddings [B, 768].
        guidance_scale: For distilled mode, the scalar input. For schnell mode,
            the classical CFG scale (>1.0 to enable CFG, 1.0 to disable).
        uncond_text_embeddings: Unconditional T5 embeddings (only used in
            schnell-mode CFG).
        uncond_pooled_prompt_embeds: Unconditional pooled CLIP embeddings (only
            used in schnell-mode CFG).
        num_train_timesteps: Scheduler training timesteps (FLUX uses 1000).

    Returns:
        Predicted velocity [B, 16, H, W] in the latent domain.
    """
    batch_size, _, latent_height, latent_width = noisy_latents.shape
    device = noisy_latents.device

    model_config = _resolve_model_config(model)
    model_dtype = _resolve_model_dtype(model, fallback=noisy_latents.dtype)
    use_distilled = bool(getattr(model_config, "guidance_embeds", False))

    # Pack VAE-latent-domain [B, C=16, H, W] -> [B, (H/2)*(W/2), C*4=64] for transformer input.
    packed_noisy = pack_flux_latents(noisy_latents).to(model_dtype)

    # img_ids: 2D coordinates over the packed token grid.
    latent_image_ids = prepare_flux_latent_image_ids(
        latent_height=latent_height,
        latent_width=latent_width,
        device=device,
        dtype=model_dtype,
    )
    # txt_ids: FLUX uses all-zero text token coordinates (2D).
    txt_ids = torch.zeros(
        prompt_embeds.shape[1], 3, device=device, dtype=model_dtype,
    )

    # Normalize timestep to [0, 1] before feeding the transformer (FLUX
    # convention: forward() multiplies by 1000 internally).
    model_timestep = timesteps.to(device=device, dtype=model_dtype) / float(num_train_timesteps)

    def _forward_once(
        cond_prompt_embeds: torch.Tensor,
        cond_pooled_embeds: torch.Tensor,
        guidance_value: float | None,
    ) -> torch.Tensor:
        if guidance_value is None:
            guidance = None
        else:
            guidance = _build_guidance_tensor(
                batch_size=batch_size, device=device, guidance_value=guidance_value,
            )
        pred = model(
            hidden_states=packed_noisy,
            timestep=model_timestep,
            guidance=guidance,
            pooled_projections=cond_pooled_embeds.to(model_dtype),
            encoder_hidden_states=cond_prompt_embeds.to(model_dtype),
            txt_ids=txt_ids,
            img_ids=latent_image_ids,
            return_dict=False,
        )[0]
        return pred

    if use_distilled:
        # FLUX.1-dev path: single forward with distilled guidance scalar.
        # No uncond pass — uncond_* arguments are intentionally ignored.
        pred_tokens = _forward_once(prompt_embeds, pooled_embeds, guidance_scale)
    else:
        # FLUX.1-schnell / wo-guidance-embed path: optional traditional CFG.
        use_traditional_cfg = (
            guidance_scale > 1.0
            and uncond_text_embeddings is not None
            and uncond_pooled_prompt_embeds is not None
        )
        if use_traditional_cfg:
            pred_uncond = _forward_once(
                uncond_text_embeddings, uncond_pooled_prompt_embeds, None,
            )
            pred_cond = _forward_once(prompt_embeds, pooled_embeds, None)
            pred_tokens = pred_uncond + guidance_scale * (pred_cond - pred_uncond)
        else:
            pred_tokens = _forward_once(prompt_embeds, pooled_embeds, None)

    return unpack_flux_latents(
        pred_tokens, latent_height=latent_height, latent_width=latent_width,
    ).to(noisy_latents.dtype)


# ---------------------------------------------------------------------------
# Sigma schedule computation (with FLUX.1-dev dynamic shifting)
# ---------------------------------------------------------------------------

def compute_mu_flux(
    image_seq_len: int,
    base_image_seq_len: int = 256,
    max_image_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
) -> float:
    """Compute mu for FLUX dynamic timestep shifting (linear in image_seq_len).

    Mirrors ``diffusers.pipelines.flux.pipeline_flux.calculate_shift``.

    For FLUX.1-dev defaults (base_image_seq_len=256, max=4096, base_shift=0.5,
    max_shift=1.15):
        - image_seq_len=1024 (512x512 image): mu ≈ 0.630
        - image_seq_len=4096 (1024x1024 image): mu = 1.150
    """
    m = (max_shift - base_shift) / (max_image_seq_len - base_image_seq_len)
    b = base_shift - m * base_image_seq_len
    return float(image_seq_len * m + b)


def _time_shift_exponential(mu: float, t: torch.Tensor) -> torch.Tensor:
    """FLUX standard time shift (matches FlowMatchEulerDiscreteScheduler)."""
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1))


def _time_shift_linear(mu: float, t: torch.Tensor) -> torch.Tensor:
    return mu / (mu + (1 / t - 1))


def compute_sigmas_flux(
    denoising_step_list: list[int],
    scheduler_config: Any,
    image_seq_len: int,
    num_steps: int | None = None,
) -> list[float]:
    """Compute shifted sigmas for a custom FLUX denoising step schedule.

    Mirrors how FluxPipeline calls the scheduler with ``mu`` for dynamic
    shifting. When ``scheduler_config.use_dynamic_shifting`` is True, ``mu`` is
    derived from ``image_seq_len`` and the FLUX exponential time shift is
    applied. Otherwise the static ``shift`` formula is used.

    Args:
        denoising_step_list: Raw denoising steps in the [0, num_train_timesteps]
            range (e.g., [1000, 750, 500, 250] for 4-step distillation).
        scheduler_config: FlowMatchEulerDiscreteScheduler config-like object
            (must expose ``num_train_timesteps``, ``use_dynamic_shifting``,
            ``base_shift``, ``max_shift``, ``base_image_seq_len``,
            ``max_image_seq_len``, ``shift`` and optionally ``time_shift_type``).
        image_seq_len: Number of packed image tokens; for FLUX this equals
            ``(H_latent / 2) * (W_latent / 2)``. For 512x512 with 8x VAE
            downsampling, this is 1024; for 1024x1024 it is 4096.
        num_steps: Optional truncation of ``denoising_step_list``.

    Returns:
        List of shifted sigmas, e.g. [1.0, ~0.83, ~0.59, ~0.31] for 4-step
        FLUX.1-dev at 512 resolution.
    """
    if num_steps is not None:
        denoising_steps = denoising_step_list[:num_steps]
    else:
        denoising_steps = list(denoising_step_list)
    if len(denoising_steps) == 0:
        raise ValueError("denoising_step_list must contain at least one step.")

    num_train_timesteps = int(getattr(scheduler_config, "num_train_timesteps", 1000))
    raw_sigmas = torch.tensor(
        [float(step) / float(num_train_timesteps) for step in denoising_steps],
        dtype=torch.float32,
    )

    use_dynamic_shifting = bool(getattr(scheduler_config, "use_dynamic_shifting", False))
    if use_dynamic_shifting:
        mu = compute_mu_flux(
            image_seq_len=image_seq_len,
            base_image_seq_len=int(getattr(scheduler_config, "base_image_seq_len", 256)),
            max_image_seq_len=int(getattr(scheduler_config, "max_image_seq_len", 4096)),
            base_shift=float(getattr(scheduler_config, "base_shift", 0.5)),
            max_shift=float(getattr(scheduler_config, "max_shift", 1.15)),
        )
        time_shift_type = str(getattr(scheduler_config, "time_shift_type", "exponential")).lower()
        if time_shift_type == "linear":
            shifted_sigmas = _time_shift_linear(mu, raw_sigmas)
        else:
            shifted_sigmas = _time_shift_exponential(mu, raw_sigmas)
    else:
        shift = float(getattr(scheduler_config, "shift", 1.0))
        shifted_sigmas = shift * raw_sigmas / (1 + (shift - 1) * raw_sigmas)

    return [float(s.item()) for s in shifted_sigmas]


# ---------------------------------------------------------------------------
# Main model loading function
# ---------------------------------------------------------------------------

def load_flux_models(config: ModelConfig) -> dict[str, Any]:
    """Load all FLUX.1 pipeline components from a pretrained checkpoint.

    Loads the transformer (with config.json's ``guidance_embeds`` setting,
    typically True for FLUX.1-dev and False for FLUX.1-schnell), VAE, text
    encoders, tokenizers, and scheduler. Models are loaded to CPU first; the
    trainer handles device placement and FSDP/DDP wrapping.

    Args:
        config: Model configuration with pretrained_path and dtype.

    Returns:
        Dict with keys:
        - "transformer": FluxTransformer2DModel
        - "vae": AutoencoderKL
        - "text_encoders": [clip, t5]
        - "tokenizers": [clip_tokenizer, t5_tokenizer]
        - "scheduler": FlowMatchEulerDiscreteScheduler
    """
    path = config.pretrained_path
    dtype = _get_torch_dtype(config.dtype)
    if is_main_process():
        logger.info(f"Loading FLUX.1 models from: {path} (dtype={config.dtype})")

    # Attention implementation: prefer flash-attn 2 → SDPA → eager.
    attn_implementation = "eager"
    if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
        try:
            import flash_attn  # noqa: F401
            attn_implementation = "flash_attention_2"
            if is_main_process():
                logger.info("  Flash Attention 2 available, enabling for transformer")
        except ImportError:
            attn_implementation = "sdpa"
            if is_main_process():
                logger.info("  Using PyTorch SDPA attention (flash_attn not found)")
    else:
        if is_main_process():
            logger.warning("  Falling back to eager attention (PyTorch too old for SDPA)")

    with fast_init(torch.device("cpu")):
        transformer = FluxTransformer2DModel.from_pretrained(
            path, subfolder="transformer", torch_dtype=dtype,
            attn_implementation=attn_implementation,
        )
        if is_main_process():
            n_params = sum(p.numel() for p in transformer.parameters()) / 1e6
            guidance_embeds = bool(getattr(transformer.config, "guidance_embeds", False))
            logger.info(
                f"  Transformer loaded: {n_params:.1f}M params, "
                f"guidance_embeds={guidance_embeds}"
            )

        vae = AutoencoderKL.from_pretrained(path, subfolder="vae", torch_dtype=dtype)
        if is_main_process():
            logger.info("  VAE loaded")

        text_encoder_1 = CLIPTextModel.from_pretrained(
            path, subfolder="text_encoder", torch_dtype=dtype,
        )
        text_encoder_2 = T5EncoderModel.from_pretrained(
            path, subfolder="text_encoder_2", torch_dtype=dtype,
        )
        if is_main_process():
            logger.info("  Text encoders loaded (CLIP + T5)")

    tokenizer_1 = CLIPTokenizer.from_pretrained(path, subfolder="tokenizer")
    tokenizer_2 = T5TokenizerFast.from_pretrained(path, subfolder="tokenizer_2")
    if is_main_process():
        logger.info("  Tokenizers loaded")

    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(path, subfolder="scheduler")
    if is_main_process():
        logger.info(
            f"  Scheduler loaded (FlowMatchEulerDiscrete, "
            f"shift={getattr(scheduler.config, 'shift', None)}, "
            f"use_dynamic_shifting={getattr(scheduler.config, 'use_dynamic_shifting', None)})"
        )

    return {
        "transformer": transformer,
        "vae": vae,
        "text_encoders": [text_encoder_1, text_encoder_2],
        "tokenizers": [tokenizer_1, tokenizer_2],
        "scheduler": scheduler,
    }
