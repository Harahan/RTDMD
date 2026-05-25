"""
FLUX.2 Klein model backend for RTDMD.

This module provides a model-backend interface parallel to sd35.py:
- load_flux2_klein_models
- encode_prompts_flux2_klein
- predict_noise_flux2_klein
- compute_sigmas_flux2_klein

The implementation follows diffusers' Flux2KleinPipeline and training examples.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import torch
import torch.nn as nn
from diffusers import (
    AutoencoderKLFlux2,
    FlowMatchEulerDiscreteScheduler,
    Flux2Transformer2DModel,
)
from transformers import Qwen2TokenizerFast, Qwen3ForCausalLM

from rtdmd.config import ModelConfig
from rtdmd.parallel.utils import is_main_process
from rtdmd.utils.fast_init import fast_init

logger = logging.getLogger(__name__)


_DTYPE_MAP = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}


def _get_torch_dtype(dtype_str: str) -> torch.dtype:
    if dtype_str not in _DTYPE_MAP:
        raise ValueError(f"Unknown dtype: {dtype_str}. Choose from: {list(_DTYPE_MAP.keys())}")
    return _DTYPE_MAP[dtype_str]


def _prepare_text_ids(prompt_embeds: torch.Tensor) -> torch.Tensor:
    """Create Flux2 text token coordinates with shape [B, L, 4]."""
    batch_size, seq_len, _ = prompt_embeds.shape
    out_ids = []
    for _ in range(batch_size):
        t = torch.arange(1)
        h = torch.arange(1)
        w = torch.arange(1)
        l = torch.arange(seq_len)
        coords = torch.cartesian_prod(t, h, w, l)
        out_ids.append(coords)
    return torch.stack(out_ids)


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


def _prepare_latent_ids(latents: torch.Tensor) -> torch.Tensor:
    """Create Flux2 latent token coordinates with shape [B, H*W, 4]."""
    batch_size, _, height, width = latents.shape
    t = torch.arange(1, device=latents.device)
    h = torch.arange(height, device=latents.device)
    w = torch.arange(width, device=latents.device)
    l = torch.arange(1, device=latents.device)
    latent_ids = torch.cartesian_prod(t, h, w, l)
    latent_ids = latent_ids.unsqueeze(0).expand(batch_size, -1, -1)
    return latent_ids


def pack_flux2_latents(latents: torch.Tensor) -> torch.Tensor:
    """Pack [B, C, H, W] -> [B, H*W, C] for Flux2 transformer."""
    batch_size, num_channels, height, width = latents.shape
    return latents.reshape(batch_size, num_channels, height * width).permute(0, 2, 1).contiguous()


def unpack_flux2_latents(tokens: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Unpack [B, H*W, C] -> [B, C, H, W] for Flux2 latent space."""
    batch_size, num_tokens, num_channels = tokens.shape
    expected_tokens = height * width
    if num_tokens != expected_tokens:
        raise ValueError(
            f"Token count mismatch when unpacking Flux2 latents: got {num_tokens}, expected {expected_tokens}"
        )
    return tokens.permute(0, 2, 1).reshape(batch_size, num_channels, height, width).contiguous()


def unpatchify_flux2_latents(latents: torch.Tensor) -> torch.Tensor:
    """Undo Flux2 2x2 latent patchification before VAE decode."""
    batch_size, num_channels, height, width = latents.shape
    latents = latents.reshape(batch_size, num_channels // 4, 2, 2, height, width)
    latents = latents.permute(0, 1, 4, 2, 5, 3)
    latents = latents.reshape(batch_size, num_channels // 4, height * 2, width * 2)
    return latents


@torch.no_grad()
def encode_prompts_flux2_klein(
    prompts: list[str],
    text_encoder: Qwen3ForCausalLM,
    tokenizer: Qwen2TokenizerFast,
    device: torch.device,
    max_sequence_length: int = 512,
    hidden_states_layers: tuple[int, int, int] = (9, 18, 27),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode prompts into Flux2 Qwen embeddings and token ids."""
    all_input_ids = []
    all_attention_masks = []

    for single_prompt in prompts:
        messages = [{"role": "user", "content": single_prompt}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        tokenized = tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_sequence_length,
        )
        all_input_ids.append(tokenized["input_ids"])
        all_attention_masks.append(tokenized["attention_mask"])

    input_ids = torch.cat(all_input_ids, dim=0).to(device)
    attention_mask = torch.cat(all_attention_masks, dim=0).to(device)

    outputs = text_encoder(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=False,
    )
    stacked_hidden = torch.stack([outputs.hidden_states[k] for k in hidden_states_layers], dim=1)
    stacked_hidden = stacked_hidden.to(dtype=text_encoder.dtype, device=device)

    batch_size, num_layers, seq_len, hidden_dim = stacked_hidden.shape
    prompt_embeds = stacked_hidden.permute(0, 2, 1, 3).reshape(batch_size, seq_len, num_layers * hidden_dim)
    text_ids = _prepare_text_ids(prompt_embeds).to(device)
    return prompt_embeds, text_ids


def _build_guidance_tensor(
    model: nn.Module,
    batch_size: int,
    device: torch.device,
    guidance_value: float,
) -> torch.Tensor | None:
    model_config = _resolve_model_config(model)
    if model_config is None or not bool(getattr(model_config, "guidance_embeds", False)):
        return None
    return torch.full((batch_size,), float(guidance_value), device=device, dtype=torch.float32)


def predict_noise_flux2_klein(
    model: nn.Module,
    noisy_latents: torch.Tensor,
    prompt_embeds: torch.Tensor,
    timesteps: torch.Tensor,
    text_ids: torch.Tensor,
    guidance_scale: float = 1.0,
    uncond_prompt_embeds: torch.Tensor | None = None,
    uncond_text_ids: torch.Tensor | None = None,
    num_train_timesteps: int = 1000,
) -> torch.Tensor:
    """Forward Flux2 transformer with optional CFG and return [B,C,H,W] velocity."""
    use_cfg = guidance_scale > 1.0
    if use_cfg:
        if uncond_prompt_embeds is None or uncond_text_ids is None:
            raise ValueError("CFG for Flux2 requires uncond_prompt_embeds and uncond_text_ids.")

    batch_size, _, height, width = noisy_latents.shape
    latent_ids = _prepare_latent_ids(noisy_latents)
    model_dtype = _resolve_model_dtype(model, fallback=noisy_latents.dtype)
    packed_noisy_latents = pack_flux2_latents(noisy_latents).to(model_dtype)

    model_timestep = timesteps.to(device=noisy_latents.device, dtype=packed_noisy_latents.dtype)
    model_timestep = model_timestep / float(num_train_timesteps)

    def _forward_once(
        cond_prompt_embeds: torch.Tensor,
        cond_text_ids: torch.Tensor,
    ) -> torch.Tensor:
        guidance = _build_guidance_tensor(
            model=model,
            batch_size=batch_size,
            device=noisy_latents.device,
            guidance_value=guidance_scale,
        )
        pred = model(
            hidden_states=packed_noisy_latents,
            timestep=model_timestep,
            guidance=guidance,
            encoder_hidden_states=cond_prompt_embeds.to(device=noisy_latents.device, dtype=model_dtype),
            txt_ids=cond_text_ids.to(device=noisy_latents.device),
            img_ids=latent_ids,
            return_dict=False,
        )[0]
        return pred[:, : packed_noisy_latents.size(1), :]

    if use_cfg:
        pred_uncond = _forward_once(uncond_prompt_embeds, uncond_text_ids)
        pred_cond = _forward_once(prompt_embeds, text_ids)
        pred_tokens = pred_uncond + guidance_scale * (pred_cond - pred_uncond)
    else:
        pred_tokens = _forward_once(prompt_embeds, text_ids)

    return unpack_flux2_latents(pred_tokens, height=height, width=width).to(noisy_latents.dtype)


def compute_empirical_mu(image_seq_len: int, num_steps: int) -> float:
    """Empirical mu function used by Flux2KleinPipeline for dynamic shifting."""
    a1, b1 = 8.73809524e-05, 1.89833333
    a2, b2 = 0.00016927, 0.45666666

    if image_seq_len > 4300:
        return float(a2 * image_seq_len + b2)

    m_200 = a2 * image_seq_len + b2
    m_10 = a1 * image_seq_len + b1
    a = (m_200 - m_10) / 190.0
    b = m_200 - 200.0 * a
    return float(a * num_steps + b)


def _time_shift_exponential(mu: float, sigma: float, t: torch.Tensor) -> torch.Tensor:
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


def _time_shift_linear(mu: float, sigma: float, t: torch.Tensor) -> torch.Tensor:
    return mu / (mu + (1 / t - 1) ** sigma)


def compute_sigmas_flux2_klein(
    denoising_step_list: list[int],
    scheduler_config: Any,
    image_seq_len: int,
    num_steps: int | None = None,
) -> list[float]:
    """Compute Flux2-Klein shifted sigmas for custom denoising step schedules."""
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
        mu = compute_empirical_mu(image_seq_len=image_seq_len, num_steps=len(denoising_steps))
        time_shift_type = str(getattr(scheduler_config, "time_shift_type", "exponential")).lower()
        if time_shift_type == "linear":
            shifted_sigmas = _time_shift_linear(mu, 1.0, raw_sigmas)
        else:
            shifted_sigmas = _time_shift_exponential(mu, 1.0, raw_sigmas)
    else:
        shift = float(getattr(scheduler_config, "shift", 1.0))
        shifted_sigmas = shift * raw_sigmas / (1 + (shift - 1) * raw_sigmas)

    shift_terminal = getattr(scheduler_config, "shift_terminal", None)
    if shift_terminal:
        one_minus_z = 1 - shifted_sigmas
        scale_factor = one_minus_z[-1] / (1 - float(shift_terminal))
        shifted_sigmas = 1 - (one_minus_z / scale_factor)

    return [float(s.item()) for s in shifted_sigmas]


def load_flux2_klein_models(config: ModelConfig) -> dict[str, Any]:
    """Load FLUX.2 Klein components from a pretrained checkpoint directory."""
    path = config.pretrained_path
    dtype = _get_torch_dtype(config.dtype)

    if is_main_process():
        logger.info(f"Loading FLUX.2 Klein models from: {path} (dtype={config.dtype})")

    with fast_init(torch.device("cpu")):
        transformer = Flux2Transformer2DModel.from_pretrained(
            path,
            subfolder="transformer",
            torch_dtype=dtype,
        )
        vae = AutoencoderKLFlux2.from_pretrained(path, subfolder="vae", torch_dtype=dtype)
        text_encoder = Qwen3ForCausalLM.from_pretrained(path, subfolder="text_encoder", torch_dtype=dtype)

    tokenizer = Qwen2TokenizerFast.from_pretrained(path, subfolder="tokenizer")
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(path, subfolder="scheduler")

    if is_main_process():
        logger.info(
            "  Flux2 components loaded: transformer + vae + text_encoder(Qwen3) + tokenizer + scheduler"
        )

    return {
        "transformer": transformer,
        "vae": vae,
        "text_encoders": [text_encoder],
        "tokenizers": [tokenizer],
        "scheduler": scheduler,
    }
