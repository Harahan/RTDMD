"""Math helpers for AC-DMD sub-interval training."""

from __future__ import annotations

import math
import torch


def unsqueeze_to_ndim(in_tensor: torch.Tensor, tgt_n_dim: int) -> torch.Tensor:
    """Unsqueeze tensor to a target ndim by appending singleton axes."""
    if in_tensor.ndim >= tgt_n_dim:
        return in_tensor
    return in_tensor[(...,) + (None,) * (tgt_n_dim - in_tensor.ndim)]


def scheduler_transition_step(
    sample: torch.Tensor,
    model_pred: torch.Tensor,
    sigma_from: torch.Tensor,
    sigma_to: torch.Tensor,
    scheduler_type: str,
    cps_eta: float = 0.0,
    noise: torch.Tensor | None = None,
) -> torch.Tensor:
    """One CPS transition step from sigma_from to sigma_to.

    Args:
        sample: Current latent x_{sigma_from}.
        model_pred: Velocity prediction at sigma_from.
        sigma_from: Source sigma, shape [B] or broadcastable.
        sigma_to: Target sigma, shape [B] or broadcastable.
        scheduler_type: Must be "cps" (the only supported scheduler).
        cps_eta: CPS stochasticity. 0.0 is deterministic (Euler-equivalent).
        noise: Optional external noise for the stochastic part.

    Returns:
        Latent x_{sigma_to}.
    """
    stype = scheduler_type.lower()
    if stype != "cps":
        raise ValueError(
            f"Unknown scheduler type {scheduler_type!r}. Only 'cps' is supported."
        )

    sample_f32 = sample.float()
    model_pred_f32 = model_pred.float()
    sigma_from_nd = unsqueeze_to_ndim(sigma_from.float(), sample.ndim)
    sigma_to_nd = unsqueeze_to_ndim(sigma_to.float(), sample.ndim)

    if cps_eta <= 0.0:
        out = sample_f32 + (sigma_to_nd - sigma_from_nd) * model_pred_f32
        return out.to(sample.dtype)

    x0 = sample_f32 - sigma_from_nd * model_pred_f32
    eps = sample_f32 + (1.0 - sigma_from_nd) * model_pred_f32
    if noise is None:
        noise = torch.randn_like(sample_f32)
    # eta interpolation matches CPSScheduler.step implementation.
    theta = float(cps_eta) * math.pi / 2.0
    rho = sigma_to_nd * math.sin(theta)
    eps_coeff = sigma_to_nd * math.cos(theta)
    out = (1.0 - sigma_to_nd) * x0 + eps_coeff * eps + rho * noise.float()
    return out.to(sample.dtype)


def flow_forward_v2(
    sigma_start: torch.Tensor,
    sigma_end: torch.Tensor,
    x_start: torch.Tensor,
    noise: torch.Tensor,
) -> torch.Tensor:
    """Forward noising from x_s at sigma_start to x_t at sigma_end."""
    s = sigma_start
    t = sigma_end
    a_ts = (1.0 - t) / (1.0 - s)
    sigma_sq = torch.clamp(t**2 - a_ts**2 * s**2, min=1e-12)
    sigma_ts = torch.sqrt(sigma_sq)
    a_ts = unsqueeze_to_ndim(a_ts, x_start.ndim)
    sigma_ts = unsqueeze_to_ndim(sigma_ts, x_start.ndim)
    return a_ts * x_start + sigma_ts * noise


def flow_forward_target_v2(
    sigma_start: torch.Tensor,
    sigma_end: torch.Tensor,
    x_start: torch.Tensor,
    noise: torch.Tensor,
    move_scale_out: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Theoretical sub-interval target used by AC-DMD fake-score training."""
    s = sigma_start
    t = sigma_end
    aa = -1.0 / (1.0 - s)
    a_ts = (1.0 - t) / (1.0 - s)
    sigma_sq = torch.clamp(t**2 - a_ts**2 * s**2, min=1e-12)
    sigma_ts = torch.sqrt(sigma_sq)
    if not move_scale_out:
        bb = (t + (1.0 - t) / (1.0 - s) ** 2 * s**2) / sigma_ts
        aa = unsqueeze_to_ndim(aa, x_start.ndim)
        bb = unsqueeze_to_ndim(bb, x_start.ndim)
        target = aa * x_start + bb * noise
        scale = torch.ones_like(aa)
        return target, scale

    scale = sigma_ts
    bb = t + (1.0 - t) / (1.0 - s) ** 2 * s**2
    aa = unsqueeze_to_ndim(aa * scale, x_start.ndim)
    bb = unsqueeze_to_ndim(bb, x_start.ndim)
    target = aa * x_start + bb * noise
    return target, unsqueeze_to_ndim(scale, x_start.ndim)


def compute_inverse_scale_weight(
    scale: torch.Tensor,
    clamp_min: float,
    clamp_max: float,
) -> torch.Tensor:
    """Compute clamped inverse-square scale weights: clamp(1/scale^2)."""
    weight = 1.0 / torch.clamp(scale.float() ** 2, min=1e-12)
    weight = torch.clamp(weight, min=clamp_min, max=clamp_max)
    return torch.nan_to_num(
        weight,
        nan=0.0,
        posinf=clamp_max,
        neginf=clamp_min,
    )
