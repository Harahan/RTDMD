"""
SDE reverse step with log-probability computation for Flow Matching models.

Ported from grpo/diffusers_patch/sd3_sde_with_logprob.py.
SDE step with per-element log-probability tracking for PPO-style updates.

Provides an independent function (not a scheduler method) that:
1. Looks up sigma per-sample via index_for_timestep() — no shared _step_index state
2. Computes one SDE reverse step: x_{t-1} = mean + std * noise
3. Computes the Gaussian log-probability of x_{t-1} under the transition kernel

Two modes:
- "sde": Standard SDE reversal with tunable noise_level ∈ (0, 1].
  WARNING: noise_level=0 causes division by zero — use CPS mode with windowed SDE.
- "cps": Coefficients-Preserving Sampling. Safe with noise_level=0 (no division).

All computation is done in float32 to avoid bf16/fp16 overflow.
"""

from __future__ import annotations

import math
from typing import Optional, Union

import torch
from diffusers.utils.torch_utils import randn_tensor
from diffusers.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)


def sde_step_with_logprob(
    scheduler: FlowMatchEulerDiscreteScheduler,
    model_output: torch.FloatTensor,
    timestep: Union[float, torch.FloatTensor],
    sample: torch.FloatTensor,
    noise_level: float = 0.7,
    prev_sample: Optional[torch.FloatTensor] = None,
    generator: Optional[torch.Generator] = None,
    sde_type: str = "sde",
    variance_noise: Optional[torch.FloatTensor] = None,
    return_unreduced: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Predict the sample from the previous timestep by reversing the SDE.

    This function propagates the flow process from the learned model outputs
    (most often the predicted velocity) while computing the exact log-probability
    of the resulting sample under the Gaussian transition kernel.

    Unlike scheduler.step(), this function is stateless — it looks up sigma
    per-sample via index_for_timestep() instead of relying on the shared
    _step_index counter. This makes it safe for parallel / out-of-order calls.

    Args:
        scheduler: A FlowMatchEulerDiscreteScheduler with timesteps already set.
        model_output: Direct output (velocity prediction) from the model. [B, C, H, W]
        timestep: Current discrete timestep(s). Scalar or [B] tensor.
        sample: Current noisy sample x_t. [B, C, H, W]
        noise_level: Controls stochasticity of the SDE. 0 = ODE (deterministic),
            1 = full SDE. Only safe with sde_type="cps" when noise_level=0.
        prev_sample: If provided, use this as x_{t-1} instead of sampling new noise.
            Used during training to compute log_prob of a stored trajectory.
        generator: Optional torch.Generator for reproducible noise sampling.
        sde_type: "sde" (standard) or "cps" (Coefficients-Preserving Sampling).
        variance_noise: Optional pre-computed noise tensor [B, C, H, W] for the
            stochastic component. When provided, this noise is used instead of
            sampling fresh random noise internally. Useful for injecting
            deterministic (e.g., prompt-seeded) noise to ensure reproducibility
            across ranks. Only used when prev_sample is None (sampling stage);
            ignored when prev_sample is provided (training stage).
        return_unreduced: If True, return log_prob as [B, C, H, W] without
            reducing over spatial dimensions. Useful for fine-grained log-prob
            aggregation where the caller applies custom group reduction.
            Default False returns scalar [B] (backward compatible).

    Returns:
        Tuple of:
        - prev_sample: The denoised sample x_{t-1}. [B, C, H, W]
        - log_prob: Log-probability of prev_sample under the transition kernel. [B]
        - prev_sample_mean: Deterministic mean of the transition. [B, C, H, W]
        - std_dev_t: Standard deviation of the transition noise. [B, 1, 1, 1]
    """
    # Cast to float32 to avoid bf16 overflow in intermediate computations
    model_output = model_output.float()
    sample = sample.float()
    if prev_sample is not None:
        prev_sample = prev_sample.float()

    # --- Look up sigma for each sample independently ---
    step_index = [scheduler.index_for_timestep(t) for t in timestep]
    prev_step_index = [s + 1 for s in step_index]

    ndim = len(sample.shape)
    sigma = scheduler.sigmas[step_index].view(-1, *([1] * (ndim - 1)))
    sigma_prev = scheduler.sigmas[prev_step_index].view(-1, *([1] * (ndim - 1)))
    sigma_max = scheduler.sigmas[1].item()
    dt = sigma_prev - sigma  # negative (going from t to t-1)

    if sde_type == "sde":
        # Standard SDE reversal
        # std_dev_t = sqrt(sigma / (1 - sigma)) * noise_level
        # Avoid sigma=1 singularity by clamping
        std_dev_t = torch.sqrt(
            sigma / (1 - torch.where(sigma == 1, sigma_max, sigma))
        ) * noise_level

        # Deterministic mean prediction
        prev_sample_mean = (
            sample * (1 + std_dev_t ** 2 / (2 * sigma) * dt)
            + model_output * (1 + std_dev_t ** 2 * (1 - sigma) / (2 * sigma)) * dt
        )

        # Sample or use provided prev_sample
        if prev_sample is None:
            if variance_noise is None:
                variance_noise = randn_tensor(
                    model_output.shape,
                    generator=generator,
                    device=model_output.device,
                    dtype=model_output.dtype,
                )
            prev_sample = prev_sample_mean + std_dev_t * torch.sqrt(-1 * dt) * variance_noise

        # Gaussian log-probability
        log_prob = (
            -((prev_sample.detach() - prev_sample_mean) ** 2)
            / (2 * ((std_dev_t * torch.sqrt(-1 * dt)) ** 2))
            - torch.log(std_dev_t * torch.sqrt(-1 * dt))
            - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
        )

    elif sde_type == "cps":
        # Coefficients-Preserving Sampling (direct sin/cos form)
        rho = sigma_prev * math.sin(noise_level * math.pi / 2)
        eps_coeff = sigma_prev * math.cos(noise_level * math.pi / 2)
        std_dev_t = rho  # alias for log_prob and return value

        pred_original_sample = sample - sigma * model_output  # predicted x_0
        noise_estimate = sample + model_output * (1 - sigma)  # predicted x_1

        prev_sample_mean = (
            pred_original_sample * (1 - sigma_prev)
            + noise_estimate * eps_coeff
        )

        if prev_sample is None:
            if variance_noise is None:
                variance_noise = randn_tensor(
                    model_output.shape,
                    generator=generator,
                    device=model_output.device,
                    dtype=model_output.dtype,
                )
            prev_sample = prev_sample_mean + rho * variance_noise

        # Simplified log-prob (constants removed — safe with noise_level=0)
        log_prob = -((prev_sample.detach() - prev_sample_mean) ** 2)

    else:
        raise ValueError(f"Unknown sde_type: {sde_type}. Expected 'sde' or 'cps'.")

    # Reduce log_prob: mean over all dims except batch (unless unreduced requested)
    if not return_unreduced:
        log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))

    return prev_sample, log_prob, prev_sample_mean, std_dev_t
