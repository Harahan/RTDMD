"""
CPS (Coefficients-Preserving Sampling) Scheduler for RTDMD.

Implements a CPS-style denoising step for flow-matching models with
controllable stochasticity via the ``eta`` parameter:

- eta = 0.0: deterministic (mathematically equivalent to an Euler ODE step)
- eta = 1.0: fully stochastic, equivalent to re-noising from the predicted x0
- 0 < eta < 1: smooth angular interpolation between predicted noise and fresh noise

Flow-matching CPS step:
1. Predict x0 from velocity:   x0  = x_t - sigma * v
2. Recover predicted noise:    eps = x_t + (1-sigma) * v
3. Compute noise std:          rho = sigma_next * sin(eta * pi/2)
4. Compute noise direction:    coeff = sigma_next * cos(eta * pi/2)
5. Step: x_{t-1} = (1-sigma_next)*x0 + coeff*eps + rho*noise

Geometric interpretation: the noise direction smoothly rotates on the unit
sphere from the predicted noise (eta=0) to fresh random noise (eta=1),
scaled by sigma_next.

Reference: CPS (arxiv 2509.05952), adapted to linear flow-matching
interpolation x_t = (1-sigma)*x0 + sigma*eps.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple, Union

import torch

from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteSchedulerOutput,
)


class CPSScheduler(FlowMatchEulerDiscreteScheduler):
    """CPS scheduler adapted for Flow Matching models.

    Extends FlowMatchEulerDiscreteScheduler with a CPS-style step that
    supports controllable stochasticity via the ``eta`` parameter.

    Also overrides set_timesteps to accept pre-shifted sigma schedules
    directly, avoiding the double-shift bug in the parent class.

    Can be used both:
    - In the training backward simulation loop (via manual calls)
    - In diffusers pipelines (as a drop-in scheduler replacement)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._eta: float = 0.0

    def set_eta(self, eta: float) -> None:
        """Set the CPS stochasticity parameter.

        Args:
            eta: Stochasticity level. 0.0 = deterministic (Euler-equivalent),
                 higher values add more fresh noise per step.
        """
        self._eta = eta

    @property
    def eta(self) -> float:
        return self._eta

    def set_timesteps(
        self,
        num_inference_steps: int | None = None,
        device: Union[str, torch.device] | None = None,
        sigmas: Optional[List[float]] = None,
        mu: float | None = None,
    ) -> None:
        """Set timesteps with support for pre-shifted custom sigmas.

        When ``sigmas`` is provided, uses them directly WITHOUT applying the
        shift transform. This avoids the double-shift bug where
        ``FlowMatchEulerDiscreteScheduler.set_timesteps`` re-applies the shift
        to already-shifted sigma values.

        When only ``num_inference_steps`` is provided, falls back to the parent
        implementation (which applies the shift transform from scratch).

        Args:
            num_inference_steps: Number of denoising steps (ignored if sigmas given).
            device: Target device for tensors.
            sigmas: Pre-shifted sigma schedule (e.g., [1.0, 0.9, 0.75, 0.5]).
            mu: Optional dynamic-shift parameter for parent scheduler path
                (used when sigmas is None and use_dynamic_shifting=True).
        """
        if sigmas is not None:
            sigmas = torch.tensor(sigmas, dtype=torch.float32)
            if len(sigmas) == 0 or sigmas[-1] != 0.0:
                sigmas = torch.cat([sigmas, torch.zeros(1)])

            timesteps = sigmas[:-1] * self.config.num_train_timesteps

            self.timesteps = timesteps.to(device=device)
            self.sigmas = sigmas.to(device=device)
            self.num_inference_steps = len(self.timesteps)

            self._step_index = None
            self._begin_index = None
        else:
            if mu is None:
                super().set_timesteps(
                    num_inference_steps=num_inference_steps,
                    device=device,
                )
            else:
                try:
                    super().set_timesteps(
                        num_inference_steps=num_inference_steps,
                        device=device,
                        mu=mu,
                    )
                except TypeError:
                    # Backward compatibility with older diffusers versions
                    # where set_timesteps does not expose `mu`.
                    super().set_timesteps(
                        num_inference_steps=num_inference_steps,
                        device=device,
                    )

    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: Union[float, torch.FloatTensor],
        sample: torch.FloatTensor,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True,
    ) -> Union[FlowMatchEulerDiscreteSchedulerOutput, Tuple]:
        """CPS step adapted for flow-matching velocity parameterization.

        Uses the CPS (Coefficients-Preserving Sampling) formulation where eta
        controls angular interpolation between predicted noise and fresh noise:
            x_{t-1} = (1-sigma_next)*x0
                     + sigma_next * [cos(eta*pi/2)*eps + sin(eta*pi/2)*noise]

        Args:
            model_output: Velocity prediction v from the model.
            timestep: Current timestep (used to look up sigma index).
            sample: Current noisy sample x_t.
            generator: Optional random generator for reproducible noise.
            return_dict: Whether to return a dataclass or tuple.

        Returns:
            Scheduler output with prev_sample.
        """
        if self.step_index is None:
            self._init_step_index(timestep)

        sigma = self.sigmas[self.step_index]
        sigma_next = self.sigmas[self.step_index + 1]

        sample_f32 = sample.to(torch.float32)
        v_pred_f32 = model_output.to(torch.float32)

        x0 = sample_f32 - sigma * v_pred_f32
        eps = sample_f32 + (1.0 - sigma) * v_pred_f32

        if sigma_next == 0.0 or self._eta == 0.0:
            # Deterministic Euler ODE step (equivalent to CPS eta=0).
            prev_sample = sample_f32 + (sigma_next - sigma) * v_pred_f32
        else:
            # CPS noise injection: rho = sigma_next * sin(eta * pi/2)
            rho = sigma_next * math.sin(self._eta * math.pi / 2)
            eps_coeff = sigma_next * math.cos(self._eta * math.pi / 2)

            prev_mean = (1.0 - sigma_next) * x0 + eps_coeff * eps
            noise = torch.randn(
                sample.shape, generator=generator,
                device=sample.device, dtype=torch.float32,
            )
            prev_sample = prev_mean + rho * noise

        prev_sample = prev_sample.to(model_output.dtype)
        self._step_index += 1

        if not return_dict:
            return (prev_sample,)

        return FlowMatchEulerDiscreteSchedulerOutput(prev_sample=prev_sample)
