"""Diffusers patches for RTDMD.

Contains modified scheduler/pipeline functions for SDE sampling with
log-probability tracking, used by GRPO-style training.

Public API:

- :func:`sde_step_with_logprob` -- one SDE Euler step that also returns the
  log-probability of the sampled next state (see :mod:`rtdmd.diffusers_patch.sde_with_logprob`).
"""

from rtdmd.diffusers_patch.sde_with_logprob import sde_step_with_logprob

__all__ = ["sde_step_with_logprob"]
