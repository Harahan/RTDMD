"""
Parameter-level Exponential Moving Average (EMA) wrapper.

Ported from grpo/ema.py. Tracks only trainable parameters (e.g., LoRA
weights) rather than deepcopy-ing the entire model, saving significant GPU memory.

Usage:
    trainable_params = list(filter(lambda p: p.requires_grad, model.parameters()))
    ema = EMAModuleWrapper(trainable_params, decay=0.9, update_step_interval=8, device=device)

    # During training, after each optimizer step:
    ema.step(trainable_params, global_step)

    # For evaluation:
    ema.copy_ema_to(trainable_params, store_temp=True)
    # ... run eval ...
    ema.copy_temp_to(trainable_params)
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch


class EMAModuleWrapper:
    """Parameter-level EMA that shadows only trainable parameters.

    Unlike model-level EMA (deepcopy entire model), this only stores shadow
    copies of requires_grad=True parameters, which is much more memory-efficient
    when using LoRA (few trainable params on a large frozen model).

    Args:
        parameters: Iterable of trainable parameters to track.
        decay: EMA decay rate. Effective decay is min((1+step)/(10+step), decay).
        update_step_interval: Update EMA every N optimizer steps.
        device: Device for shadow parameters (e.g., "cuda" or "cpu").
    """

    def __init__(
        self,
        parameters: Iterable[torch.nn.Parameter],
        decay: float = 0.9999,
        update_step_interval: int = 1,
        device: torch.device | None = None,
    ):
        parameters = list(parameters)
        self.ema_parameters = [p.clone().detach().to(device) for p in parameters]
        self.temp_stored_parameters: list[torch.Tensor] | None = None
        self.decay = decay
        self.update_step_interval = update_step_interval
        self.device = device

    def get_current_decay(self, optimization_step: int) -> float:
        """Warm-up decay: starts small, ramps up to self.decay."""
        return min(
            (1 + optimization_step) / (10 + optimization_step),
            self.decay,
        )

    @torch.no_grad()
    def step(
        self,
        parameters: Iterable[torch.nn.Parameter],
        optimization_step: int,
    ) -> None:
        """Update EMA shadow parameters.

        Only updates every ``update_step_interval`` steps. Uses warm-up decay
        that ramps from ~0.1 to ``self.decay`` over the first ~10 steps.

        Args:
            parameters: Current model trainable parameters (same order as __init__).
            optimization_step: Current global optimizer step count.
        """
        parameters = list(parameters)

        if (optimization_step + 1) % self.update_step_interval != 0:
            return

        one_minus_decay = 1 - self.get_current_decay(optimization_step)

        for ema_param, param in zip(self.ema_parameters, parameters, strict=True):
            if param.requires_grad:
                if ema_param.device == param.device:
                    ema_param.add_(one_minus_decay * (param - ema_param))
                else:
                    # In-place to save memory when devices differ
                    param_copy = param.detach().to(ema_param.device)
                    param_copy.sub_(ema_param)
                    param_copy.mul_(one_minus_decay)
                    ema_param.add_(param_copy)
                    del param_copy

    def to(
        self,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """Move EMA parameters to a different device/dtype."""
        self.device = device
        self.ema_parameters = [
            p.to(device=device, dtype=dtype)
            if p.is_floating_point()
            else p.to(device=device)
            for p in self.ema_parameters
        ]

    def copy_ema_to(
        self,
        parameters: Iterable[torch.nn.Parameter],
        store_temp: bool = True,
    ) -> None:
        """Copy EMA weights into the model (e.g., for evaluation).

        Args:
            parameters: Model trainable parameters to overwrite.
            store_temp: If True, save current model weights for later restoration.
        """
        if store_temp:
            self.temp_stored_parameters = [
                param.detach().cpu() for param in parameters
            ]

        parameters = list(parameters)
        for ema_param, param in zip(self.ema_parameters, parameters, strict=True):
            param.data.copy_(ema_param.to(param.device).data)

    def copy_temp_to(
        self,
        parameters: Iterable[torch.nn.Parameter],
    ) -> None:
        """Restore model weights from the temporary storage.

        Must be called after copy_ema_to(..., store_temp=True).

        Args:
            parameters: Model trainable parameters to restore.
        """
        if self.temp_stored_parameters is None:
            raise RuntimeError("No temporary parameters stored. Call copy_ema_to(store_temp=True) first.")

        for temp_param, param in zip(
            self.temp_stored_parameters, parameters, strict=True
        ):
            param.data.copy_(temp_param.data)

        self.temp_stored_parameters = None

    def state_dict(self) -> dict[str, Any]:
        """Serialize EMA state for checkpointing."""
        return {
            "decay": self.decay,
            "ema_parameters": self.ema_parameters,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore EMA state from checkpoint."""
        self.decay = state_dict.get("decay", self.decay)
        self.ema_parameters = state_dict.get("ema_parameters", self.ema_parameters)
        self.to(self.device)
