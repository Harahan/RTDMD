"""Prompt datasets and dataloaders for RTDMD training.

Public API:

- :class:`PromptDataset` -- text-prompt-only dataset (no real images).
- :func:`create_prompt_dataloader` -- helper that builds a distributed
  ``DataLoader`` from a prompt file.
"""

from rtdmd.data.prompt_dataset import PromptDataset, create_prompt_dataloader

__all__ = ["PromptDataset", "create_prompt_dataloader"]
