"""
ImageReward v1.0 scorer.

"""

import importlib
import os
import torch
from rtdmd.rewards.reward_ckpt_path import CKPT_PATH


def _load_imagereward_module():
    """Load ImageReward with compatibility patch for new transformers versions."""
    try:
        from transformers import modeling_utils as hf_modeling_utils
    except Exception:
        hf_modeling_utils = None

    if hf_modeling_utils is not None:
        try:
            from transformers import pytorch_utils as hf_pytorch_utils
        except Exception:
            hf_pytorch_utils = None
        if hf_pytorch_utils is not None:
            for symbol_name in (
                "apply_chunking_to_forward",
                "find_pruneable_heads_and_indices",
                "prune_linear_layer",
            ):
                if not hasattr(hf_modeling_utils, symbol_name) and hasattr(hf_pytorch_utils, symbol_name):
                    # ImageReward imports these symbols from the old module path.
                    setattr(hf_modeling_utils, symbol_name, getattr(hf_pytorch_utils, symbol_name))

    return importlib.import_module("ImageReward")


class ImageRewardScorer(torch.nn.Module):
    def __init__(self, device="cuda", dtype=torch.float32):
        super().__init__()
        self.device = torch.device(device)
        self.dtype = dtype
        rm_module = _load_imagereward_module()
        self.model = (
            rm_module.load(
                "ImageReward-v1.0",
                device=str(self.device),
                # Keep ImageReward cache/checkpoints under unified reward_ckpt_path.
                download_root=os.path.join(os.path.expanduser(CKPT_PATH), "ImageReward"),
            )
            .eval()
            .to(dtype=dtype)
        )
        self.model.requires_grad_(False)

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        device = kwargs.get("device")
        if device is None and args:
            first = args[0]
            if isinstance(first, (str, int, torch.device)):
                device = first
        if device is not None:
            self.device = torch.device(device)
            if hasattr(self.model, "to"):
                self.model.to(self.device)
            if hasattr(self.model, "device"):
                # ImageReward.inference_rank relies on this attribute to place inputs.
                self.model.device = str(self.device)
        return self

    @torch.no_grad()
    def __call__(self, prompts, images):
        if hasattr(self.model, "device"):
            # Keep inference input placement aligned with model weights.
            self.model.device = str(self.device)
        _, rewards = self.model.inference_rank(prompts, images)
        rewards = torch.diagonal(
            torch.as_tensor(rewards, device=self.device).reshape(len(prompts), len(prompts)),
            0,
        )
        return rewards.contiguous()
