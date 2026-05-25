"""
Default reward checkpoint path configuration.

Resolution priority:
1) RTDMD_REWARD_CKPT_PATH environment variable
2) <workspace>/models/reward_ckpts (workspace convention)
3) <workspace>/DiffusionNFT/reward_ckpts (legacy fallback)

Can also be overridden at runtime via EvalConfig.reward_ckpt_path.
"""

import os
from pathlib import Path


def _resolve_default_ckpt_path() -> str:
    env = os.environ.get("RTDMD_REWARD_CKPT_PATH", "").strip()
    if env:
        return os.path.expanduser(env)

    workspace_root = Path(__file__).resolve().parents[3]
    candidates = [
        workspace_root / "models" / "reward_ckpts",
        workspace_root / "DiffusionNFT" / "reward_ckpts",
    ]
    for path in candidates:
        if path.exists():
            return str(path)
    return str(candidates[-1])


CKPT_PATH = _resolve_default_ckpt_path()


def set_ckpt_path(path: str) -> None:
    """Override the global reward checkpoint path."""
    global CKPT_PATH
    if path:
        CKPT_PATH = os.path.expanduser(path)
