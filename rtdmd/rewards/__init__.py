"""Reward scoring module for RTDMD evaluation and BP / GRPO training.

Public API:

- :class:`MultiScorer`              -- weighted multi-reward dispatcher.
- :func:`set_ckpt_path`             -- set the global directory used to
  locate downloaded scorer checkpoints (DINO, HPSv2, ImageReward, ...).
- :data:`CKPT_PATH`                 -- current value of that directory.

Individual scorer classes (``PickScoreScorer``, ``HPSv2Scorer``,
``AestheticScorer``, ``ClipScorer``, ``ImageRewardScorer``, ``OcrScorer``,
``GenEval2SoftTIFAScorer``) live in their respective submodules and are
usually accessed through :class:`MultiScorer`. Import them directly only when
you need a single backend.

Supported scorer keys (see :class:`MultiScorer`):

- ``pickscore``  -- PickScore v1 (text-image alignment)
- ``hpsv2``      -- Human Preference Score v2
- ``hpsv3``      -- Human Preference Score v3
- ``clipscore``  -- CLIP-based text-image similarity
- ``aesthetic``  -- LAION aesthetic predictor
- ``imagereward``-- ImageReward v1.0
- ``ocr``        -- PaddleOCR-based text rendering accuracy
- ``geneval``    -- Object-detection-based compositional evaluation
- ``geneval2``   -- Qwen3-VL Soft-TIFA compositional evaluation
"""

from rtdmd.rewards.multi_scorer import MultiScorer
from rtdmd.rewards.reward_ckpt_path import CKPT_PATH, set_ckpt_path

__all__ = ["CKPT_PATH", "MultiScorer", "set_ckpt_path"]
