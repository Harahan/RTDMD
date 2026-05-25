"""RTDMD: Reward-Tilted Distribution Matching Distillation.

Top-level public API:

- :class:`RTDMDConfig`, :class:`LoRAConfig`, :class:`ModelConfig` --
  configuration dataclasses (see :mod:`rtdmd.config`).
- :class:`ACDMDTrainer`, :class:`RTDMDTrainer` --
  the two trainers registered in ``main.py`` (see :mod:`rtdmd.trainers`).
- :class:`RTDMDInference`, :class:`InferenceConfig` --
  CLI inference engine (see :mod:`rtdmd.inference`).

All other modules (``rtdmd.models``, ``rtdmd.parallel``, ``rtdmd.utils``,
``rtdmd.rewards``, ``rtdmd.schedulers``, ``rtdmd.data``,
``rtdmd.diffusers_patch``) are internal building blocks. Import their public
names from the corresponding subpackage's ``__init__`` when you need them,
or from the submodule for advanced use.
"""

from rtdmd.config import LoRAConfig, ModelConfig, RTDMDConfig
from rtdmd.inference import InferenceConfig, RTDMDInference
from rtdmd.trainers import ACDMDTrainer, RTDMDTrainer

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "ACDMDTrainer",
    "InferenceConfig",
    "LoRAConfig",
    "ModelConfig",
    "RTDMDConfig",
    "RTDMDInference",
    "RTDMDTrainer",
]
