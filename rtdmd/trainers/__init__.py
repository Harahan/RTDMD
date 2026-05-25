"""Public trainer API.

Only the two trainers registered in :data:`main.TRAINER_REGISTRY` are exported:

- :class:`ACDMDTrainer` (``--trainer ac_dmd``): AC-DMD distillation.
- :class:`RTDMDTrainer` (``--trainer rtdmd``):  GRPO + optional AC-DMD / BP
  auxiliary loss on deterministic CPS steps.

All other modules in this package (``base_trainer``, ``dmd_trainer``,
``grpo_trainer``, ``ac_dmd_mixin``, ``bp_mixin``, ``ac_dmd_utils``,
``grpo_utils``) are internal implementation details and should not be imported
from outside the ``rtdmd.trainers`` package.
"""

from rtdmd.trainers.ac_dmd_trainer import ACDMDTrainer
from rtdmd.trainers.rtdmd_trainer import RTDMDTrainer

__all__ = ["ACDMDTrainer", "RTDMDTrainer"]
