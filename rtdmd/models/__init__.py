"""Model loaders and per-family helpers for the supported backbones.

Three model families are supported, each with the same surface
(``load_*_models``, ``encode_prompts_*``, ``predict_noise_*``,
``compute_sigmas_*``):

- :mod:`rtdmd.models.sd35`         -- Stable Diffusion 3 / 3.5
- :mod:`rtdmd.models.flux`         -- FLUX.1
- :mod:`rtdmd.models.flux2_klein`  -- FLUX.2 (Klein variant)

For convenience the three top-level *loaders* are re-exported here. For the
detailed per-family helpers import directly from the corresponding submodule.
"""

from rtdmd.models.flux import load_flux_models
from rtdmd.models.flux2_klein import load_flux2_klein_models
from rtdmd.models.sd35 import load_sd35_models

__all__ = [
    "load_flux2_klein_models",
    "load_flux_models",
    "load_sd35_models",
]
