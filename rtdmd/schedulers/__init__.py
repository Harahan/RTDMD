"""Generator-side backward-simulation schedulers.

Only one scheduler is supported:

- :class:`CPSScheduler` -- Coefficients-Preserving Sampling (stochastic
  re-noise step on the flow-matching forward process).

:func:`get_scheduler` is a thin factory kept for symmetry with the YAML
``dmd.generator_scheduler`` field.
"""

from rtdmd.schedulers.cps_scheduler import CPSScheduler

__all__ = ["CPSScheduler", "get_scheduler"]


def get_scheduler(scheduler_type: str):
    """Return the scheduler class.

    Args:
        scheduler_type: Must be ``"cps"`` (the only supported scheduler).

    Returns:
        The :class:`CPSScheduler` class (not an instance).

    Raises:
        ValueError: If ``scheduler_type`` is anything other than ``"cps"``.
    """
    if scheduler_type.lower() != "cps":
        raise ValueError(
            f"Unknown generator_scheduler '{scheduler_type}'. Must be 'cps'."
        )
    return CPSScheduler
