"""
RTDMD Main Entry Point.

Dispatches to the appropriate trainer based on --trainer flag. Supports:
- ac_dmd: AC-DMD distillation (sub-interval objective + optional consistency loss)
- rtdmd:  GRPO + optional AC-DMD / BP auxiliary loss on deterministic CPS steps

Usage:
    # Single-node AC-DMD distillation (SD3.5 Medium)
    torchrun --nproc_per_node=8 main.py \
        configs/cold_start/sd35m.yaml \
        --trainer ac_dmd

    # Multi-node RTDMD (GRPO + AC-DMD aux loss, SD3)
    torchrun --nproc_per_node=8 main.py \
        configs/rtdmd/sd3m.yaml \
        --trainer rtdmd

    # CLI overrides (dot-notation)
    torchrun --nproc_per_node=8 main.py <config.yaml> \
        --override train.seed=123 dmd.fake_update_ratio=10
"""

import argparse

from rtdmd.config import RTDMDConfig
from rtdmd.trainers import ACDMDTrainer, RTDMDTrainer


# Registry of available trainers. Extend this dict when adding new algorithms.
TRAINER_REGISTRY = {
    "ac_dmd": ACDMDTrainer,
    "rtdmd": RTDMDTrainer,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RTDMD Training")
    parser.add_argument("config", type=str, help="Path to YAML config file")
    parser.add_argument(
        "--trainer",
        type=str,
        default="rtdmd",
        choices=list(TRAINER_REGISTRY.keys()),
        help="Trainer type to use (default: rtdmd)",
    )
    parser.add_argument(
        "--override",
        nargs="*",
        default=[],
        help="Config overrides in key=value format, e.g., train.seed=123 dmd.fake_update_ratio=10",
    )
    return parser.parse_args()


def parse_overrides(override_list: list[str]) -> dict:
    """Parse CLI overrides like ['train.seed=123', 'dmd.lr=1e-5'] into a dict."""
    overrides = {}
    for item in override_list:
        if "=" not in item:
            raise ValueError(f"Invalid override format: {item}. Expected key=value.")
        key, value = item.split("=", 1)
        try:
            value = int(value)
        except ValueError:
            try:
                value = float(value)
            except ValueError:
                if value.lower() in ("true", "false"):
                    value = value.lower() == "true"
        overrides[key] = value
    return overrides


def main():
    args = parse_args()

    overrides = parse_overrides(args.override) if args.override else None
    config = RTDMDConfig.from_yaml(args.config, overrides=overrides)

    trainer_cls = TRAINER_REGISTRY[args.trainer]
    trainer = trainer_cls(config)
    trainer.train()


if __name__ == "__main__":
    main()
