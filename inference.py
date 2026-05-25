"""
RTDMD Inference CLI.

Drives ``rtdmd.inference.RTDMDInference`` from a YAML config under
``configs/inference/``. Three LoRA regimes are toggled by the YAML's
``lora_paths`` (0/1/2 entries). Reward eval is toggled by ``eval_reward``.

Usage:
    # Plain inference, single GPU
    python inference.py configs/inference/sd35m.yaml \\
        --prompt "a cute cat sitting on a windowsill"

    # Distilled + RL LoRAs (the YAML default) on 8 GPUs with reward eval
    torchrun --nproc_per_node=8 inference.py configs/inference/sd35m.yaml \\
        --override distributed=true --override eval_reward=true

    # No LoRA, plain pretrained model
    python inference.py configs/inference/sd35m.yaml --override lora_paths=

    # Override individual fields with dot-notation
    python inference.py configs/inference/sd35m.yaml \\
        --override cps_eta=0.0 --override seed=123 --prompt "a cat"
"""

import argparse
import logging

from rtdmd.inference import InferenceConfig, RTDMDInference

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RTDMD Inference")
    parser.add_argument("config", type=str, help="Path to inference YAML config")
    parser.add_argument(
        "--override",
        nargs="+",
        action="append",
        default=[],
        help=(
            "Config overrides in key=value form. Repeatable, e.g. "
            "--override eval_reward=true --override seed=123"
        ),
    )
    parser.add_argument(
        "--prompt",
        action="append",
        default=[],
        dest="cli_prompts",
        help="Inline prompt (repeatable: --prompt 'a cat' --prompt 'a dog')",
    )
    parser.add_argument(
        "--prompt_file",
        type=str,
        default="",
        help="Path to a .txt/.json/.jsonl prompt file (overrides YAML prompt_file).",
    )
    return parser.parse_args()


def parse_overrides(override_list: list[str]) -> dict:
    """Parse ['key=value', ...] into a dict, coercing int/float/bool literals."""
    overrides: dict = {}
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


def main() -> None:
    args = parse_args()

    override_items: list[str] = []
    for item in args.override:
        if isinstance(item, list):
            override_items.extend(item)
        else:
            override_items.append(item)

    overrides = parse_overrides(override_items) if override_items else None
    config = InferenceConfig.from_yaml(args.config, overrides=overrides)

    if args.cli_prompts:
        config.prompts = args.cli_prompts
    if args.prompt_file:
        config.prompt_file = args.prompt_file
    # CLI prompts override the YAML's prompt source: without this flip,
    # `eval_prompt_source="dataset"` (the default in configs/inference/*) would
    # keep loading the bundled dataset and ignore --prompt/--prompt_file.
    if args.cli_prompts or args.prompt_file:
        config.eval_prompt_source = "prompt_file"

    config.validate()

    engine = RTDMDInference(config)
    engine.setup()
    engine.run()


if __name__ == "__main__":
    main()
