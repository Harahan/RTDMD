"""
RTDMD Configuration System.

Pure dataclass-based configuration with YAML loading. Designed for extensibility:
- New model types (FLUX, SDXL, video models) add fields to ModelConfig
- New distillation algorithms add their own config dataclass

Usage:
    config = RTDMDConfig.from_yaml("configs/cold_start/sd35m.yaml")
    config = RTDMDConfig.from_yaml(path, overrides={"train.seed": 123})
"""

from __future__ import annotations

import copy
import os
import yaml
from dataclasses import dataclass, field, fields, asdict
from typing import Any


def _merge_dict(base: dict, override: dict) -> dict:
    """Recursively merge override dict into base dict."""
    result = copy.deepcopy(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _merge_dict(result[k], v)
        else:
            result[k] = v
    return result


def _apply_flat_overrides(d: dict, overrides: dict[str, Any]) -> dict:
    """Apply dot-notation overrides like {"train.seed": 123} to a nested dict."""
    for key, value in overrides.items():
        parts = key.split(".")
        target = d
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = value
    return d


def _dataclass_from_dict(cls, data: Any):
    """Recursively instantiate a dataclass from a dict, ignoring unknown keys."""
    if not isinstance(data, dict):
        return data
    field_names = {f.name for f in fields(cls)}
    field_types = {f.name: f.type for f in fields(cls)}
    kwargs = {}
    for name in field_names:
        if name not in data:
            continue
        value = data[name]
        ftype = field_types[name]
        if isinstance(ftype, str):
            ftype = eval(ftype)
        if isinstance(value, dict) and hasattr(ftype, "__dataclass_fields__"):
            kwargs[name] = _dataclass_from_dict(ftype, value)
        else:
            kwargs[name] = value
    return cls(**kwargs)


# ---------------------------------------------------------------------------
# LoRA Configuration
# ---------------------------------------------------------------------------

@dataclass
class LoRAConfig:
    """Configuration for LoRA (Low-Rank Adaptation) on a single model.

    When enabled=True, LoRA adapters are injected into the specified target
    modules, and only the LoRA parameters are trained. This dramatically
    reduces memory and optimizer state compared to full-weight training.
    """
    enabled: bool = False
    rank: int = 32
    lora_alpha: int = 8
    target_modules: list = field(default_factory=lambda: [
        "attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0",
    ])
    init_lora_weights: str = "gaussian"
    # Path to pretrained LoRA weights for initialization (empty = init from scratch).
    # This is applied during setup_models() BEFORE resume. If train.resume_from is
    # also set, the resume checkpoint will overwrite these weights — so load_path
    # is only useful when starting a new training run from existing LoRA weights
    # (e.g., fine-tuning a LoRA trained on a different task). For resuming
    # interrupted training, use train.resume_from instead.
    load_path: str = ""
    # If True and load_path is set, merge the loaded LoRA weights into the base
    # model before injecting a fresh LoRA adapter for training. This allows
    # training a new LoRA on top of a previously fine-tuned model.
    # Ignored when load_path is empty.
    merge_before_training: bool = False


# ---------------------------------------------------------------------------
# Model Configuration
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    """Configuration for the base generative model.

    Extensible for different model families (SD3.5, FLUX, SDXL, video DiT, etc.)
    by adding new fields or subclassing.
    """
    pretrained_path: str = ""
    # Optional per-role initialization checkpoints for DMD-family trainers.
    # If empty, that role initializes from pretrained_path.
    # Supported path forms:
    # - Diffusers transformer directory (contains config.json, or parent with transformer/)
    # - .pt/.pth state_dict checkpoint
    # - peft LoRA directory (adapter_config.json + adapter_model.safetensors), only when that role has LoRA enabled
    generator_init_path: str = ""
    fake_score_init_path: str = ""
    teacher_init_path: str = ""
    # Model family identifier: "sd35", "flux", "sdxl", etc.
    model_type: str = "sd35"
    gradient_checkpointing: bool = True
    dtype: str = "bf16"
    # Target image resolution in pixels (used to compute latent spatial size)
    image_resolution: int = 1024
    generator_lora: LoRAConfig = field(default_factory=LoRAConfig)
    fake_score_lora: LoRAConfig = field(default_factory=LoRAConfig)
    teacher_lora: LoRAConfig = field(default_factory=LoRAConfig)
    # SD3-Medium FSDP compatibility: when True, replace PatchEmbed.proj Conv2d
    # with an equivalent nn.Linear before FSDP wrapping. Works around an FSDP
    # use_orig_params=True writeback bug on 4-D Conv2d weights (only triggers
    # for SD3-Medium's Conv2d(16, 1536, k=2, s=2)). Mathematically equivalent
    # but changes the model structure and the saved state_dict shape, so it
    # is DEFAULT FALSE to keep SD3.5 / Flux2 / etc. behavior and existing
    # checkpoints untouched. Enable only for SD3-Medium FSDP runs.
    fsdp_patch_embed_linear: bool = False


# ---------------------------------------------------------------------------
# DMD Algorithm Configuration
# ---------------------------------------------------------------------------

@dataclass
class DMDConfig:
    """Configuration for Distribution Matching Distillation.

    Core DMD parameters following the DMD/DMD2 papers.
    """
    # Number of denoising steps for the generator to produce clean latents.
    # NOTE: For SD3.5, the actual step count is determined by len(sigmas).
    # This field is kept for future model types where sigmas may not be
    # explicitly listed. When both are set, the effective steps = min(num_inference_steps, len(sigmas)).
    num_inference_steps: int = 4
    # Timestep range for DMD loss (as absolute step indices in [0, num_train_timesteps))
    min_step: int = 20
    max_step: int = 980
    num_train_timesteps: int = 1000
    # Teacher (real score) CFG scale range for DMD loss.
    # Each step uniformly samples from [min, max]. Set min == max for a fixed scale.
    real_guidance_scale_min: float = 2.0
    real_guidance_scale_max: float = 8.0
    # CFG scale for the fake score network at DMD LOSS time (typically 1.0 = no guidance)
    fake_guidance_scale: float = 1.0
    # CFG scale for fake score network TRAINING (flow matching loss forward pass).
    # 1.0 = no guidance (default, standard DMD). >1.0 enables CFG during fake
    # score training, which requires uncond embeddings and doubles the forward cost.
    fake_train_guidance_scale: float = 1.0
    # How many fake score updates per 1 generator update (TTUR, default 10)
    fake_update_ratio: int = 10
    gradient_normalization: bool = True
    fake_min_step: int = 0
    fake_max_step: int = 999
    # Timestep sampling for fake score training: "logit_normal" or "uniform"
    fake_timestep_sampling: str = "logit_normal"
    logit_mean: float = 0.0
    logit_std: float = 1.0
    # Timestep sampling for DMD loss (generator update): "uniform" or "logit_normal"
    dmd_timestep_sampling: str = "uniform"
    dmd_logit_mean: float = 0.0
    dmd_logit_std: float = 1.0
    # Whether to randomly stop the backward simulation at an intermediate step.
    random_stop_idx: bool = True
    # Raw denoising steps for backward simulation. Shifted sigmas are computed
    # automatically from these steps using the model-specific shift formula.
    # e.g. [1000, 750, 500, 250] -> [1.0, 0.9, 0.75, 0.5] for SD3.5 shift=3.0
    denoising_step_list: list = field(default_factory=lambda: [1000, 750, 500, 250])
    # Scheduler type for backward simulation. Only "cps" is supported. The CPS
    # step has controllable stochasticity via cps_eta (0.0 = deterministic).
    generator_scheduler: str = "cps"
    # CPS eta parameter (stochasticity). 0.0 = deterministic (Euler-equivalent).
    cps_eta: float = 0.0


# ---------------------------------------------------------------------------
# AC-DMD Algorithm Configuration
# ---------------------------------------------------------------------------

@dataclass
class ACDMDConfig:
    """Configuration for AC-DMD sub-interval training.

    This block controls the AC-DMD objective:
    - Choose an boundary sigma (x_s extraction point)
    - Sample renoising timesteps in the sub-interval [s, t]
    - Train fake score with theoretical flow target weighting
    """

    # Enable the AC-DMD objective in trainers that support it.
    enabled: bool = True
    # Start mode:
    # - "next_sigma" (default, strict paper-style one-step-one-segment):
    #     s = sigma_next of the executed last denoising step.
    # - "fixed_boundary" (legacy approximation):
    #     s = min(sigma_cur, denoising_sigmas[boundary_sigma_index]).
    start_mode: str = "next_sigma"
    # Boundary index used only when start_mode == "fixed_boundary".
    # For 4-step [1000, 750, 500, 250], index=2 corresponds to sigma_s ~= 0.75.
    boundary_sigma_index: int = 2
    # Upper bound (in timestep space [0,1000]) for renoising sampling.
    end_timestep: float = 1000.0
    # Small offset added to interval_start_timestep to avoid degenerate zero-width ranges.
    timestep_offset: float = 10.0
    # When random_stop_idx=False, sample from the full range up to terminal timestep.
    all_include_terminal_when_no_random_stop: bool = True
    # Clamp sampled renoising timestep into [min_renoise_step, max_renoise_step].
    min_renoise_step: int = 20
    max_renoise_step: int = 980
    # Theoretical target toggle:
    # False = paper-style target (move_scale_out=False in flow_forward_target_v2).
    # True  = move scale outside target, mainly for ablation compatibility.
    target_formula_move_scale_out: bool = False
    # Fake score loss weighting clamp for 1 / scale^2.
    fake_loss_weight_clamp_min: float = 0.0
    fake_loss_weight_clamp_max: float = 10.0
    # Optional consistency loss on fake-score training.
    # Disabled by default to preserve existing behavior.
    consistency_enabled: bool = False
    # Weight for consistency loss:
    #   total_fake_loss = fake_ac_loss + consistency_weight * consistency_loss
    consistency_weight: float = 0.0
    # Maximum timestep rollback for one consistency move.
    # For each move, sample delta ~ UniformInt(1, consistency_epsilon_timestep),
    # then move from t to s = max(t - delta, 0).
    consistency_epsilon_timestep: float = 10.0
    # Number of chained consistency moves before evaluating h_theta at the final s.
    consistency_num_steps: int = 1
    # If True, use the two-sample unbiased estimator
    # (two independent stochastic transitions with shared timesteps).
    # If False, use a non-negative one-sample MSE variant.
    consistency_use_two_sample_unbiased: bool = True
    # If True, transition paths are built under no_grad (non-differentiable path
    # sampling), while gradients still flow through final h_theta evaluations.
    # If False, gradients can flow through transition construction as well.
    consistency_stopgrad_transitions: bool = True


# ---------------------------------------------------------------------------
# LastStepLossConfig (auxiliary loss for deterministic CPS steps in GRPO)
# ---------------------------------------------------------------------------

@dataclass
class LastStepLossConfig:
    """Configuration for auxiliary loss on deterministic CPS steps.

    When CPS scheduler is used, the last denoising step (sigma_next=0) is
    deterministic — log_prob=0, ratio=1, so PPO contributes zero gradient.
    This config enables optional base aux loss (DMD / AC-DMD)
    plus optional additive BP reward loss on selected steps.

    Supports applying the auxiliary loss to:
    - Only the last step (default: train_steps=[-1])
    - Multiple steps (e.g., [-1, -2] for last two)
    - All steps (every step gets PPO + aux loss)
    """
    enabled: bool = False
    # "dmd": full DMD loss (requires fake_score_net + teacher)
    # "ac_dmd": AC-DMD regularization + AC-DMD fake-score training
    # "none": disable base aux branch; useful for BP-only training
    loss_type: str = "dmd"
    # Pipeline step indices to apply aux loss. Negative indices supported.
    # [-1] = last step only; [-1, -2] = last two steps.
    train_steps: list = field(default_factory=lambda: [-1])
    # Weight of aux loss relative to PPO loss.
    weight: float = 1.0

    # --- Teacher CFG ---
    real_guidance_scale_min: float = 3.5
    real_guidance_scale_max: float = 3.5

    # --- DMD-specific ---
    fake_guidance_scale: float = 1.0
    gradient_normalization: bool = True
    # Timestep range for forward diffusion of x0
    min_step: int = 20
    max_step: int = 980
    timestep_sampling: str = "logit_normal"
    logit_mean: float = 0.0
    logit_std: float = 1.0

    # --- Fake score training (only when loss_type="dmd") ---
    # NOTE: Use fake_* fields for fake score training, NOT the non-fake variants above.
    fake_min_step: int = 0
    fake_max_step: int = 999
    fake_timestep_sampling: str = "logit_normal"
    fake_logit_mean: float = 0.0
    fake_logit_std: float = 1.0
    # Sub-batch size for fake score training. 0 = all samples at once.
    fake_score_batch_size: int = 0

    # --- AC-DMD extension (used when loss_type == "ac_dmd") ---
    # Start mode:
    # - "next_sigma": boundary starts from the next scheduler sigma.
    # - "fixed_boundary": boundary starts from ac_fixed_boundary_sigma.
    ac_start_mode: str = "next_sigma"
    # Fixed boundary sigma for ac_start_mode == "fixed_boundary".
    ac_fixed_boundary_sigma: float = 0.75
    # Renoise upper bound in model timestep space [0, num_train_timesteps].
    ac_end_timestep: float = 1000.0
    # Small positive offset added to interval_start_timestep.
    ac_timestep_offset: float = 10.0
    # Clamp sampled timestep range.
    ac_min_step: int = 20
    ac_max_step: int = 980
    # Whether to move sigma_ts scale outside the target formula.
    ac_target_formula_move_scale_out: bool = False
    # Inverse-scale weighting clamps for AC-DMD fake-score loss.
    ac_fake_loss_weight_clamp_min: float = 0.0
    ac_fake_loss_weight_clamp_max: float = 10.0
    # Fake-score guidance during AC-DMD fake training / consistency.
    ac_fake_train_guidance_scale: float = 1.0
    # Optional AC-DMD consistency loss on fake-score update.
    ac_consistency_enabled: bool = False
    ac_consistency_weight: float = 0.0
    ac_consistency_epsilon_timestep: float = 10.0
    ac_consistency_num_steps: int = 1
    ac_consistency_use_two_sample_unbiased: bool = True
    ac_consistency_stopgrad_transitions: bool = True

    # --- BP-style differentiable reward loss (optional, additive) ---
    # If enabled, add an extra reward loss on aux steps.
    # Two modes:
    # 1) Training-reward mode (default when supported rewards exist in grpo.reward_fn):
    #      loss_bp_raw = mean(relu(bp_reward_margin - sum_i weight_i * reward_i))
    #      loss_bp = loss_bp_raw * bp_grad_scale
    # 2) ImageReward fallback mode:
    #      loss_bp_raw = mean(relu(bp_reward_margin - reward))
    #      loss_bp = loss_bp_raw * bp_grad_scale
    bp_enabled: bool = False
    # Reward model name/path for ImageReward fallback mode.
    bp_reward_model: str = "ImageReward-v1.0"
    # Hinge margin used in both training-reward and ImageReward fallback modes.
    bp_reward_margin: float = 2.0
    # Base gradient scale from original BP implementation.
    bp_grad_scale: float = 1e-3
    # If True, apply bp only on deterministic CPS steps (sigma_next == 0).
    bp_only_deterministic: bool = True
    # If True, build BP reward from grpo.reward_fn (weighted sum) using
    # currently supported differentiable backends in rtdmd_trainer:
    # {"pickscore", "clipscore", "hpsv2"}.
    # If False, uses ImageReward score_gard path via bp_reward_model.
    bp_use_training_reward_fn: bool = True
    # Text tokenization max length for ImageReward BLIP tokenizer.
    bp_token_max_length: int = 35
    # Sub-batch size for BP VAE decode + reward forward.
    # 0 = disable chunking (use full training sub-batch).
    # Set to 1~4 to reduce peak VRAM when BP OOMs.
    bp_batch_size: int = 0


# ---------------------------------------------------------------------------
# GRPO (Group Relative Policy Optimization) Configuration
# ---------------------------------------------------------------------------

@dataclass
class GRPOConfig:
    """Configuration for GRPO: online RL fine-tuning via SDE sampling + PPO.

    GRPO generates K images per prompt via stochastic SDE sampling (with
    log-probability tracking), scores them with reward models, and optimizes
    using PPO-style clipped policy gradients. Completely independent of DMD —
    no teacher or fake score network required.

    Reference: grpo (https://arxiv.org/abs/2505.05470)
    """

    # --- Sampling ---
    # Sampling mode:
    #   "sde" — Multi-step SDE sampling (e.g., 10 steps), for base/pretrained models.
    #           Timesteps are evenly spaced via set_timesteps(num_inference_steps=num_steps).
    #   "dmd" — Few-step sampling using DMD's sigma schedule, for distilled models.
    #           Timesteps come from denoising_step_list -> compute_sigmas_sd35().
    #           num_steps is ignored (determined by len(denoising_step_list)).
    sampling_mode: str = "sde"
    num_steps: int = 10
    # DMD sigma schedule: raw denoising steps shifted via model's shift formula.
    # Only used when sampling_mode="dmd". E.g., [1000, 750, 500, 250] -> sigmas [1.0, 0.9, 0.75, 0.5].
    # Set noise_level to match DMD training (e.g., cps_eta=X -> noise_level=X).
    denoising_step_list: list = field(default_factory=lambda: [1000, 750, 500, 250])
    # SDE stochasticity: 0 = ODE (deterministic), 1 = full re-noise.
    # For DMD mode, match the DMD training scheduler's eta/noise level.
    noise_level: float = 0.7
    # SDE type: "sde" (standard) or "cps" (Coefficients-Preserving Sampling).
    # "cps" strongly recommended for DMD mode (preserves coefficients with large dt steps).
    sde_type: str = "sde"
    guidance_scale: float = 4.5
    # Use SHA256(prompt) to generate deterministic initial noise per prompt
    same_latent: bool = False
    # --- Evaluation ---
    eval_num_steps: int = 40
    eval_guidance_scale: float = 4.5
    eval_noise_level: float = 0.0
    eval_sde_type: str = "cps"
    eval_freq: int = 60
    save_freq: int = 60

    # --- PPO Training ---
    clip_range: float = 1e-4
    clip_range_lt: float = 0.0
    clip_range_gt: float = 0.0
    # KL coefficient for reference model regularization (0 = no KL).
    # LoRA: reference via disable_adapter(). Full weight: frozen deepcopy.
    beta: float = 0.04
    adv_clip_max: float = 5.0
    # Fraction of timesteps to train on (e.g., 0.99 -> 9 out of 10)
    timestep_fraction: float = 0.99
    num_inner_epochs: int = 1
    # Use CFG during training forward pass (concatenate neg+pos embeds + latents)
    cfg_in_training: bool = True
    max_grad_norm: float = 1.0

    # --- Optimizer (independent from DMD's SolverConfig) ---
    learning_rate: float = 3e-4
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_weight_decay: float = 1e-4
    adam_epsilon: float = 1e-8

    # --- Reward ---
    reward_fn: dict = field(default_factory=lambda: {"pickscore": 1.0})
    reward_ckpt_path: str = ""
    per_prompt_stat_tracking: bool = True
    per_prompt_global_std: bool = True
    # Weight advantages instead of weighting rewards before advantage computation.
    # When True: compute advantage per reward function independently, then
    # weighted-sum the advantages, and finally normalize (zero-mean, unit-std).
    # When False (default): compute advantage on the weighted-sum reward (current behavior).
    weight_advantages: bool = False

    # --- K-repeat sampling ---
    num_batches_per_epoch: int = 2
    sample_batch_size: int = 9
    num_image_per_prompt: int = 24
    test_batch_size: int = 16
    vae_decode_batch_size: int = 4

    # --- Training ---
    train_batch_size: int = 9
    gradient_accumulation_steps: int = 1
    num_epochs: int = 100000

    # --- EMA (parameter-level, not model-level) ---
    use_ema: bool = True
    ema_decay: float = 0.9
    ema_update_interval: int = 8

    # --- Logging ---
    log_num_groups: int = 0
    log_image_freq: int = 10

    # --- Step Selection (DMD+CPS mode only) ---
    # Random step selection for DMD+CPS mode.
    # When enabled and conditions are met (sampling_mode="dmd", sde_type="cps"),
    # randomly selects `num_train_steps` steps from
    # denoising_step_list[range_start:range_end+1] to train on each epoch.
    # Non-selected steps use prompt-deterministic shared CPS noise across
    # K-repeat group members (even across ranks), reducing variance in
    # advantage computation from non-trained steps.
    # When enabled, timestep_fraction is ignored.
    # Set enabled=False or leave empty dict to use timestep_fraction (default).
    # Keys: enabled (bool), range_start (int), range_end (int), num_train_steps (int)
    # Example: {"enabled": true, "range_start": 0, "range_end": 3, "num_train_steps": 2}
    step_selection: dict = field(default_factory=dict)

    # --- Last-step auxiliary loss (DMD / AC-DMD for deterministic steps) ---
    last_step_loss: LastStepLossConfig = field(default_factory=LastStepLossConfig)


# ---------------------------------------------------------------------------
# Solver (Optimizer & LR Scheduler) Configuration
# ---------------------------------------------------------------------------

@dataclass
class GeneratorSolverConfig:
    """Optimizer config for the generator model."""
    lr: float = 1e-6
    warmup_steps: int = 0
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0


@dataclass
class FakeSolverConfig:
    """Optimizer config for the fake score network.

    Separated from generator solver to allow independent tuning (e.g.,
    different LR for fake score as suggested in DMD2 TTUR).
    """
    lr: float = 1e-6
    warmup_steps: int = 500
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0


@dataclass
class TeacherSolverConfig:
    """Optimizer config for the teacher model (reserved for trainers that train the teacher)."""
    lr: float = 3e-6
    warmup_steps: int = 0
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0


@dataclass
class SolverConfig:
    """Combined solver configuration for all trainable components."""
    generator: GeneratorSolverConfig = field(default_factory=GeneratorSolverConfig)
    fake_score: FakeSolverConfig = field(default_factory=FakeSolverConfig)
    teacher: TeacherSolverConfig = field(default_factory=TeacherSolverConfig)


# ---------------------------------------------------------------------------
# Training Configuration
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    """General training parameters."""
    total_steps: int = 100000
    save_interval: int = 1000
    log_interval: int = 10
    seed: int = 42
    # Per-GPU batch size
    batch_size: int = 1
    gradient_accumulation_steps: int = 1
    # Autocast dtype for forward pass computation: "bf16", "fp16", or "no".
    # Decoupled from model.dtype which controls weight storage & FSDP comm precision.
    autocast_dtype: str = "bf16"
    # Resume from checkpoint path (empty = train from scratch)
    resume_from: str = ""
    # Generator EMA (off by default).
    use_ema: bool = False
    ema_decay: float = 0.9999
    ema_every: int = 1


# ---------------------------------------------------------------------------
# Distributed Training Configuration
# ---------------------------------------------------------------------------

@dataclass
class DistributedConfig:
    """Distributed training parameters.

    Supports FSDP and DDP. FSDP is recommended for large models (SD3.5 2B+).
    """
    # "fsdp" or "ddp"
    strategy: str = "fsdp"
    # FSDP sharding strategy: "full_shard", "hybrid", "shard_grad_op"
    fsdp_sharding: str = "full_shard"
    # Offload frozen FSDP models (e.g. DMD teacher) to CPU between forwards.
    # Enable via yaml when large aux models OOM (e.g. FLUX2-9B AC-DMD).
    fsdp_cpu_offload_frozen: bool = False


# ---------------------------------------------------------------------------
# Evaluation Configuration
# ---------------------------------------------------------------------------

@dataclass
class EvalConfig:
    """Evaluation configuration for periodic reward-based assessment.

    During training, the evaluator generates images from fixed prompts,
    computes reward scores, and logs results to wandb.
    """
    enabled: bool = True
    eval_interval: int = 5000
    # Prompt source: "prompt_file" uses eval_prompt_path, "dataset" uses dataset_path/dataset/
    eval_prompt_source: str = "prompt_file"
    # Path to evaluation prompt file (when eval_prompt_source == "prompt_file").
    # Accepts .json, .jsonl, .txt (same formats as prompt_path above).
    eval_prompt_path: str = "prompts.json"
    # Dataset name when eval_prompt_source == "dataset".
    # The evaluator searches dataset_path/dataset/ for:
    #   test.txt (pickscore, ocr), test.jsonl (geneval2), test_metadata.jsonl (geneval),
    #   prompts.json, prompts.txt
    # Supported: "pickscore" | "ocr" | "geneval" | "geneval2" | "drawbench" (or custom)
    dataset: str = "pickscore"
    dataset_path: str = "dataset"
    # Reward functions with weights: {name: weight}, e.g.,
    # {"pickscore": 1.0, "hpsv2": 1.0, "clipscore": 1.0}
    reward_fn: dict = field(default_factory=lambda: {"pickscore": 1.0})
    # Optional per-reward dataset override for evaluation.
    # Keys are reward names from reward_fn, values are dataset names under dataset_path.
    # Example:
    #   reward_dataset_map: {"pickscore": "pickscore", "geneval2": "geneval2"}
    # Unspecified rewards fall back to the default prompt source
    # (eval_prompt_source + dataset/eval_prompt_path).
    reward_dataset_map: dict = field(default_factory=dict)
    reward_ckpt_path: str = ""
    eval_num_steps: int = 4
    # CFG scale for evaluation generation. DMD generator is CFG-free so this
    # defaults to 1.0. Currently unused by generator eval (_generate_latents
    # hardcodes 1.0).
    eval_guidance_scale: float = 1.0
    eval_batch_size: int = 4
    # Number of images to log to wandb media panel (0 = don't log images)
    num_media_images: int = 8
    eval_seed: int = 12345


# ---------------------------------------------------------------------------
# Fake Score Evaluation Configuration
# ---------------------------------------------------------------------------

@dataclass
class FakeEvalConfig:
    """Configuration for evaluating the fake score network as a denoiser.

    When enabled, the fake score network is used as a standard flow-matching
    denoiser (with configurable num_steps and guidance_scale) to generate images,
    which are then scored by the reward model. Logs to fake_eval/* section.
    """
    enabled: bool = False
    num_steps: int = 28
    guidance_scale: float = 4.5
    batch_size: int = 4


# ---------------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------------

@dataclass
class LoggingConfig:
    """Logging and experiment tracking configuration.

    Uses wandb for experiment tracking with structured metric sections:
    - dmd/*: DMD loss metrics (loss_dm, gradient_norm, etc.)
    - fake_score/*: Fake score network training metrics
    - generator/*: Generator training metrics
    - system/*: GPU memory, throughput, etc.
    """
    project: str = "RTDMD"
    run_name: str = ""
    entity: str = ""
    tags: list = field(default_factory=list)
    enabled: bool = True


# ---------------------------------------------------------------------------
# Root Configuration
# ---------------------------------------------------------------------------

@dataclass
class RTDMDConfig:
    """Root configuration for RTDMD.

    Aggregates all sub-configs. To add a new algorithm, add a new sub-config
    field and handle it in the trainer dispatch.
    """
    model: ModelConfig = field(default_factory=ModelConfig)
    dmd: DMDConfig = field(default_factory=DMDConfig)
    ac_dmd: ACDMDConfig = field(default_factory=ACDMDConfig)
    grpo: GRPOConfig = field(default_factory=GRPOConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    distributed: DistributedConfig = field(default_factory=DistributedConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    fake_eval: FakeEvalConfig = field(default_factory=FakeEvalConfig)
    # Path to training prompt source. Accepts:
    #   - A file path: "prompts.json", "data/train.txt", "data/train_metadata.jsonl"
    #     Supported formats: .json (list of strings or list of dicts with "prompt" key),
    #     .jsonl (one JSON dict per line with "prompt" key, e.g., geneval metadata),
    #     .txt (one prompt per line)
    #   - A dataset directory: "dataset/pickscore", "dataset/geneval"
    #     Auto-detects: train.txt, train.jsonl, train_metadata.jsonl, prompts.json, prompts.txt
    prompt_path: str = "prompts.json"
    output_dir: str = "./outputs"
    # Root-level reward checkpoint path. Auto-propagates to sub-configs if not set.
    reward_ckpt_path: str = ""

    def __post_init__(self) -> None:
        """Propagate root-level reward_ckpt_path to sub-configs that don't have their own set.

        This allows users to set a single reward_ckpt_path at the root level and have it
        automatically apply to all sub-configs (grpo, eval) unless they have their
        own value set.

        Priority: sub-config value > root config value
        """
        if not self.reward_ckpt_path:
            return

        sub_configs = [
            self.grpo,
            self.eval,
        ]

        for sub_cfg in sub_configs:
            if hasattr(sub_cfg, "reward_ckpt_path") and not sub_cfg.reward_ckpt_path:
                sub_cfg.reward_ckpt_path = self.reward_ckpt_path

    @classmethod
    def from_yaml(cls, path: str, overrides: dict[str, Any] | None = None) -> "RTDMDConfig":
        """Load config from YAML file with optional dot-notation overrides.

        Args:
            path: Path to the YAML config file.
            overrides: Optional dict of dot-notation overrides, e.g.,
                {"train.seed": 123, "dmd.fake_update_ratio": 10}.

        Returns:
            Fully instantiated RTDMDConfig.
        """
        with open(path, "r") as f:
            raw = yaml.safe_load(f)
        if overrides:
            raw = _apply_flat_overrides(raw, overrides)
        return _dataclass_from_dict(cls, raw)

    def to_dict(self) -> dict:
        """Serialize config to a plain dict (for wandb logging, etc.)."""
        return asdict(self)

    def save_yaml(self, path: str) -> None:
        """Save config to a YAML file for reproducibility."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)
