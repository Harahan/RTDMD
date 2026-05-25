"""
GRPO + DMD Trainer for RTDMD.

Extends GRPOTrainer with auxiliary DMD / AC-DMD losses for
deterministic CPS steps (sigma_next=0) where PPO contributes zero gradient
(ratio=1).

Training loop per epoch:
    1. [Sampling]   Inherited from GRPOTrainer (SDE multi-step + reward scoring)
    2. [Fake Score] Train fake_score_net (DMD/AC-DMD modes)
    3. [Advantage]  Inherited from GRPOTrainer (per-prompt normalized advantages)
    4. [Training]   PPO for stochastic steps + DMD / AC-DMD for deterministic steps

Class hierarchy:
    BaseTrainer → GRPOTrainer → RTDMDTrainer

Reference:
    - GRPO: arXiv 2505.05470
    - DMD: arXiv 2311.18828
    - DMD2: arXiv 2405.14867
"""

from __future__ import annotations

import contextlib
import functools
import logging
import os
import random
from collections import defaultdict
from typing import Any

import torch
import torch.distributed as dist
from diffusers.training_utils import (
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.nn.parallel import DistributedDataParallel as DDP

from rtdmd.config import RTDMDConfig
from rtdmd.diffusers_patch.sde_with_logprob import sde_step_with_logprob
from rtdmd.parallel.utils import (
    ddp_wrap_model,
    free_fsdp_unsharded_params,
    fsdp_wrap_model,
    get_rank,
    get_world_size,
    is_main_process,
)
from rtdmd.trainers.ac_dmd_mixin import ACDMDMixin
from rtdmd.trainers.bp_mixin import BPMixin
from rtdmd.trainers.dmd_trainer import dmd_loss
from rtdmd.trainers.grpo_trainer import GRPOTrainer
from rtdmd.utils.lora import setup_lora

logger = logging.getLogger(__name__)


class RTDMDTrainer(BPMixin, ACDMDMixin, GRPOTrainer):
    """GRPO + DMD / AC-DMD Trainer.

    Extends GRPOTrainer with auxiliary loss on deterministic CPS steps.
    When ``last_step_loss.enabled=False``, behaves identically to GRPOTrainer.
    """

    def __init__(self, config: RTDMDConfig):
        super().__init__(config)
        self._has_fake_score = False
        self.teacher = None
        self.fake_score_net = None
        self._dmd_sigmas = None
        self._init_ac_state()
        self._init_bp_state()

    def _is_redundant_pretrained_init_path(self, init_path: str) -> bool:
        """Return True when init_path points to the same pretrained root."""
        if not init_path:
            return False
        path = str(init_path).strip()
        if not path or os.path.isfile(path):
            return False

        pretrained_root = os.path.realpath(
            os.path.abspath(str(self.config.model.pretrained_path))
        )
        candidate = os.path.realpath(os.path.abspath(path))
        if candidate == pretrained_root:
            return True
        pretrained_transformer = os.path.realpath(
            os.path.join(pretrained_root, "transformer")
        )
        return candidate == pretrained_transformer

    # ------------------------------------------------------------------
    # Model setup
    # ------------------------------------------------------------------

    def setup_models(self) -> None:
        """Extend GRPOTrainer setup with teacher + optional fake_score."""
        super().setup_models()

        ls_cfg = self.config.grpo.last_step_loss
        if not ls_cfg.enabled:
            return  # zero diff from GRPOTrainer

        loss_type = self._normalized_last_step_loss_type()
        base_aux_enabled = loss_type != "none"

        # --- Early validation ---
        if loss_type not in ("dmd", "ac_dmd", "none"):
            raise ValueError(
                "last_step_loss.loss_type must be one of "
                "('dmd', 'ac_dmd', 'none'), "
                f"got '{ls_cfg.loss_type}'"
            )
        if base_aux_enabled and ls_cfg.real_guidance_scale_min > ls_cfg.real_guidance_scale_max:
            raise ValueError(
                "last_step_loss real guidance range is invalid: "
                f"min={ls_cfg.real_guidance_scale_min}, "
                f"max={ls_cfg.real_guidance_scale_max}"
            )
        if loss_type == "ac_dmd":
            self._validate_ac_last_step_config(ls_cfg)
        self._validate_bp_config(ls_cfg)

        if not base_aux_enabled and not self._bp_enabled():
            logger.info(
                "  last_step_loss enabled but both base aux (loss_type=none) and "
                "BP are disabled; skipping extra model setup."
            )
            return
        if self.config.grpo.sde_type != "cps":
            raise ValueError(
                f"last_step_loss requires sde_type='cps', "
                f"got '{self.config.grpo.sde_type}'"
            )

        model_cfg = self.config.model
        dist_cfg = self.config.distributed

        # Resolve dtype
        dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
        dtype = dtype_map.get(model_cfg.dtype, torch.bfloat16)

        # FSDP wrap policy (model-type specific, with import fallback)
        wrap_policy = self._build_transformer_wrap_policy()

        if base_aux_enabled:
            # --- Teacher: frozen pretrained transformer ---
            logger.info("  Loading teacher transformer for last_step_loss...")
            self.teacher = self._load_pretrained_transformer(torch_dtype=dtype)
            if self._is_redundant_pretrained_init_path(model_cfg.teacher_init_path):
                logger.info(
                    "  teacher_init_path matches pretrained_path; "
                    "skip redundant initialization reload."
                )
            else:
                self._load_model_init_from_path(
                    self.teacher,
                    model_cfg.teacher_init_path,
                    "teacher",
                )
            self.teacher.requires_grad_(False)
            self.teacher.eval()
            # Apply the SD3-Medium PatchEmbed Conv2d -> Linear workaround BEFORE
            # FSDP wrap (no-op when the flag is False or for Flux2). Without this,
            # SD3-Medium teacher would hit the FSDP use_orig_params writeback bug
            # because this teacher is loaded after super().setup_models() and so
            # bypasses the standard _pre_wrap_models() hook.
            self._maybe_replace_patch_embed_with_linear(self.teacher)
            # Frozen teacher: FSDP for memory sharding, no DDP (no gradient sync needed)
            if dist_cfg.strategy == "fsdp":
                self.teacher = fsdp_wrap_model(
                    self.teacher,
                    sharding_strategy=dist_cfg.fsdp_sharding,
                    fsdp_precision=model_cfg.dtype,
                    auto_wrap_policy=wrap_policy,
                    cpu_offload=dist_cfg.fsdp_cpu_offload_frozen,
                )
            else:
                self.teacher = self.teacher.to(torch.device("cuda"))
            # NOT in self.models → not in checkpoint
            offload_note = (
                ", FSDP cpu_offload=True"
                if dist_cfg.strategy == "fsdp" and dist_cfg.fsdp_cpu_offload_frozen
                else ""
            )
            logger.info(
                f"  Teacher loaded (frozen, not in checkpoint{offload_note})"
            )

        # --- Fake score (DMD / AC-DMD) ---
        if loss_type in ("dmd", "ac_dmd"):
            logger.info(
                "  Loading fake_score_net for %s last_step_loss...",
                ls_cfg.loss_type,
            )
            self.fake_score_net = self._load_pretrained_transformer(torch_dtype=dtype)
            if self._is_redundant_pretrained_init_path(model_cfg.fake_score_init_path):
                logger.info(
                    "  fake_score_init_path matches pretrained_path; "
                    "skip redundant initialization reload."
                )
            else:
                self._load_model_init_from_path(
                    self.fake_score_net,
                    model_cfg.fake_score_init_path,
                    "fake_score",
                )
            if model_cfg.fake_score_lora.enabled:
                setup_lora(self.fake_score_net, model_cfg.fake_score_lora)
                logger.info(
                    f"  Fake score: LoRA injected (rank={model_cfg.fake_score_lora.rank})"
                )

            # Apply the SD3-Medium PatchEmbed Conv2d -> Linear workaround BEFORE
            # FSDP wrap, mirroring the teacher above. Done after LoRA injection
            # so the LoRA adapter targets (attn.*) are unaffected, while pos_embed
            # gets the base-weight Conv2d -> Linear rewrite.
            self._maybe_replace_patch_embed_with_linear(self.fake_score_net)

            has_lora = model_cfg.fake_score_lora.enabled
            if dist_cfg.strategy == "fsdp":
                self.fake_score_net = fsdp_wrap_model(
                    self.fake_score_net,
                    sharding_strategy=dist_cfg.fsdp_sharding,
                    fsdp_precision=model_cfg.dtype,
                    auto_wrap_policy=wrap_policy,
                )
            elif dist.is_initialized() and get_world_size() > 1:
                # DDP: .to(cuda) BEFORE ddp_wrap_model (it doesn't move model)
                self.fake_score_net = self.fake_score_net.to(torch.device("cuda"))
                self.fake_score_net = ddp_wrap_model(
                    self.fake_score_net,
                    find_unused_parameters=has_lora,
                    # Reduce extra DDP BROADCAST collectives; transformer fake-score
                    # path does not rely on running buffers that require sync.
                    broadcast_buffers=False,
                )
            else:
                self.fake_score_net = self.fake_score_net.to(torch.device("cuda"))
            self.models["fake_score_net"] = self.fake_score_net  # → checkpoint
            self._has_fake_score = True
            logger.info("  Fake score loaded (in checkpoint)")

        # --- DMD sigma table for timestep sampling ---
        if base_aux_enabled:
            self._setup_dmd_sigmas()

        # Fake-score x0 reconstruction requires per-step policy outputs
        # (noise_preds / prev_means / std_devs) cached during sampling.
        self._store_policy_outputs = bool(ls_cfg.enabled and self._has_fake_score)

        bp_enabled = self._bp_enabled()
        if not bp_enabled:
            bp_mode = "disabled"
        elif self._use_training_rewards_for_bp():
            bp_mode = "training_reward_fn"
        else:
            bp_mode = "imagereward"

        logger.info(
            f"  last_step_loss: type={ls_cfg.loss_type}, weight={ls_cfg.weight}, "
            f"train_steps={ls_cfg.train_steps}, "
            f"base_enabled={base_aux_enabled}, "
            f"has_fake_score={self._has_fake_score}, "
            f"bp_enabled={bp_enabled}, "
            f"bp_mode={bp_mode}"
        )

    def _setup_dmd_sigmas(self) -> None:
        """Build sigma lookup table for DMD timestep sampling.

        Matches DMDTrainer behavior:
            self.sigmas = torch.flip(self.scheduler.sigmas, dims=[0]).cuda()
        so sampled timestep indices [0, 999] map to the same sigma convention
        used by DMD loss / fake-score training.
        """
        self._dmd_sigmas = torch.flip(self.scheduler.sigmas, dims=[0]).to(
            torch.device("cuda")
        )

    def _normalized_last_step_loss_type(self) -> str:
        return str(self.config.grpo.last_step_loss.loss_type).lower()

    def _base_aux_enabled(self) -> bool:
        ls_cfg = self.config.grpo.last_step_loss
        return bool(ls_cfg.enabled and self._normalized_last_step_loss_type() != "none")

    def _any_aux_enabled(self) -> bool:
        return bool(self._base_aux_enabled() or self._bp_enabled())

    def _validate_ac_last_step_config(self, ls_cfg) -> None:
        start_mode = str(ls_cfg.ac_start_mode).lower()
        if start_mode not in ("next_sigma", "fixed_boundary"):
            raise ValueError(
                "last_step_loss.ac_start_mode must be 'next_sigma' "
                f"or 'fixed_boundary', got {ls_cfg.ac_start_mode!r}"
            )
        if ls_cfg.ac_min_step > ls_cfg.ac_max_step:
            raise ValueError(
                "Invalid sub-interval timestep bounds in last_step_loss: "
                f"ac_min_step={ls_cfg.ac_min_step}, "
                f"ac_max_step={ls_cfg.ac_max_step}"
            )
        if ls_cfg.ac_consistency_num_steps < 1:
            raise ValueError(
                "last_step_loss.ac_consistency_num_steps must be >= 1, "
                f"got {ls_cfg.ac_consistency_num_steps}"
            )
        if ls_cfg.ac_consistency_epsilon_timestep < 0.0:
            raise ValueError(
                "last_step_loss.ac_consistency_epsilon_timestep must be >= 0, "
                f"got {ls_cfg.ac_consistency_epsilon_timestep}"
            )

    def _num_train_timesteps(self) -> float:
        if hasattr(self.scheduler, "config"):
            return float(
                getattr(
                    self.scheduler.config,
                    "num_train_timesteps",
                    self.config.dmd.num_train_timesteps,
                )
            )
        return float(self.config.dmd.num_train_timesteps)

    def _sigma_to_model_timestep(self, sigma: torch.Tensor | float) -> torch.Tensor | float:
        num_ts = self._num_train_timesteps()
        if isinstance(sigma, torch.Tensor):
            return sigma.float() * num_ts
        return float(sigma) * num_ts

    def _model_timestep_to_sigma(
        self,
        timestep: torch.Tensor | float,
    ) -> torch.Tensor | float:
        num_ts = self._num_train_timesteps()
        if isinstance(timestep, torch.Tensor):
            return timestep.float() / num_ts
        return float(timestep) / num_ts

    def _lookup_step_sigmas(self, timestep: torch.Tensor) -> tuple[float, float]:
        """Return (sigma_cur, sigma_next) for one pipeline timestep."""
        step_idx = self.sde_scheduler.index_for_timestep(timestep)
        sigma_cur = float(self.sde_scheduler.sigmas[step_idx].item())
        next_idx = min(step_idx + 1, len(self.sde_scheduler.sigmas) - 1)
        sigma_next = float(self.sde_scheduler.sigmas[next_idx].item())
        return sigma_cur, sigma_next

    # ------------------------------------------------------------------
    # Optimizer setup
    # ------------------------------------------------------------------

    def setup_optimizers(self) -> None:
        """Extend GRPOTrainer optimizers with fake_score optimizer."""
        super().setup_optimizers()

        ls_cfg = self.config.grpo.last_step_loss
        if not ls_cfg.enabled or not self._has_fake_score:
            return  # zero diff

        # Only trainable (requires_grad=True) parameters
        trainable_params = [
            p for p in self.fake_score_net.parameters() if p.requires_grad
        ]
        solver_cfg = self.config.solver.fake_score
        self.optimizers["fake_score"] = torch.optim.AdamW(
            trainable_params,
            lr=solver_cfg.lr,
            betas=(solver_cfg.beta1, solver_cfg.beta2),
            eps=solver_cfg.eps,
            weight_decay=solver_cfg.weight_decay,
        )
        self.schedulers["fake_score"] = self.create_warmup_constant_scheduler(
            self.optimizers["fake_score"], warmup_steps=solver_cfg.warmup_steps,
        )
        logger.info(
            f"  Fake score optimizer: lr={solver_cfg.lr}, "
            f"params={len(trainable_params)}"
        )

    # ------------------------------------------------------------------
    # LoRA wiring
    # ------------------------------------------------------------------

    def _get_lora_model_names(self) -> set[str]:
        names = super()._get_lora_model_names()
        if (
            getattr(self, "_has_fake_score", False)
            and self.config.model.fake_score_lora.enabled
        ):
            names.add("fake_score_net")
        return names

    def _get_lora_configs(self) -> dict[str, Any]:
        configs = super()._get_lora_configs()
        if (
            getattr(self, "_has_fake_score", False)
            and self.config.model.fake_score_lora.enabled
        ):
            configs["fake_score_net"] = self.config.model.fake_score_lora
        return configs

    # ------------------------------------------------------------------
    # Aux step resolution
    # ------------------------------------------------------------------

    def _resolve_aux_steps(self, total_pipeline_steps: int) -> set[int]:
        """Parse train_steps list → set of absolute step indices.

        Called AFTER Stage 1, using ``samples["timesteps"].shape[1]``
        as total_pipeline_steps (actual T from sampling, not stale scheduler state).
        """
        raw = self.config.grpo.last_step_loss.train_steps
        result = set()
        for s in raw:
            idx = s if s >= 0 else total_pipeline_steps + s
            if not (0 <= idx < total_pipeline_steps):
                raise ValueError(
                    f"last_step_loss.train_steps index {s} resolves to {idx}, "
                    f"out of range [0, {total_pipeline_steps})"
                )
            result.add(idx)
        return result

    # ------------------------------------------------------------------
    # Timestep sampling helper (standalone, not inherited from DMDTrainer)
    # ------------------------------------------------------------------

    @staticmethod
    def _sample_timesteps_static(
        batch_size: int,
        min_step: int,
        max_step: int,
        sampling: str,
        logit_mean: float,
        logit_std: float,
        device: torch.device,
    ) -> torch.Tensor:
        """Sample integer timestep indices with configurable strategy."""
        max_step = min(max_step, 999)
        step_range = max_step - min_step + 1
        if sampling == "logit_normal":
            u = compute_density_for_timestep_sampling(
                weighting_scheme="logit_normal",
                batch_size=batch_size,
                logit_mean=logit_mean,
                logit_std=logit_std,
            )
            indices = (u * step_range).long().clamp(0, step_range - 1) + min_step
            return indices.to(device)
        else:
            return torch.randint(
                min_step, max_step + 1, (batch_size,),
                device=device, dtype=torch.long,
            )

    # ------------------------------------------------------------------
    # Sampling override (append BP prompt encodings to samples)
    # ------------------------------------------------------------------

    def _grpo_sampling_stage(
        self,
        grpo_cfg,
        autocast_ctx,
        executor,
        neg_embeds,
        neg_pooled,
        selected_steps: list[int] | None = None,
        epoch_seed: int = 0,
    ) -> tuple[dict, list[str], list]:
        """Run sampling and append BP prompt encodings when BP is enabled.

        Whether the sampling pipeline caches per-step policy outputs
        (``noise_preds``/``prev_means``/``std_devs``) is controlled by
        ``self._store_policy_outputs``, which is set in ``setup_models`` based
        on whether fake-score training needs them.
        """
        ls_cfg = self.config.grpo.last_step_loss
        need_policy_outputs = ls_cfg.enabled and self._has_fake_score

        samples, all_prompts_local, all_images_np = super()._grpo_sampling_stage(
            grpo_cfg=grpo_cfg,
            autocast_ctx=autocast_ctx,
            executor=executor,
            neg_embeds=neg_embeds,
            neg_pooled=neg_pooled,
            selected_steps=selected_steps,
            epoch_seed=epoch_seed,
        )

        if need_policy_outputs and "noise_preds" not in samples:
            raise RuntimeError(
                "Expected 'noise_preds' in sampling outputs for fake score x0 reconstruction, "
                "but it is missing."
            )

        if self._bp_enabled():
            expected = samples["timesteps"].shape[0]
            if self._use_training_rewards_for_bp():
                encoded = self._encode_bp_training_prompts(all_prompts_local)
                if not encoded:
                    raise RuntimeError(
                        "BP training-reward mode enabled but no prompt encodings were produced."
                    )
                for key, value in encoded.items():
                    if value.shape[0] != expected:
                        raise RuntimeError(
                            "BP prompt tokenization shape mismatch for key "
                            f"{key}: num_tokens={value.shape[0]}, num_samples={expected}"
                        )
                    samples[key] = value.cpu()
            else:
                bp_input_ids, bp_attention_mask = self._encode_bp_prompts(
                    all_prompts_local
                )
                if bp_input_ids.shape[0] != expected:
                    raise RuntimeError(
                        "BP prompt tokenization shape mismatch: "
                        f"num_tokens={bp_input_ids.shape[0]}, num_samples={expected}"
                    )
                samples["bp_input_ids"] = bp_input_ids.cpu()
                samples["bp_attention_mask"] = bp_attention_mask.cpu()
        return samples, all_prompts_local, all_images_np

    # ------------------------------------------------------------------
    # Epoch flow (full override)
    # ------------------------------------------------------------------

    def _run_grpo_epoch(
        self,
        grpo_cfg,
        autocast_ctx,
        executor,
        neg_embeds,
        neg_pooled,
        train_neg_embeds,
        train_neg_pooled,
        num_train_timesteps,
        clip_lt,
        clip_gt,
        step_sel_config: dict | None = None,
    ) -> dict[str, float]:
        """GRPO epoch with auxiliary DMD / AC-DMD loss."""
        ls_cfg = grpo_cfg.last_step_loss
        if not self._any_aux_enabled():
            raise NotImplementedError(
                "Pure GRPO mode is not supported in this release. "
                "Enable a base aux loss (last_step_loss.loss_type='dmd' or 'ac_dmd') "
                "or set last_step_loss.bp_enabled=true."
            )

        # --- Step selection ---
        selected_steps = None
        epoch_seed = self._grpo_epoch * 100000
        if step_sel_config is not None:
            selected_steps = self._sample_selected_steps(
                step_sel_config,
                epoch_seed=epoch_seed,
            )

        # --- Stage 1: Sampling (inherited) ---
        sampling_start = self._sync_time()
        samples, all_prompts_local, all_images_np = self._grpo_sampling_stage(
            grpo_cfg, autocast_ctx, executor, neg_embeds, neg_pooled,
            selected_steps=selected_steps, epoch_seed=epoch_seed,
        )
        sampling_time_s = self._sync_time() - sampling_start

        # --- Resolve steps using ACTUAL pipeline T ---
        T = samples["timesteps"].shape[1]
        aux_steps = self._resolve_aux_steps(T)

        # PPO steps (from parent logic)
        if selected_steps is not None:
            ppo_steps = set(s for s in selected_steps if 0 <= s < T)
        else:
            num_ppo = min(int(T * grpo_cfg.timestep_fraction), T)
            ppo_steps = set(range(num_ppo))

        # Boundary check
        for s in ppo_steps:
            if not (0 <= s < T):
                raise ValueError(f"PPO step {s} out of range [0, {T})")

        # --- Fake score training (DMD / AC-DMD) ---
        fake_metrics = None
        fake_update_time_s = 0.0
        if self._has_fake_score:
            fake_update_start = self._sync_time()
            if self._is_ac_dmd():
                fake_metrics = self._train_fake_score_ac_from_rollout(
                    samples, ls_cfg, autocast_ctx, aux_steps=aux_steps,
                )
            else:
                fake_metrics = self._train_fake_score_from_x0(
                    samples, ls_cfg, autocast_ctx, aux_steps=aux_steps,
                )
            fake_update_time_s = self._sync_time() - fake_update_start

        # --- Stage 2: Advantages (use FULL T to avoid OOB on aux steps) ---
        advantages_start = self._sync_time()
        epoch_metrics = self._grpo_compute_advantages(
            grpo_cfg, samples, all_prompts_local, all_images_np, T,
        )
        advantages_time_s = self._sync_time() - advantages_start
        if fake_metrics:
            for k, v in fake_metrics.items():
                epoch_metrics[f"fake_score/{k}"] = float(v)
        rollout_time_s = sampling_time_s + advantages_time_s

        # --- Merged steps + effective_grad_accum ---
        all_steps = sorted(ppo_steps | aux_steps)
        if not all_steps:
            raise ValueError("No training steps: ppo_steps and aux_steps are both empty")
        effective_grad_accum = grpo_cfg.gradient_accumulation_steps * len(all_steps)

        # --- Stage 3: Training (overridden with aux loss) ---
        training_start = self._sync_time()
        self._grpo_dmd_training_stage(
            grpo_cfg, autocast_ctx, samples,
            train_neg_embeds, train_neg_pooled,
            effective_grad_accum, clip_lt, clip_gt,
            ppo_steps=ppo_steps, aux_steps=aux_steps, all_steps=all_steps,
            selected_steps=selected_steps,
        )
        training_time_s = self._sync_time() - training_start

        epoch_metrics.update(
            self._build_stage_time_metrics(
                rollout_s=rollout_time_s,
                training_s=training_time_s,
                fake_update_s=fake_update_time_s,
            )
        )
        epoch_metrics["grpo_global_step"] = float(self._grpo_global_step)
        return epoch_metrics

    # ------------------------------------------------------------------
    # Fake score training
    # ------------------------------------------------------------------

    def _train_fake_score_from_x0(
        self,
        samples: dict,
        ls_cfg,
        autocast_ctx,
        aux_steps: set[int] | None,
    ) -> dict[str, float]:
        """Train fake score on sampled x0 from Stage 1 trajectories.

        Builds x0 training sources from aux steps using sampling-time noise_preds:
            x0 = x_t - sigma_t * v_pred
        Then trains fake_score with sub-batched CPU→GPU transfers to avoid OOM.

        NOTE: Uses ls_cfg.fake_* fields for timestep sampling, NOT the
        non-fake variants (which are for DMD forward diffusion in aux loss).
        """
        device = torch.device("cuda")
        prompt_base_cpu = samples["prompt_embeds"]  # [N, seq, dim], CPU
        pooled_base_cpu = samples["pooled_prompt_embeds"]  # [N, ...], CPU
        num_per_step = prompt_base_cpu.shape[0]

        step_indices = sorted(aux_steps) if aux_steps else []
        if not step_indices:
            # Safety fallback: still train on final x0 if aux_steps is empty.
            T = samples["timesteps"].shape[1]
            step_indices = [T - 1]

        if "noise_preds" not in samples:
            raise RuntimeError(
                "Missing 'noise_preds' in samples. This should not happen because "
                "sampling is forced to cache policy outputs in RTDMDTrainer."
            )

        if samples["noise_preds"].shape[1] < max(step_indices) + 1:
            raise RuntimeError(
                f"'noise_preds' has insufficient T dimension "
                f"({samples['noise_preds'].shape[1]}) for required steps {step_indices}"
            )

        # Reuse sampling-time policy outputs (exactly matches sampled trajectory policy)
        x0_sources: list[torch.Tensor] = []
        for j in step_indices:
            t0 = samples["timesteps"][0, j].to(device)
            step_idx = self.sde_scheduler.index_for_timestep(t0)
            sigma_j = self.sde_scheduler.sigmas[step_idx].item()
            x_t = samples["latents"][:, j].float()
            v_pred = samples["noise_preds"][:, j].float()
            x0_j = x_t - sigma_j * v_pred
            x0_sources.append(x0_j.cpu())

        x0_all_cpu = torch.cat(x0_sources, dim=0)
        prompt_embeds_cpu = torch.cat([prompt_base_cpu for _ in step_indices], dim=0)
        pooled_embeds_cpu = torch.cat([pooled_base_cpu for _ in step_indices], dim=0)
        N = x0_all_cpu.shape[0]

        sub_bs = ls_cfg.fake_score_batch_size
        if sub_bs <= 0 or sub_bs >= N:
            sub_bs = N

        self.fake_score_net.train()
        self.optimizers["fake_score"].zero_grad(set_to_none=True)

        total_loss = 0.0
        num_chunks = (N + sub_bs - 1) // sub_bs

        for c in range(num_chunks):
            s = c * sub_bs
            e = min(s + sub_bs, N)
            B = e - s

            # CPU → GPU per chunk
            x0_chunk = x0_all_cpu[s:e].to(device)
            pe_chunk = prompt_embeds_cpu[s:e].to(device)
            pp_chunk = pooled_embeds_cpu[s:e].to(device)

            # Timestep sampling — use fake_* fields
            t_indices = self._sample_timesteps_static(
                B, ls_cfg.fake_min_step, ls_cfg.fake_max_step,
                sampling=ls_cfg.fake_timestep_sampling,
                logit_mean=ls_cfg.fake_logit_mean,
                logit_std=ls_cfg.fake_logit_std,
                device=device,
            )

            # Sigmas and noisy latents
            sigma_tp = self._dmd_sigmas[t_indices].view(B, 1, 1, 1)
            timestep_tp = self._sigma_to_model_timestep(sigma_tp.view(B))
            noise = torch.randn_like(x0_chunk)
            noisy_latents = (1.0 - sigma_tp) * x0_chunk + sigma_tp * noise

            # Weighting
            weighting = compute_loss_weighting_for_sd3(
                weighting_scheme=ls_cfg.fake_timestep_sampling, sigmas=sigma_tp.view(B, 1, 1, 1),
            )

            # Forward (reuse the same autocast policy as PPO branch)
            with autocast_ctx():
                v_pred = self._predict_noise(
                    self.fake_score_net, noisy_latents, pe_chunk, timestep_tp,
                    pp_chunk, guidance_scale=1.0,
                )

            # Flow matching target: v = noise - x0
            target = noise - x0_chunk
            chunk_loss = torch.mean(
                (weighting.float() * (v_pred.float() - target.float()) ** 2).reshape(B, -1),
                dim=1,
            ).mean()

            # Scale by chunk fraction
            (chunk_loss * B / N).backward()
            total_loss += chunk_loss.item() * B

        # Step
        if self.config.solver.fake_score.max_grad_norm > 0:
            self._clip_grad_norm(
                self.fake_score_net,
                self.config.solver.fake_score.max_grad_norm,
            )
        self.optimizers["fake_score"].step()
        self.schedulers["fake_score"].step()
        mean_loss = total_loss / max(1, N)
        logger.debug(f"  Fake score loss: {mean_loss:.6f}")
        metrics = {
            "loss_fake": mean_loss,
            "lr": self.schedulers["fake_score"].get_last_lr()[0],
            "x0_source_steps": float(len(step_indices)),
            "x0_source_samples": float(N),
        }
        if dist.is_initialized() and get_world_size() > 1:
            for k, v in metrics.items():
                t = torch.tensor(v, device=device)
                dist.all_reduce(t, op=dist.ReduceOp.SUM)
                metrics[k] = t.item() / get_world_size()
        return metrics

    # ------------------------------------------------------------------
    # Training stage (full override with PPO + aux branching)
    # ------------------------------------------------------------------
    # IMPORTANT: This is a full copy of GRPOTrainer._grpo_training_stage
    # with added aux_loss branching. Keep in sync with parent on:
    #   - KL reference order (before policy forward)
    #   - Gradient accumulation and sync semantics
    #   - Metric collection and wandb logging

    def _grpo_dmd_training_stage(
        self,
        grpo_cfg,
        autocast_ctx,
        samples: dict,
        train_neg_embeds: torch.Tensor,
        train_neg_pooled: torch.Tensor,
        effective_grad_accum: int,
        clip_lt: float,
        clip_gt: float,
        ppo_steps: set[int] | None = None,
        aux_steps: set[int] | None = None,
        all_steps: list[int] | None = None,
        selected_steps: list[int] | None = None,
    ) -> None:
        """PPO + DMD / AC-DMD training with explicit branching.

        For steps in ``ppo_steps``: full PPO loss (ratio, advantages, clipping).
        For steps in ``aux_steps``: DMD / AC-DMD auxiliary loss.
        Steps in both sets get combined loss.
        """
        device = torch.device("cuda")
        total_batch_size = samples["timesteps"].shape[0]
        num_batches = grpo_cfg.num_batches_per_epoch

        for inner_epoch in range(grpo_cfg.num_inner_epochs):
            perm = torch.randperm(total_batch_size)
            for k in samples:
                samples[k] = samples[k][perm]

            sub_batch_size = total_batch_size // num_batches
            samples_batched = {
                k: v.reshape(num_batches, sub_batch_size, *v.shape[1:])
                for k, v in samples.items()
            }
            batched_list = [
                {k: v[b_idx] for k, v in samples_batched.items()}
                for b_idx in range(num_batches)
            ]

            self.generator.train()
            info = defaultdict(list)
            accum_count = 0
            self.optimizers["generator"].zero_grad(set_to_none=True)

            for batch in batched_list:
                batch = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                B = batch["prompt_embeds"].shape[0]

                if grpo_cfg.cfg_in_training:
                    embeds = torch.cat([train_neg_embeds[:B], batch["prompt_embeds"]], dim=0)
                    pooled = torch.cat([train_neg_pooled[:B], batch["pooled_prompt_embeds"]], dim=0)
                else:
                    embeds = batch["prompt_embeds"]
                    pooled = batch["pooled_prompt_embeds"]

                for j in all_steps:
                    is_ppo = j in ppo_steps
                    is_aux = j in aux_steps
                    accum_count += 1
                    is_sync_step = (accum_count % effective_grad_accum == 0)

                    # === KL reference (only for PPO, BEFORE policy forward) ===
                    prev_mean_ref = None
                    kl_loss = torch.tensor(0.0, device=device)
                    if is_ppo and grpo_cfg.beta > 0:
                        with torch.no_grad():
                            if self._lora_enabled:
                                prev_mean_ref = self._compute_reference_mean_lora(
                                    batch, j, embeds, pooled, grpo_cfg, autocast_ctx,
                                )
                            else:
                                prev_mean_ref = self._compute_reference_mean_fullweight(
                                    batch, j, embeds, pooled, grpo_cfg, autocast_ctx,
                                )

                    # === Generator forward (shared by PPO and aux) ===
                    with autocast_ctx():
                        if grpo_cfg.cfg_in_training:
                            noise_pred = self._forward_transformer(
                                self.generator,
                                torch.cat([batch["latents"][:, j]] * 2, dim=0),
                                torch.cat([batch["timesteps"][:, j]] * 2, dim=0),
                                embeds, pooled,
                            )
                            uncond, cond = noise_pred.chunk(2)
                            noise_pred = uncond + grpo_cfg.guidance_scale * (cond - uncond)
                        else:
                            noise_pred = self._forward_transformer(
                                self.generator,
                                batch["latents"][:, j],
                                batch["timesteps"][:, j],
                                embeds, pooled,
                            )

                    # === PPO branch (only for ppo_steps) ===
                    ppo_loss = torch.tensor(0.0, device=device)
                    if is_ppo:
                        _prev_sample, log_prob, prev_mean, std_dev_t = sde_step_with_logprob(
                            self.sde_scheduler,
                            noise_pred.float(),
                            batch["timesteps"][:, j],
                            batch["latents"][:, j].float(),
                            prev_sample=batch["next_latents"][:, j].float(),
                            noise_level=grpo_cfg.noise_level,
                            sde_type=grpo_cfg.sde_type,
                        )
                        if prev_mean_ref is not None and std_dev_t.min() > 0:
                            kl_loss = (
                                ((prev_mean - prev_mean_ref) ** 2)
                                .mean(dim=(1, 2, 3), keepdim=True)
                                / (2 * std_dev_t ** 2)
                            )
                            kl_loss = torch.mean(kl_loss)

                        adv = torch.clamp(
                            batch["advantages"][:, j],
                            -grpo_cfg.adv_clip_max, grpo_cfg.adv_clip_max,
                        )

                        ratio = torch.exp(log_prob - batch["log_probs"][:, j])
                        unclipped_loss = -adv * ratio

                        clipped_loss = -adv * torch.clamp(
                            ratio, 1.0 - clip_lt, 1.0 + clip_gt,
                        )
                        policy_loss = torch.mean(
                            torch.maximum(unclipped_loss, clipped_loss)
                        )

                        ppo_loss = policy_loss + grpo_cfg.beta * kl_loss

                        # PPO metrics
                        info["policy_loss"].append(policy_loss.detach())
                        info["ratio_mean"].append(ratio.mean().detach())
                        info["ratio_std"].append(ratio.std().detach())
                        info["ratio_max"].append(ratio.max().detach())
                        info["ratio_min"].append(ratio.min().detach())
                        info["log_prob"].append(log_prob.mean().detach())
                        info["old_log_prob"].append(batch["log_probs"][:, j].mean().detach())
                        info["approx_kl"].append(
                            0.5 * torch.mean((log_prob - batch["log_probs"][:, j]) ** 2).detach()
                        )
                        info["advantage_mean"].append(adv.mean().detach())
                        info["advantage_std"].append(adv.std().detach())
                        info["clipfrac"].append(
                            torch.mean((torch.abs(ratio - 1.0) > grpo_cfg.clip_range).float()).detach()
                        )
                        info["clipfrac_gt_one"].append(
                            torch.mean((ratio - 1.0 > clip_gt).float()).detach()
                        )
                        info["clipfrac_lt_one"].append(
                            torch.mean((1.0 - ratio > clip_lt).float()).detach()
                        )
                        if grpo_cfg.beta > 0:
                            info["kl_loss"].append(kl_loss.detach())

                    # === Aux branch (only for aux_steps) ===
                    aux_loss = torch.tensor(0.0, device=device)
                    if is_aux:
                        aux_loss, aux_metrics = self._compute_aux_loss(
                            batch, j, noise_pred, autocast_ctx,
                        )
                        for mk, mv in aux_metrics.items():
                            info[f"last_step/{mk}"].append(
                                torch.tensor(float(mv), device=device)
                            )

                    # === Combined backward ===
                    loss = ppo_loss + aux_loss
                    info["loss"].append(loss.detach())
                    info["aux_loss"].append(aux_loss.detach())

                    if is_sync_step:
                        sync_ctx = contextlib.nullcontext()
                    else:
                        needs_nosync = isinstance(self.generator, (FSDP, DDP))
                        sync_ctx = (
                            self.generator.no_sync()
                            if needs_nosync
                            else contextlib.nullcontext()
                        )

                    with sync_ctx:
                        (loss / effective_grad_accum).backward()

                    # === Optimizer step at sync point ===
                    if is_sync_step:
                        if grpo_cfg.max_grad_norm > 0:
                            self._clip_grad_norm(self.generator, grpo_cfg.max_grad_norm)
                        self.optimizers["generator"].step()
                        self.schedulers["generator"].step()
                        self.optimizers["generator"].zero_grad(set_to_none=True)

                        step_info = {
                            k: torch.mean(torch.stack(v)).item()
                            for k, v in info.items()
                        }

                        if dist.is_initialized() and get_world_size() > 1:
                            for k, v in step_info.items():
                                t = torch.tensor(v, device=device)
                                dist.all_reduce(t, op=dist.ReduceOp.SUM)
                                step_info[k] = t.item() / get_world_size()

                        # Wandb sections
                        loss_keys = {"policy_loss", "loss", "aux_loss", "kl_loss"}
                        ratio_keys = {
                            "ratio_mean", "ratio_std", "ratio_max", "ratio_min",
                            "clipfrac", "clipfrac_gt_one", "clipfrac_lt_one",
                            "approx_kl",
                        }
                        adv_keys = {"advantage_mean", "advantage_std"}
                        prob_keys = {"log_prob", "old_log_prob"}
                        aux_keys = {k for k in step_info if k.startswith("last_step/")}

                        sections = {
                            "grpo_loss": {k: v for k, v in step_info.items() if k in loss_keys},
                            "grpo_ratio": {k: v for k, v in step_info.items() if k in ratio_keys},
                            "grpo_advantage": {k: v for k, v in step_info.items() if k in adv_keys},
                            "grpo_prob": {k: v for k, v in step_info.items() if k in prob_keys},
                        }
                        if aux_keys:
                            sections["grpo_aux"] = {k: v for k, v in step_info.items() if k in aux_keys}

                        sections["grpo_loss"]["lr"] = self.schedulers["generator"].get_last_lr()[0]

                        if self.wandb_logger:
                            self.wandb_logger.log_multi_section(
                                sections, step=self._grpo_global_step,
                            )

                        if is_main_process():
                            parts = [f"step={self._grpo_global_step}", f"epoch={self._grpo_epoch}"]
                            for k, v in step_info.items():
                                parts.append(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}")
                            logger.info(" | ".join(parts))

                        self._grpo_global_step += 1
                        self.global_step = self._grpo_global_step
                        info = defaultdict(list)

                if self._ema is not None:
                    self._ema.step(self._trainable_params, self._grpo_global_step)

    # ------------------------------------------------------------------
    # Compute auxiliary loss
    # ------------------------------------------------------------------

    def _compute_aux_loss(
        self,
        batch: dict,
        j: int,
        noise_pred: torch.Tensor,
        autocast_ctx,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute base aux loss + optional BP loss for pipeline step ``j``.

        Uses ``batch["prompt_embeds"]`` directly (NOT the cfg-doubled
        ``embeds`` tensor) to ensure correct conditioning.

        Args:
            batch: Training batch dict (GPU tensors).
            j: Pipeline step index.
            noise_pred: Generator's velocity prediction at step j (has grad).
            autocast_ctx: Autocast context from the training loop.

        Returns:
            (loss, metrics_dict): Loss tensor (has grad) and logging metrics.
        """
        ls_cfg = self.config.grpo.last_step_loss
        loss_type = str(ls_cfg.loss_type).lower()
        device = noise_pred.device
        base_loss = torch.tensor(0.0, device=device)
        base_metrics: dict[str, float] = {}

        if loss_type == "ac_dmd":
            base_loss, base_metrics = self._compute_aux_loss_ac_dmd(
                batch=batch,
                j=j,
                noise_pred=noise_pred,
                autocast_ctx=autocast_ctx,
            )
        elif loss_type == "dmd":
            B = batch["prompt_embeds"].shape[0]

            # Recover x0 (grad flows through noise_pred → generator)
            step_idx = self.sde_scheduler.index_for_timestep(batch["timesteps"][0, j])
            sigma_j = self.sde_scheduler.sigmas[step_idx].view(1, 1, 1, 1).to(device)
            x0 = batch["latents"][:, j].float() - sigma_j * noise_pred.float()

            # Forward diffuse x0 to random t'
            t_indices = self._sample_timesteps_static(
                B, ls_cfg.min_step, ls_cfg.max_step,
                sampling=ls_cfg.timestep_sampling,
                logit_mean=ls_cfg.logit_mean,
                logit_std=ls_cfg.logit_std,
                device=device,
            )
            sigma_tp = self._dmd_sigmas[t_indices].view(B, 1, 1, 1)
            noise = torch.randn_like(x0)
            noisy_x = (1.0 - sigma_tp) * x0 + sigma_tp * noise
            timestep_tp = self._sigma_to_model_timestep(sigma_tp.view(B))

            # Teacher CFG (broadcast for multi-GPU consistency)
            cfg_tensor = torch.empty(1, device=device).uniform_(
                ls_cfg.real_guidance_scale_min, ls_cfg.real_guidance_scale_max,
            )
            if dist.is_initialized():
                dist.broadcast(cfg_tensor, src=0)
            teacher_cfg = cfg_tensor.item()

            uncond_embeds, uncond_pooled = self._get_uncond_embeds(B)

            # Use CORRECT prompt embeds (not cfg-doubled embeds/pooled)
            prompt_embeds = batch["prompt_embeds"]
            pooled_embeds = batch["pooled_prompt_embeds"]

            # Teacher prediction (no_grad, using passed autocast_ctx)
            with torch.no_grad():
                with autocast_ctx():
                    real_v = self._predict_noise(
                        self.teacher, noisy_x, prompt_embeds, timestep_tp, pooled_embeds,
                        guidance_scale=teacher_cfg,
                        uncond_text_embeddings=uncond_embeds,
                        uncond_pooled_prompt_embeds=uncond_pooled,
                    )
            real_x0 = (noisy_x - sigma_tp * real_v).float().detach()
            free_fsdp_unsharded_params(self.teacher)

            with torch.no_grad():
                with autocast_ctx():
                    fake_v = self._predict_noise(
                        self.fake_score_net, noisy_x, prompt_embeds, timestep_tp,
                        pooled_embeds,
                        guidance_scale=ls_cfg.fake_guidance_scale,
                        uncond_text_embeddings=(
                            uncond_embeds if ls_cfg.fake_guidance_scale > 1.0 else None
                        ),
                        uncond_pooled_prompt_embeds=(
                            uncond_pooled if ls_cfg.fake_guidance_scale > 1.0 else None
                        ),
                    )
            fake_x0 = (noisy_x - sigma_tp * fake_v).float().detach()
            free_fsdp_unsharded_params(self.fake_score_net)
            base_loss, dmd_metrics = dmd_loss(
                x0, fake_x0, real_x0, normalize=ls_cfg.gradient_normalization,
            )
            dmd_metrics = {k: v for k, v in dmd_metrics.items()}
            base_loss = ls_cfg.weight * base_loss
            dmd_metrics["loss"] = base_loss.item()
            dmd_metrics["loss_dm_weighted"] = base_loss.item()
            base_metrics = dmd_metrics
        elif loss_type == "none":
            base_loss = torch.tensor(0.0, device=device)
            base_metrics = {}
        else:
            raise ValueError(
                "Unsupported last_step_loss.loss_type in _compute_aux_loss: "
                f"{ls_cfg.loss_type!r}"
            )

        if not self._bp_enabled():
            return base_loss, base_metrics

        bp_loss, bp_metrics = self._compute_bp_loss(
            batch=batch,
            j=j,
            noise_pred=noise_pred,
        )
        total_loss = base_loss + bp_loss
        merged_metrics = {k: float(v) for k, v in base_metrics.items()}
        merged_metrics.update({k: float(v) for k, v in bp_metrics.items()})
        merged_metrics["loss_base"] = float(base_loss.detach().item())
        merged_metrics["loss_bp_scaled"] = float(bp_loss.detach().item())
        merged_metrics["loss"] = float(total_loss.detach().item())
        return total_loss, merged_metrics

    # ------------------------------------------------------------------
    # Checkpoint hooks
    # ------------------------------------------------------------------

    def _build_last_step_loss_signature(self) -> dict[str, Any]:
        ls_cfg = self.config.grpo.last_step_loss
        return {
            "model_type": str(self.model_type),
            "enabled": bool(ls_cfg.enabled),
            "loss_type": self._normalized_last_step_loss_type(),
            "has_fake_score": bool(self._has_fake_score),
            "train_steps": sorted(ls_cfg.train_steps),
            "sde_type": str(self.config.grpo.sde_type),
            "fake_score_lora_enabled": (
                bool(self.config.model.fake_score_lora.enabled)
                if self._has_fake_score else False
            ),
            # BP-related fields are part of training objective and must be
            # checkpoint-compatible to avoid silent resume behavior drift.
            "bp_enabled": bool(ls_cfg.bp_enabled),
            "bp_use_training_reward_fn": bool(ls_cfg.bp_use_training_reward_fn),
            "bp_only_deterministic": bool(ls_cfg.bp_only_deterministic),
            "bp_reward_model": str(ls_cfg.bp_reward_model),
            "bp_reward_margin": float(ls_cfg.bp_reward_margin),
            "bp_grad_scale": float(ls_cfg.bp_grad_scale),
            "bp_token_max_length": int(ls_cfg.bp_token_max_length),
            "bp_batch_size": int(getattr(ls_cfg, "bp_batch_size", 0)),
        }

    def _get_extra_checkpoint_state(self) -> dict:
        extra = super()._get_extra_checkpoint_state()
        extra["last_step_loss_signature"] = self._build_last_step_loss_signature()
        return extra

    def _restore_extra_checkpoint_state(self, extra: dict) -> None:
        super()._restore_extra_checkpoint_state(extra)
        saved = extra.get("last_step_loss_signature")
        if saved is None:
            return
        if not isinstance(saved, dict):
            raise ValueError(
                "Invalid checkpoint field last_step_loss_signature: expected dict, "
                f"got {type(saved).__name__}"
            )

        current = self._build_last_step_loss_signature()
        saved_norm = dict(saved)
        if "loss_type" in saved_norm:
            saved_norm["loss_type"] = str(saved_norm["loss_type"]).lower()

        unknown_saved_keys = sorted(set(saved_norm.keys()) - set(current.keys()))
        if unknown_saved_keys:
            raise ValueError(
                "Checkpoint last_step_loss signature contains unknown keys "
                f"{unknown_saved_keys}. Saved={saved_norm}, current={current}"
            )

        common_keys = sorted(set(saved_norm.keys()) & set(current.keys()))
        saved_common = {k: saved_norm[k] for k in common_keys}
        current_common = {k: current[k] for k in common_keys}
        if saved_common != current_common:
            raise ValueError(
                "Checkpoint last_step_loss config mismatch on shared keys:\n"
                f"  saved(common):   {saved_common}\n"
                f"  current(common): {current_common}\n"
                f"  saved(all):      {saved_norm}\n"
                f"  current(all):    {current}"
            )

        missing_saved_keys = sorted(set(current.keys()) - set(saved_norm.keys()))
        if missing_saved_keys:
            logger.warning(
                "  last_step_loss signature missing keys %s in checkpoint; "
                "likely an older checkpoint format. Shared-key validation passed.",
                missing_saved_keys,
            )
        logger.info(f"  last_step_loss signature validated: {current_common}")
