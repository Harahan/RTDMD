"""
AC-DMD Trainer (V3 theoretical, single-expert) for RTDMD.

This trainer extends DMDTrainer with AC-DMD sub-interval training:
1. Extract a boundary sample x_s from generator's backward simulation.
2. Forward-noise x_s to x_t in a sampled sub-interval [s, t].
3. Train fake score on the theoretical flow target in that sub-interval.
4. Train generator with DMD loss defined on the boundary sample x_s using
   one-step projections from x_t back to x_s.

The implementation uses the "theoretical" target style (move_scale_out=False,
clamp max=10) while keeping RTDMD's single generator/fake/teacher architecture.
"""

from __future__ import annotations

import logging

import torch
import torch.distributed as dist
import torch.nn.functional as F

from rtdmd.config import RTDMDConfig
from rtdmd.trainers.dmd_trainer import DMDTrainer, dmd_loss
from rtdmd.trainers.ac_dmd_utils import (
    compute_inverse_scale_weight,
    flow_forward_target_v2,
    flow_forward_v2,
    scheduler_transition_step,
)

logger = logging.getLogger(__name__)


class ACDMDTrainer(DMDTrainer):
    """DMD trainer with V3-theoretical AC-DMD objective (single expert)."""

    def __init__(self, config: RTDMDConfig):
        super().__init__(config)
        self._ac_boundary_warned = False
        self._consistency_deterministic_warned = False

    def _ac_step_bounds(self) -> tuple[int, int]:
        """Return valid timestep clamp range for renoising."""
        dmd_cfg = self.config.dmd
        ac_cfg = self.config.ac_dmd
        num_train_timesteps = int(self._num_train_timesteps())
        min_step = max(0, int(ac_cfg.min_renoise_step), int(dmd_cfg.min_step))
        max_step = min(
            int(ac_cfg.max_renoise_step),
            int(dmd_cfg.max_step),
            num_train_timesteps - 1,
        )
        if max_step < min_step:
            raise ValueError(
                "Invalid sub-interval timestep bounds: "
                f"min_renoise_step={ac_cfg.min_renoise_step}, "
                f"max_renoise_step={ac_cfg.max_renoise_step}, "
                f"dmd_range=[{dmd_cfg.min_step}, {dmd_cfg.max_step}]"
            )
        return min_step, max_step

    def _ac_boundary_sigma(self) -> float:
        """Select boundary sigma from denoising schedule by index."""
        denoising_sigmas = self._denoising_sigmas
        if not denoising_sigmas:
            raise RuntimeError("Denoising sigmas are not initialized.")
        idx_cfg = self.config.ac_dmd.boundary_sigma_index
        idx = max(0, min(int(idx_cfg), len(denoising_sigmas) - 1))
        if idx != idx_cfg and not self._ac_boundary_warned:
            logger.warning(
                "ac_dmd.boundary_sigma_index=%s out of range; clamped to %s.",
                idx_cfg,
                idx,
            )
            self._ac_boundary_warned = True
        return float(denoising_sigmas[idx])

    def _sample_ac_timesteps(
        self,
        interval_start_timestep: float,
        batch_size: int,
        device: torch.device,
        all_include_terminal: bool,
    ) -> torch.Tensor:
        """Sample renoising timesteps in [interval_start + offset, interval_end]."""
        ac_cfg = self.config.ac_dmd
        dmd_cfg = self.config.dmd
        interval_end = float(ac_cfg.end_timestep)
        if all_include_terminal:
            interval_end = float(self._num_train_timesteps())
        interval_start = float(interval_start_timestep) + float(ac_cfg.timestep_offset)
        if interval_end <= interval_start:
            interval_end = interval_start + 1.0
        sampled = torch.rand(batch_size, device=device, dtype=torch.float32)
        sampled = sampled * (interval_end - interval_start) + interval_start
        min_step, max_step = self._ac_step_bounds()
        sampled = torch.clamp(sampled, min=float(min_step), max=float(max_step))
        # Keep sub-interval timestep sampling rank-synchronized so all ranks follow
        # identical control flow around boundary logic.
        if dist.is_initialized():
            dist.broadcast(sampled, src=0)
        return sampled

    def _prepare_ac_subinterval(
        self,
        prompt_embeds: torch.Tensor,
        pooled_embeds: torch.Tensor,
        uncond_embeds: torch.Tensor,
        uncond_pooled: torch.Tensor,
        *,
        model: torch.nn.Module | None = None,
    ) -> dict[str, torch.Tensor | float | int]:
        """Build AC-DMD training tensors: x_s, x_t, target, scale, and metadata."""
        dmd_cfg = self.config.dmd
        ac_cfg = self.config.ac_dmd

        x0_gen, actual_steps, last_step_data = self._generate_latents(
            prompt_embeds,
            pooled_embeds,
            uncond_embeds,
            uncond_pooled,
            num_steps=dmd_cfg.num_inference_steps,
            random_stop=dmd_cfg.random_stop_idx,
            model=model,
            return_last_step_data=True,
        )
        if last_step_data is None:
            raise RuntimeError("Expected last_step_data from _generate_latents.")

        x_t = last_step_data["x_t"]
        v_pred_last = last_step_data["v_pred"]
        sigma_cur_value = float(last_step_data["sigma"])
        start_mode = str(getattr(ac_cfg, "start_mode", "next_sigma")).lower()

        if start_mode == "next_sigma":
            # Strict paper-style one-step-one-segment:
            # the start sigma is the next sigma of the executed last denoising step.
            sigma_start_value = float(last_step_data["sigma_next"])
            x_next = last_step_data.get("x_next")
            if isinstance(x_next, torch.Tensor):
                x_s = x_next
            else:
                # Safety fallback (for older checkpoints/code paths): rebuild x_s
                # by one transition from (x_t, v_pred_last) to sigma_next.
                sigma_cur = torch.full(
                    (x_t.shape[0],), sigma_cur_value, device=x_t.device, dtype=x_t.dtype
                )
                sigma_start = torch.full(
                    (x_t.shape[0],), sigma_start_value, device=x_t.device, dtype=x_t.dtype
                )
                x_s = scheduler_transition_step(
                    sample=x_t,
                    model_pred=v_pred_last,
                    sigma_from=sigma_cur,
                    sigma_to=sigma_start,
                    scheduler_type=dmd_cfg.generator_scheduler,
                    cps_eta=dmd_cfg.cps_eta,
                )
        elif start_mode == "fixed_boundary":
            sigma_boundary = self._ac_boundary_sigma()
            sigma_start_value = min(sigma_cur_value, sigma_boundary)
            sigma_cur = torch.full(
                (x_t.shape[0],), sigma_cur_value, device=x_t.device, dtype=x_t.dtype
            )
            sigma_start = torch.full(
                (x_t.shape[0],), sigma_start_value, device=x_t.device, dtype=x_t.dtype
            )
            x_s = scheduler_transition_step(
                sample=x_t,
                model_pred=v_pred_last,
                sigma_from=sigma_cur,
                sigma_to=sigma_start,
                scheduler_type=dmd_cfg.generator_scheduler,
                cps_eta=dmd_cfg.cps_eta,
            )
        else:
            raise ValueError(
                f"Unknown ac_dmd.start_mode={start_mode!r}. "
                "Expected 'next_sigma' or 'fixed_boundary'."
            )

        # Numeric safety in downstream sub-interval timestep sampling.
        sigma_start_value = max(0.0, min(float(sigma_start_value), 1.0))
        sigma_start = torch.full(
            (x_s.shape[0],), sigma_start_value, device=x_s.device, dtype=x_s.dtype
        )

        all_include_terminal = (
            (not dmd_cfg.random_stop_idx)
            and ac_cfg.all_include_terminal_when_no_random_stop
        )
        interval_start_timestep = float(self._sigma_to_model_timestep(sigma_start_value))
        renoise_timestep = self._sample_ac_timesteps(
            interval_start_timestep=interval_start_timestep,
            batch_size=x_s.shape[0],
            device=x_s.device,
            all_include_terminal=all_include_terminal,
        )
        sigma_end = self._model_timestep_to_sigma(renoise_timestep).to(dtype=x_s.dtype)

        noise = torch.randn_like(x_s)
        xt_noised = flow_forward_v2(
            sigma_start=sigma_start,
            sigma_end=sigma_end,
            x_start=x_s,
            noise=noise,
        )
        v_target, scale = flow_forward_target_v2(
            sigma_start=sigma_start,
            sigma_end=sigma_end,
            x_start=x_s,
            noise=noise,
            move_scale_out=ac_cfg.target_formula_move_scale_out,
        )

        return {
            "x0_gen": x0_gen,
            "x_s": x_s,
            "xt_noised": xt_noised,
            "v_target": v_target,
            "scale": scale,
            "renoise_timestep": renoise_timestep,
            "sigma_start": sigma_start,
            "sigma_end": sigma_end,
            "actual_steps": actual_steps,
            "sigma_start_value": sigma_start_value,
        }

    def _ac_metrics(
        self,
        ac_data: dict[str, torch.Tensor | float | int],
    ) -> dict[str, float]:
        """Build lightweight AC-DMD diagnostics for logging."""
        renoise_timestep = ac_data["renoise_timestep"]
        scale = ac_data["scale"]
        assert isinstance(renoise_timestep, torch.Tensor)
        assert isinstance(scale, torch.Tensor)
        sigma_start_step = float(
            self._sigma_to_model_timestep(float(ac_data["sigma_start_value"]))
        )
        return {
            # Kept for backward compatibility with existing dashboards.
            "ac_sigma_start": float(ac_data["sigma_start_value"]) * float(
                self._num_train_timesteps()
            ),
            # Transformer model-timestep equivalent of interval start.
            "ac_timestep_start": sigma_start_step,
            "ac_timestep_mean": renoise_timestep.float().mean().item(),
            "ac_scale_mean": scale.float().mean().item(),
            "ac_scale_min": scale.float().min().item(),
            "ac_scale_max": scale.float().max().item(),
        }

    def _compute_consistency_loss(
        self,
        *,
        xt_noised: torch.Tensor,
        renoise_timestep: torch.Tensor,
        sigma_end: torch.Tensor,
        v_pred_end: torch.Tensor,
        prompt_embeds: torch.Tensor,
        pooled_embeds: torch.Tensor,
        uncond_embeds: torch.Tensor,
        uncond_pooled: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute the optional AC-DMD consistency loss.

        This term is added on top of the AC-DMD fake-score loss:
            L_fake_total = L_fake_ac + w_cons * L_consistency

        Two-sample mode follows the same core estimator style:
            E[(h(x_s^1, s) - h(x_t, t)) * (h(x_s^2, s) - h(x_t, t))]
        where x_s^1 and x_s^2 are independently sampled transitions that share
        the same sampled timestep trajectory.
        """
        ac_cfg = self.config.ac_dmd
        dmd_cfg = self.config.dmd
        zero = v_pred_end.new_zeros(())
        if (not ac_cfg.consistency_enabled) or ac_cfg.consistency_weight <= 0.0:
            return zero, {}

        t_anchor = renoise_timestep.float()
        sigma_anchor = sigma_end.to(dtype=xt_noised.dtype)
        sigma_anchor_nd = sigma_anchor.float().view(
            sigma_anchor.shape[0], *([1] * (xt_noised.ndim - 1))
        )
        h_anchor = xt_noised.float() - sigma_anchor_nd * v_pred_end.float()

        max_steps_diff = max(1, int(round(float(ac_cfg.consistency_epsilon_timestep))))
        num_consistency_steps = max(
            1, int(getattr(ac_cfg, "consistency_num_steps", 1))
        )
        use_two_sample = bool(ac_cfg.consistency_use_two_sample_unbiased)

        use_stochastic_transition = dmd_cfg.cps_eta > 0.0
        fake_train_cfg = dmd_cfg.fake_train_guidance_scale

        if use_two_sample and (not use_stochastic_transition) and (
            not self._consistency_deterministic_warned
        ):
            logger.warning(
                "consistency_use_two_sample_unbiased=True but CPS transition is "
                "deterministic (cps_eta=%.3f). Two-sample paths may collapse.",
                dmd_cfg.cps_eta,
            )
            self._consistency_deterministic_warned = True

        def _predict_fake_velocity(
            sample: torch.Tensor,
            timestep: torch.Tensor,
        ) -> torch.Tensor:
            with self._autocast():
                return self._predict_noise(
                    self.fake_score_net,
                    sample,
                    prompt_embeds,
                    timestep,
                    pooled_embeds,
                    guidance_scale=fake_train_cfg,
                    uncond_text_embeddings=uncond_embeds if fake_train_cfg > 1.0 else None,
                    uncond_pooled_prompt_embeds=uncond_pooled if fake_train_cfg > 1.0 else None,
                )

        def _sample_prev_timestep(t_curr: torch.Tensor) -> torch.Tensor:
            step_diffs = torch.randint(
                low=1,
                high=max_steps_diff + 1,
                size=t_curr.shape,
                device=t_curr.device,
                dtype=torch.long,
            )
            if dist.is_initialized():
                dist.broadcast(step_diffs, src=0)
            return torch.clamp(t_curr - step_diffs.to(dtype=t_curr.dtype), min=0.0)

        def _transition(
            sample: torch.Tensor,
            model_pred: torch.Tensor,
            sigma_from: torch.Tensor,
            sigma_to: torch.Tensor,
        ) -> torch.Tensor:
            return scheduler_transition_step(
                sample=sample,
                model_pred=model_pred,
                sigma_from=sigma_from,
                sigma_to=sigma_to,
                scheduler_type=dmd_cfg.generator_scheduler,
                cps_eta=dmd_cfg.cps_eta,
                noise=torch.randn_like(sample) if use_stochastic_transition else None,
            )

        def _rollout_path(
            *,
            x_init: torch.Tensor,
            t_init: torch.Tensor,
            sigma_init: torch.Tensor,
            fixed_t_sequence: list[torch.Tensor] | None = None,
            first_step_pred: torch.Tensor | None = None,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor]]:
            x_curr = x_init
            t_curr = t_init
            sigma_curr = sigma_init
            sampled_t_sequence: list[torch.Tensor] = []
            for step_idx in range(num_consistency_steps):
                if step_idx == 0 and first_step_pred is not None:
                    v_pred_curr = first_step_pred
                else:
                    v_pred_curr = _predict_fake_velocity(x_curr, t_curr)

                if fixed_t_sequence is None:
                    t_next = _sample_prev_timestep(t_curr)
                else:
                    t_next = fixed_t_sequence[step_idx]

                sigma_next = self._model_timestep_to_sigma(t_next).to(dtype=xt_noised.dtype)
                x_next = _transition(
                    sample=x_curr,
                    model_pred=v_pred_curr,
                    sigma_from=sigma_curr,
                    sigma_to=sigma_next,
                )
                sampled_t_sequence.append(t_next)
                x_curr = x_next
                t_curr = t_next
                sigma_curr = sigma_next

            return x_curr, t_curr, sigma_curr, sampled_t_sequence

        # Keep anchor differentiable. When stop-grad transitions are enabled, we
        # can reuse the already-computed anchor prediction for rollout step 0 to
        # avoid a redundant fake-model forward pass.
        first_step_pred = (
            v_pred_end.detach()
            if ac_cfg.consistency_stopgrad_transitions
            else v_pred_end
        )
        with torch.set_grad_enabled(not ac_cfg.consistency_stopgrad_transitions):
            x_prime_1, t_prime, sigma_prime, sampled_t_sequence = _rollout_path(
                x_init=xt_noised,
                t_init=t_anchor,
                sigma_init=sigma_anchor,
                fixed_t_sequence=None,
                first_step_pred=first_step_pred,
            )

            x_prime_2 = None
            if use_two_sample:
                x_prime_2, _, _, _ = _rollout_path(
                    x_init=xt_noised,
                    t_init=t_anchor,
                    sigma_init=sigma_anchor,
                    fixed_t_sequence=sampled_t_sequence,
                    first_step_pred=first_step_pred,
                )

        v_pred_prime_1 = _predict_fake_velocity(x_prime_1, t_prime)
        sigma_prime_nd = sigma_prime.float().view(
            sigma_prime.shape[0], *([1] * (x_prime_1.ndim - 1))
        )
        h_prime_1 = x_prime_1.float() - sigma_prime_nd * v_pred_prime_1.float()

        if use_two_sample and x_prime_2 is not None:
            v_pred_prime_2 = _predict_fake_velocity(x_prime_2, t_prime)
            h_prime_2 = x_prime_2.float() - sigma_prime_nd * v_pred_prime_2.float()
            prod = (h_prime_1 - h_anchor) * (h_prime_2 - h_anchor)
            loss_consistency = prod.flatten(1).mean(dim=1).mean()
        else:
            loss_consistency = F.mse_loss(h_prime_1, h_anchor, reduction="mean")

        loss_consistency = torch.nan_to_num(
            loss_consistency.float(),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        metrics = {
            "consistency_t_anchor_mean": t_anchor.mean().item(),
            "consistency_t_prime_mean": t_prime.mean().item(),
            "consistency_num_steps": float(num_consistency_steps),
            "consistency_max_steps_diff": float(max_steps_diff),
            "consistency_stochastic_transition": float(use_stochastic_transition),
        }
        if use_two_sample:
            metrics["consistency_t_double_mean"] = t_prime.mean().item()
        return loss_consistency, metrics

    def _update_fake_score(
        self,
        prompt_embeds: torch.Tensor,
        pooled_embeds: torch.Tensor,
        uncond_embeds: torch.Tensor,
        uncond_pooled: torch.Tensor,
    ) -> dict[str, float]:
        """Train fake score with AC-DMD theoretical target on [s, t]."""
        if not self.config.ac_dmd.enabled:
            return super()._update_fake_score(
                prompt_embeds, pooled_embeds, uncond_embeds, uncond_pooled
            )

        dmd_cfg = self.config.dmd
        ac_cfg = self.config.ac_dmd

        with torch.no_grad():
            ac_data = self._prepare_ac_subinterval(
                prompt_embeds, pooled_embeds, uncond_embeds, uncond_pooled
            )
            xt_noised = ac_data["xt_noised"]
            v_target = ac_data["v_target"]
            scale = ac_data["scale"]
            renoise_timestep = ac_data["renoise_timestep"]
            sigma_end = ac_data["sigma_end"]
        assert isinstance(xt_noised, torch.Tensor)
        assert isinstance(v_target, torch.Tensor)
        assert isinstance(scale, torch.Tensor)
        assert isinstance(renoise_timestep, torch.Tensor)
        assert isinstance(sigma_end, torch.Tensor)

        fake_train_cfg = dmd_cfg.fake_train_guidance_scale
        with self._autocast():
            v_pred = self._predict_noise(
                self.fake_score_net,
                xt_noised,
                prompt_embeds,
                renoise_timestep,
                pooled_embeds,
                guidance_scale=fake_train_cfg,
                uncond_text_embeddings=uncond_embeds if fake_train_cfg > 1.0 else None,
                uncond_pooled_prompt_embeds=uncond_pooled if fake_train_cfg > 1.0 else None,
            )

        loss_raw = F.mse_loss(
            (v_pred.float() * scale.float()),
            v_target.float(),
            reduction="none",
        )
        weight = compute_inverse_scale_weight(
            scale=scale,
            clamp_min=ac_cfg.fake_loss_weight_clamp_min,
            clamp_max=ac_cfg.fake_loss_weight_clamp_max,
        )
        loss_fake_ac = (weight * loss_raw).mean()
        loss_consistency, cons_metrics = self._compute_consistency_loss(
            xt_noised=xt_noised,
            renoise_timestep=renoise_timestep,
            sigma_end=sigma_end,
            v_pred_end=v_pred,
            prompt_embeds=prompt_embeds,
            pooled_embeds=pooled_embeds,
            uncond_embeds=uncond_embeds,
            uncond_pooled=uncond_pooled,
        )
        loss_fake = (
            loss_fake_ac
            + float(ac_cfg.consistency_weight) * loss_consistency
        )

        self.optimizers["fake_score"].zero_grad(set_to_none=True)
        loss_fake.backward()
        fake_grad_norm = self._compute_grad_norm(self.fake_score_net)
        if self.config.solver.fake_score.max_grad_norm > 0:
            self._clip_grad_norm(
                self.fake_score_net,
                self.config.solver.fake_score.max_grad_norm,
            )
        self.optimizers["fake_score"].step()
        self.schedulers["fake_score"].step()

        metrics = {
            "loss_fake": loss_fake.item(),
            "loss_fake_ac": loss_fake_ac.item(),
            "loss_consistency": loss_consistency.item(),
            "consistency_weight": float(ac_cfg.consistency_weight),
            "grad_norm": fake_grad_norm,
            "lr": self.schedulers["fake_score"].get_last_lr()[0],
            "gen_steps": float(ac_data["actual_steps"]),
        }
        metrics.update(cons_metrics)
        metrics.update(self._ac_metrics(ac_data))
        return metrics

    def _update_generator(
        self,
        prompt_embeds: torch.Tensor,
        pooled_embeds: torch.Tensor,
        uncond_embeds: torch.Tensor,
        uncond_pooled: torch.Tensor,
    ) -> dict[str, float]:
        """Update generator using AC-DMD objective on boundary sample x_s."""
        if not self.config.ac_dmd.enabled:
            return super()._update_generator(
                prompt_embeds, pooled_embeds, uncond_embeds, uncond_pooled
            )

        dmd_cfg = self.config.dmd

        ac_data = self._prepare_ac_subinterval(
            prompt_embeds, pooled_embeds, uncond_embeds, uncond_pooled
        )
        x_s = ac_data["x_s"]
        xt_noised = ac_data["xt_noised"]
        renoise_timestep = ac_data["renoise_timestep"]
        sigma_start = ac_data["sigma_start"]
        sigma_end = ac_data["sigma_end"]
        assert isinstance(x_s, torch.Tensor)
        assert isinstance(xt_noised, torch.Tensor)
        assert isinstance(renoise_timestep, torch.Tensor)
        assert isinstance(sigma_start, torch.Tensor)
        assert isinstance(sigma_end, torch.Tensor)

        with torch.no_grad():
            cfg_tensor = torch.empty(1, device=xt_noised.device).uniform_(
                dmd_cfg.real_guidance_scale_min,
                dmd_cfg.real_guidance_scale_max,
            )
            if dist.is_initialized():
                dist.broadcast(cfg_tensor, src=0)
            teacher_cfg = cfg_tensor.item()

            with self._autocast():
                pred_real_noise = self._predict_noise(
                    self.dmd_real_score_model,
                    xt_noised,
                    prompt_embeds,
                    renoise_timestep,
                    pooled_embeds,
                    guidance_scale=teacher_cfg,
                    uncond_text_embeddings=uncond_embeds,
                    uncond_pooled_prompt_embeds=uncond_pooled,
                )
            use_stochastic_transition = dmd_cfg.cps_eta > 0.0
            transition_noise = (
                torch.randn_like(xt_noised) if use_stochastic_transition else None
            )
            xs_teacher = scheduler_transition_step(
                sample=xt_noised,
                model_pred=pred_real_noise,
                sigma_from=sigma_end,
                sigma_to=sigma_start,
                scheduler_type=dmd_cfg.generator_scheduler,
                cps_eta=dmd_cfg.cps_eta,
                noise=transition_noise,
            )

        with torch.no_grad():
            with self._autocast():
                pred_fake_noise = self._predict_noise(
                    self.fake_score_net,
                    xt_noised,
                    prompt_embeds,
                    renoise_timestep,
                    pooled_embeds,
                    guidance_scale=dmd_cfg.fake_guidance_scale,
                    uncond_text_embeddings=uncond_embeds
                    if dmd_cfg.fake_guidance_scale > 1.0
                    else None,
                    uncond_pooled_prompt_embeds=uncond_pooled
                    if dmd_cfg.fake_guidance_scale > 1.0
                    else None,
                )
            xs_fake = scheduler_transition_step(
                sample=xt_noised,
                model_pred=pred_fake_noise,
                sigma_from=sigma_end,
                sigma_to=sigma_start,
                scheduler_type=dmd_cfg.generator_scheduler,
                cps_eta=dmd_cfg.cps_eta,
                noise=transition_noise,
            )

        loss_dm, dm_metrics = dmd_loss(
            x_s,
            xs_fake,
            xs_teacher,
            normalize=dmd_cfg.gradient_normalization,
        )

        self.optimizers["generator"].zero_grad(set_to_none=True)
        loss_dm.backward()
        generator_grad_norm = self._compute_grad_norm(self.generator)
        if self.config.solver.generator.max_grad_norm > 0:
            self._clip_grad_norm(
                self.generator,
                self.config.solver.generator.max_grad_norm,
            )
        self.optimizers["generator"].step()
        self.schedulers["generator"].step()

        dm_metrics["lr"] = self.schedulers["generator"].get_last_lr()[0]
        dm_metrics["teacher_cfg"] = teacher_cfg
        dm_metrics["gen_steps"] = float(ac_data["actual_steps"])
        dm_metrics["model_grad_norm"] = generator_grad_norm
        dm_metrics.update(self._ac_metrics(ac_data))
        return dm_metrics
