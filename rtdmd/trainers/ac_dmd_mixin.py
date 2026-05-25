"""
Ambient-Consistent DMD (AC-DMD) sub-interval objective mixin for RTDMDTrainer.

This mixin provides the AC-DMD sub-interval objective: starting from a
boundary (x_t, v_pred) on the sampled trajectory, transition to a
sub-interval start (x_s, sigma_start), re-noise to a renoise timestep
(xt_noised, sigma_end), compute the theoretical fake-score target via
``flow_forward_target_v2``, and optionally apply a consistency loss along
short fake-score rollouts within the sub-interval.

It also provides the matching AC-DMD aux loss for the deterministic CPS
step in the GRPO training stage and the fake-score training routine that
consumes the cached rollout (latents/next_latents/noise_preds) produced by
``GRPOTrainer``.

The inheriting class must provide ``self.config``, ``self._autocast()``,
``self.fake_score_net``, ``self.teacher``, ``self.sde_scheduler``,
``self._dmd_sigmas``, ``self.optimizers``, ``self.schedulers``,
``self._predict_noise``, ``self._get_uncond_embeds``,
``self._sigma_to_model_timestep``, ``self._model_timestep_to_sigma``,
``self._num_train_timesteps``, ``self._lookup_step_sigmas``,
``self._compute_grad_norm``, and ``self._clip_grad_norm``.

Methods here are kept byte-equivalent to the original RTDMDTrainer
implementation (only the class indentation context changes).
"""

from __future__ import annotations

import logging
from collections import defaultdict

import torch
import torch.distributed as dist
import torch.nn.functional as F

from rtdmd.parallel.utils import free_fsdp_unsharded_params, get_world_size
from rtdmd.trainers.ac_dmd_utils import (
    compute_inverse_scale_weight,
    flow_forward_target_v2,
    flow_forward_v2,
    scheduler_transition_step,
)
from rtdmd.trainers.dmd_trainer import dmd_loss

logger = logging.getLogger(__name__)


class ACDMDMixin:
    """Ambient-Consistent DMD (AC-DMD) sub-interval objective mixin.

    See module docstring for the sub-interval objective, the consistency
    loss, and the trainer attributes that must be available on the
    inheriting class.
    """

    def _init_ac_state(self) -> None:
        """Initialize all AC-DMD-related runtime state on ``self``.

        Must be called once from the inheriting class' ``__init__`` so the
        AC-DMD code paths see consistent initial values regardless of
        whether AC-DMD is enabled at config time.
        """
        self._ac_consistency_deterministic_warned = False

    def _is_ac_dmd(self) -> bool:
        return self._normalized_last_step_loss_type() == "ac_dmd"

    def _ac_transition_is_stochastic(self) -> bool:
        return self.config.dmd.cps_eta > 0.0

    def _ac_step_bounds(self, ls_cfg) -> tuple[int, int]:
        """Return valid timestep clamp range for renoising."""
        min_step = max(0, int(ls_cfg.ac_min_step), int(ls_cfg.min_step))
        max_step = min(
            int(ls_cfg.ac_max_step),
            int(ls_cfg.max_step),
            int(self._num_train_timesteps()) - 1,
        )
        if max_step < min_step:
            raise ValueError(
                "Invalid sub-interval timestep bounds in last_step_loss: "
                f"ac_range=[{ls_cfg.ac_min_step}, {ls_cfg.ac_max_step}], "
                f"dmd_range=[{ls_cfg.min_step}, {ls_cfg.max_step}]"
            )
        return min_step, max_step

    def _sample_ac_timesteps(
        self,
        interval_start_timestep: torch.Tensor,
        *,
        ls_cfg,
    ) -> torch.Tensor:
        interval_end = float(ls_cfg.ac_end_timestep)
        interval_start = interval_start_timestep.float() + float(ls_cfg.ac_timestep_offset)
        interval_end_tensor = torch.full_like(interval_start, interval_end)
        interval_end_tensor = torch.maximum(interval_end_tensor, interval_start + 1.0)
        min_step, max_step = self._ac_step_bounds(ls_cfg)
        sampled = torch.rand_like(interval_start)
        sampled = sampled * (interval_end_tensor - interval_start) + interval_start
        sampled = torch.clamp(
            sampled,
            min=float(min_step),
            max=float(max_step),
        )
        # Keep timestep sampling local to each rank. Broadcasting this tensor
        # is fragile because batch shapes can differ across ranks after
        # rank-local filtering, which can deadlock NCCL collectives.
        return sampled

    def _ac_metrics(
        self,
        *,
        sigma_start_value: float,
        renoise_timestep: torch.Tensor,
        scale: torch.Tensor,
    ) -> dict[str, float]:
        return {
            "ac_sigma_start": float(sigma_start_value) * float(self._num_train_timesteps()),
            "ac_timestep_start": float(
                self._sigma_to_model_timestep(float(sigma_start_value))
            ),
            "ac_timestep_mean": renoise_timestep.float().mean().item(),
            "ac_scale_mean": scale.float().mean().item(),
            "ac_scale_min": scale.float().min().item(),
            "ac_scale_max": scale.float().max().item(),
        }

    def _build_ac_subinterval(
        self,
        *,
        x_t: torch.Tensor,
        v_pred_boundary: torch.Tensor,
        sigma_cur_value: float,
        sigma_next_value: float,
        ls_cfg,
        x_s_override: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | float]:
        """Build AC-DMD (x_s, x_t') tensors from one rollout boundary."""
        start_mode = str(ls_cfg.ac_start_mode).lower()
        if start_mode == "next_sigma":
            sigma_start_value = sigma_next_value
        elif start_mode == "fixed_boundary":
            sigma_boundary = max(0.0, min(float(ls_cfg.ac_fixed_boundary_sigma), 1.0))
            sigma_start_value = min(sigma_cur_value, sigma_boundary)
        else:
            raise ValueError(
                f"Unknown last_step_loss.ac_start_mode={start_mode!r}. "
                "Expected 'next_sigma' or 'fixed_boundary'."
            )

        sigma_start_value = max(0.0, min(float(sigma_start_value), 1.0))
        sigma_cur = torch.full(
            (x_t.shape[0],), float(sigma_cur_value), device=x_t.device, dtype=x_t.dtype
        )
        sigma_start = torch.full(
            (x_t.shape[0],), float(sigma_start_value), device=x_t.device, dtype=x_t.dtype
        )

        if x_s_override is not None:
            x_s = x_s_override
        else:
            x_s = scheduler_transition_step(
                sample=x_t,
                model_pred=v_pred_boundary,
                sigma_from=sigma_cur,
                sigma_to=sigma_start,
                scheduler_type=self.config.dmd.generator_scheduler,
                cps_eta=self.config.dmd.cps_eta,
                noise=(
                    torch.randn_like(x_t)
                    if self._ac_transition_is_stochastic()
                    else None
                ),
            )

        interval_start_timestep = self._sigma_to_model_timestep(sigma_start).float()
        renoise_timestep = self._sample_ac_timesteps(
            interval_start_timestep,
            ls_cfg=ls_cfg,
        )
        sigma_end = self._model_timestep_to_sigma(renoise_timestep).to(dtype=x_t.dtype)

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
            move_scale_out=ls_cfg.ac_target_formula_move_scale_out,
        )
        return {
            "x_s": x_s,
            "xt_noised": xt_noised,
            "v_target": v_target,
            "scale": scale,
            "renoise_timestep": renoise_timestep,
            "sigma_start": sigma_start,
            "sigma_end": sigma_end,
            "sigma_start_value": sigma_start_value,
        }

    def _compute_ac_consistency_loss(
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
        ls_cfg,
        fake_train_guidance_scale: float,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Consistency loss for AC-DMD fake-score updates."""
        zero = v_pred_end.new_zeros(())
        if (not ls_cfg.ac_consistency_enabled) or ls_cfg.ac_consistency_weight <= 0.0:
            return zero, {}

        t_anchor = renoise_timestep.float()
        sigma_anchor = sigma_end.to(dtype=xt_noised.dtype)
        sigma_anchor_nd = sigma_anchor.float().view(
            sigma_anchor.shape[0], *([1] * (xt_noised.ndim - 1))
        )
        h_anchor = xt_noised.float() - sigma_anchor_nd * v_pred_end.float()

        max_steps_diff = max(
            1,
            int(round(float(ls_cfg.ac_consistency_epsilon_timestep))),
        )
        num_consistency_steps = max(1, int(ls_cfg.ac_consistency_num_steps))
        use_two_sample = bool(ls_cfg.ac_consistency_use_two_sample_unbiased)
        use_stochastic_transition = self._ac_transition_is_stochastic()

        if (
            use_two_sample
            and (not use_stochastic_transition)
            and (not self._ac_consistency_deterministic_warned)
        ):
            logger.warning(
                "ac_consistency_use_two_sample_unbiased=True but transition is "
                "deterministic (%s, cps_eta=%.3f). Two-sample paths may collapse.",
                self.config.dmd.generator_scheduler,
                self.config.dmd.cps_eta,
            )
            self._ac_consistency_deterministic_warned = True

        def _predict_fake_velocity(sample: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
            with self._autocast():
                return self._predict_noise(
                    self.fake_score_net,
                    sample,
                    prompt_embeds,
                    timestep,
                    pooled_embeds,
                    guidance_scale=fake_train_guidance_scale,
                    uncond_text_embeddings=(
                        uncond_embeds if fake_train_guidance_scale > 1.0 else None
                    ),
                    uncond_pooled_prompt_embeds=(
                        uncond_pooled if fake_train_guidance_scale > 1.0 else None
                    ),
                )

        def _sample_prev_timestep(t_curr: torch.Tensor) -> torch.Tensor:
            step_diffs = torch.randint(
                low=1,
                high=max_steps_diff + 1,
                size=t_curr.shape,
                device=t_curr.device,
                dtype=torch.long,
            )
            # Keep random step offsets rank-local to avoid shape-dependent
            # cross-rank collectives in this inner training loop.
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
                scheduler_type=self.config.dmd.generator_scheduler,
                cps_eta=self.config.dmd.cps_eta,
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

        first_step_pred = (
            v_pred_end.detach()
            if ls_cfg.ac_consistency_stopgrad_transitions
            else v_pred_end
        )
        with torch.set_grad_enabled(not ls_cfg.ac_consistency_stopgrad_transitions):
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

    def _train_fake_score_ac_from_rollout(
        self,
        samples: dict,
        ls_cfg,
        autocast_ctx,
        aux_steps: set[int] | None,
    ) -> dict[str, float]:
        """Train fake score with AC-DMD sub-interval objective from rollout cache."""
        device = torch.device("cuda")
        prompt_base_cpu = samples["prompt_embeds"]
        pooled_base_cpu = samples["pooled_prompt_embeds"]  # [N, ...], CPU
        num_per_step = prompt_base_cpu.shape[0]

        step_indices = sorted(aux_steps) if aux_steps else []
        if not step_indices:
            T = samples["timesteps"].shape[1]
            step_indices = [T - 1]

        if "noise_preds" not in samples:
            raise RuntimeError(
                "Missing 'noise_preds' in samples for AC-DMD fake-score training."
            )

        if samples["noise_preds"].shape[1] < max(step_indices) + 1:
            raise RuntimeError(
                f"'noise_preds' has insufficient T dimension "
                f"({samples['noise_preds'].shape[1]}) for required steps {step_indices}"
            )

        sub_bs = ls_cfg.fake_score_batch_size
        if sub_bs <= 0 or sub_bs >= num_per_step:
            sub_bs = num_per_step

        total_samples = max(1, num_per_step * len(step_indices))
        fake_train_cfg = float(ls_cfg.ac_fake_train_guidance_scale)
        start_mode = str(ls_cfg.ac_start_mode).lower()

        self.fake_score_net.train()
        self.optimizers["fake_score"].zero_grad(set_to_none=True)

        agg: defaultdict[str, float] = defaultdict(float)

        for j in step_indices:
            t0 = samples["timesteps"][0, j].to(device)
            sigma_cur_value, sigma_next_value = self._lookup_step_sigmas(t0)

            x_t_step_cpu = samples["latents"][:, j].float()
            x_next_step_cpu = samples["next_latents"][:, j].float()
            v_step_cpu = samples["noise_preds"][:, j].float()

            num_chunks = (num_per_step + sub_bs - 1) // sub_bs
            for c in range(num_chunks):
                s = c * sub_bs
                e = min(s + sub_bs, num_per_step)
                B = e - s

                x_t_chunk = x_t_step_cpu[s:e].to(device)
                x_next_chunk = x_next_step_cpu[s:e].to(device)
                v_step_chunk = v_step_cpu[s:e].to(device)
                pe_chunk = prompt_base_cpu[s:e].to(device)
                pp_chunk = pooled_base_cpu[s:e].to(device)

                ac_data = self._build_ac_subinterval(
                    x_t=x_t_chunk,
                    v_pred_boundary=v_step_chunk,
                    sigma_cur_value=sigma_cur_value,
                    sigma_next_value=sigma_next_value,
                    ls_cfg=ls_cfg,
                    x_s_override=x_next_chunk if start_mode == "next_sigma" else None,
                )
                xt_noised = ac_data["xt_noised"]
                v_target = ac_data["v_target"]
                ac_scale = ac_data["scale"]
                renoise_timestep = ac_data["renoise_timestep"]
                sigma_end = ac_data["sigma_end"]
                sigma_start_value = float(ac_data["sigma_start_value"])
                assert isinstance(xt_noised, torch.Tensor)
                assert isinstance(v_target, torch.Tensor)
                assert isinstance(ac_scale, torch.Tensor)
                assert isinstance(renoise_timestep, torch.Tensor)
                assert isinstance(sigma_end, torch.Tensor)

                uncond_embeds, uncond_pooled = self._get_uncond_embeds(B)
                with autocast_ctx():
                    v_pred = self._predict_noise(
                        self.fake_score_net,
                        xt_noised,
                        pe_chunk,
                        renoise_timestep,
                        pp_chunk,
                        guidance_scale=fake_train_cfg,
                        uncond_text_embeddings=(
                            uncond_embeds if fake_train_cfg > 1.0 else None
                        ),
                        uncond_pooled_prompt_embeds=(
                            uncond_pooled if fake_train_cfg > 1.0 else None
                        ),
                    )

                loss_raw = F.mse_loss(
                    v_pred.float() * ac_scale.float(),
                    v_target.float(),
                    reduction="none",
                )
                inv_scale_weight = compute_inverse_scale_weight(
                    scale=ac_scale,
                    clamp_min=ls_cfg.ac_fake_loss_weight_clamp_min,
                    clamp_max=ls_cfg.ac_fake_loss_weight_clamp_max,
                )
                loss_fake_ac = (inv_scale_weight * loss_raw).mean()
                grad_scale = B / total_samples
                cons_weight = float(ls_cfg.ac_consistency_weight)
                if ls_cfg.ac_consistency_enabled and cons_weight > 0.0:
                    loss_consistency, cons_metrics = self._compute_ac_consistency_loss(
                        xt_noised=xt_noised,
                        renoise_timestep=renoise_timestep,
                        sigma_end=sigma_end,
                        v_pred_end=v_pred,
                        prompt_embeds=pe_chunk,
                        pooled_embeds=pp_chunk,
                        uncond_embeds=uncond_embeds,
                        uncond_pooled=uncond_pooled,
                        ls_cfg=ls_cfg,
                        fake_train_guidance_scale=fake_train_cfg,
                    )
                else:
                    loss_consistency = loss_fake_ac.new_zeros(())
                    cons_metrics = {}

                loss_fake = loss_fake_ac + cons_weight * loss_consistency
                (loss_fake * grad_scale).backward()

                agg["loss_fake"] += loss_fake.item() * B
                agg["loss_fake_ac"] += loss_fake_ac.item() * B
                agg["loss_consistency"] += loss_consistency.item() * B
                agg["consistency_weight"] += float(ls_cfg.ac_consistency_weight) * B
                for mk, mv in self._ac_metrics(
                    sigma_start_value=sigma_start_value,
                    renoise_timestep=renoise_timestep,
                    scale=ac_scale,
                ).items():
                    agg[mk] += float(mv) * B
                for mk, mv in cons_metrics.items():
                    agg[mk] += float(mv) * B

        if self.config.solver.fake_score.max_grad_norm > 0:
            self._clip_grad_norm(
                self.fake_score_net,
                self.config.solver.fake_score.max_grad_norm,
            )
        self.optimizers["fake_score"].step()
        self.schedulers["fake_score"].step()

        metrics = {
            k: v / total_samples
            for k, v in agg.items()
        }
        metrics["x0_source_steps"] = float(len(step_indices))
        metrics["x0_source_samples"] = float(total_samples)
        metrics["lr"] = self.schedulers["fake_score"].get_last_lr()[0]
        if dist.is_initialized() and get_world_size() > 1:
            for k, v in metrics.items():
                t = torch.tensor(float(v), device=device)
                dist.all_reduce(t, op=dist.ReduceOp.SUM)
                metrics[k] = t.item() / get_world_size()
        return metrics

    def _compute_aux_loss_ac_dmd(
        self,
        batch: dict,
        j: int,
        noise_pred: torch.Tensor,
        autocast_ctx,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute AC-DMD auxiliary loss for one pipeline step."""
        ls_cfg = self.config.grpo.last_step_loss
        device = noise_pred.device
        B = batch["prompt_embeds"].shape[0]

        sigma_cur_value, sigma_next_value = self._lookup_step_sigmas(
            batch["timesteps"][0, j]
        )
        x_t = batch["latents"][:, j].float()
        ac_data = self._build_ac_subinterval(
            x_t=x_t,
            v_pred_boundary=noise_pred.float(),
            sigma_cur_value=sigma_cur_value,
            sigma_next_value=sigma_next_value,
            ls_cfg=ls_cfg,
            x_s_override=None,
        )
        x_s = ac_data["x_s"]
        xt_noised = ac_data["xt_noised"]
        renoise_timestep = ac_data["renoise_timestep"]
        sigma_start = ac_data["sigma_start"]
        sigma_end = ac_data["sigma_end"]
        sigma_start_value = float(ac_data["sigma_start_value"])
        assert isinstance(x_s, torch.Tensor)
        assert isinstance(xt_noised, torch.Tensor)
        assert isinstance(renoise_timestep, torch.Tensor)
        assert isinstance(sigma_start, torch.Tensor)
        assert isinstance(sigma_end, torch.Tensor)

        cfg_tensor = torch.empty(1, device=device).uniform_(
            ls_cfg.real_guidance_scale_min,
            ls_cfg.real_guidance_scale_max,
        )
        if dist.is_initialized():
            dist.broadcast(cfg_tensor, src=0)
        teacher_cfg = cfg_tensor.item()

        uncond_embeds, uncond_pooled = self._get_uncond_embeds(B)
        prompt_embeds = batch["prompt_embeds"]
        pooled_embeds = batch["pooled_prompt_embeds"]

        with torch.no_grad():
            with autocast_ctx():
                pred_real_noise = self._predict_noise(
                    self.teacher,
                    xt_noised,
                    prompt_embeds,
                    renoise_timestep,
                    pooled_embeds,
                    guidance_scale=teacher_cfg,
                    uncond_text_embeddings=uncond_embeds,
                    uncond_pooled_prompt_embeds=uncond_pooled,
                )

            transition_noise = (
                torch.randn_like(xt_noised)
                if self._ac_transition_is_stochastic()
                else None
            )
            xs_teacher = scheduler_transition_step(
                sample=xt_noised,
                model_pred=pred_real_noise,
                sigma_from=sigma_end,
                sigma_to=sigma_start,
                scheduler_type=self.config.dmd.generator_scheduler,
                cps_eta=self.config.dmd.cps_eta,
                noise=transition_noise,
            )
        free_fsdp_unsharded_params(self.teacher)

        with torch.no_grad():
            with autocast_ctx():
                pred_fake_noise = self._predict_noise(
                    self.fake_score_net,
                    xt_noised,
                    prompt_embeds,
                    renoise_timestep,
                    pooled_embeds,
                    guidance_scale=ls_cfg.fake_guidance_scale,
                    uncond_text_embeddings=(
                        uncond_embeds if ls_cfg.fake_guidance_scale > 1.0 else None
                    ),
                    uncond_pooled_prompt_embeds=(
                        uncond_pooled if ls_cfg.fake_guidance_scale > 1.0 else None
                    ),
                )
            xs_fake = scheduler_transition_step(
                sample=xt_noised,
                model_pred=pred_fake_noise,
                sigma_from=sigma_end,
                sigma_to=sigma_start,
                scheduler_type=self.config.dmd.generator_scheduler,
                cps_eta=self.config.dmd.cps_eta,
                noise=transition_noise,
            )
        free_fsdp_unsharded_params(self.fake_score_net)

        loss_dm, dm_metrics = dmd_loss(
            x_s,
            xs_fake,
            xs_teacher,
            normalize=ls_cfg.gradient_normalization,
        )
        loss = ls_cfg.weight * loss_dm
        dm_metrics = {k: v for k, v in dm_metrics.items()}
        dm_metrics["loss"] = loss.item()
        dm_metrics["loss_dm_weighted"] = loss.item()
        dm_metrics["teacher_cfg"] = teacher_cfg
        dm_metrics.update(
            self._ac_metrics(
                sigma_start_value=sigma_start_value,
                renoise_timestep=renoise_timestep,
                scale=ac_data["scale"],
            )
        )
        return loss, dm_metrics
