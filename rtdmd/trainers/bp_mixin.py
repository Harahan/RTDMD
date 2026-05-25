"""
Back-Propagation (BP) reward loss mixin for RTDMDTrainer.

This mixin provides differentiable BP reward loss support for deterministic
CPS steps. Two reward modes are supported:

1. ``training_reward_fn`` mode (default when ``bp_use_training_reward_fn=True``
   and at least one supported reward in ``grpo.reward_fn`` is non-zero):
   reuses the GRPO MultiScorer reward backends (``clipscore``, ``pickscore``,
   ``hpsv2``) on the differentiable image path.
2. ``imagereward`` fallback mode (used otherwise): loads the standalone
   ImageReward model and calls ``score_gard`` for a differentiable reward.

The inheriting class must provide ``self.config``, ``self.text_encoders``,
``self.tokenizers``, ``self.generator``, ``self.vae``, and ``self._autocast()``.

Methods here are kept byte-equivalent to the original RTDMDTrainer
implementation (only the class indentation context changes).
"""

from __future__ import annotations

import logging
import os
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from rtdmd.models.flux2_klein import unpatchify_flux2_latents

logger = logging.getLogger(__name__)


class BPMixin:
    """Differentiable BP reward loss mixin.

    See module docstring for the two reward modes and the trainer attributes
    that must be available on the inheriting class.
    """

    def _init_bp_state(self) -> None:
        """Initialize all BP-related runtime state on ``self``.

        Must be called once from the inheriting class' ``__init__`` so the
        BP code paths see consistent initial values regardless of whether BP
        is enabled at config time.
        """
        self._bp_warned_unsupported_rewards = False
        self._bp_reward_model = None
        self._bp_reward_device = None
        self._bp_train_reward_models: dict[str, Any] = {}
        self._bp_train_reward_weights: dict[str, float] = {}
        self._bp_train_reward_device = None
        self._bp_train_tokenizers: dict[str, Any] = {}
        self._bp_pickscore_image_tform = None

    def _validate_bp_config(self, ls_cfg) -> None:
        """Validate BP-specific fields on ``last_step_loss`` config.

        Mirrors the original inline validation block in ``setup_models``:
        checks ``bp_grad_scale``, ``bp_token_max_length``, ``bp_batch_size``
        ranges, and warns when ``bp_enabled`` + ``bp_use_training_reward_fn``
        is set without any supported training reward.
        """
        if ls_cfg.bp_grad_scale < 0:
            raise ValueError(
                f"last_step_loss.bp_grad_scale must be >= 0, got {ls_cfg.bp_grad_scale}"
            )
        if int(ls_cfg.bp_token_max_length) < 1:
            raise ValueError(
                "last_step_loss.bp_token_max_length must be >= 1, "
                f"got {ls_cfg.bp_token_max_length}"
            )
        bp_batch_size = int(getattr(ls_cfg, "bp_batch_size", 0))
        if bp_batch_size < 0:
            raise ValueError(
                "last_step_loss.bp_batch_size must be >= 0, "
                f"got {ls_cfg.bp_batch_size}"
            )
        if ls_cfg.bp_enabled and ls_cfg.bp_use_training_reward_fn:
            if not self._get_bp_training_reward_weights(require_nonempty=False):
                logger.warning(
                    "last_step_loss.bp_enabled=True and "
                    "bp_use_training_reward_fn=True but no supported training "
                    "reward found in grpo.reward_fn. BP will be disabled "
                    "for this run."
                )

    def _bp_enabled(self) -> bool:
        ls_cfg = self.config.grpo.last_step_loss
        enabled = bool(
            ls_cfg.enabled
            and ls_cfg.bp_enabled
            and float(ls_cfg.bp_grad_scale) > 0.0
        )
        if not enabled:
            return False

        # In training-reward mode, only supported rewards are used.
        # If none are supported, disable BP instead of falling back.
        if bool(getattr(ls_cfg, "bp_use_training_reward_fn", True)):
            return bool(self._get_bp_training_reward_weights(require_nonempty=False))
        return True

    @staticmethod
    def _supported_bp_training_rewards() -> set[str]:
        return {"pickscore", "clipscore", "hpsv2"}

    def _get_grpo_reward_backend(self, reward_name: str) -> Any | None:
        """Best-effort fetch of an already-loaded backend from GRPO MultiScorer."""
        scorer = getattr(self, "_scorer", None)
        if scorer is None:
            return None
        score_fns = getattr(scorer, "score_fns", None)
        if not isinstance(score_fns, dict):
            return None

        fn = score_fns.get(reward_name)
        if fn is None:
            return None
        if isinstance(fn, nn.Module):
            return fn

        closure = getattr(fn, "__closure__", None)
        if not closure:
            return None
        for cell in closure:
            try:
                obj = cell.cell_contents
            except ValueError:
                continue
            if isinstance(obj, nn.Module):
                return obj
        return None

    @staticmethod
    def _is_bp_backend_compatible(reward_name: str, backend: Any) -> bool:
        """Check whether a reused backend satisfies BP differentiable path needs."""
        required = {
            "clipscore": ("model", "processor", "_process"),
            "pickscore": ("model", "processor"),
            "hpsv2": ("model", "processor", "preprocess_val"),
        }
        attrs = required.get(reward_name, ())
        return all(hasattr(backend, attr) for attr in attrs)

    @staticmethod
    def _infer_module_device(module: nn.Module) -> torch.device | None:
        """Best-effort infer runtime device of a module."""
        try:
            return next(module.parameters()).device
        except StopIteration:
            pass
        except Exception:
            return None

        try:
            return next(module.buffers()).device
        except StopIteration:
            pass
        except Exception:
            return None

        attr_device = getattr(module, "device", None)
        if attr_device is None:
            return None
        try:
            return torch.device(attr_device)
        except Exception:
            return None

    def _get_bp_training_reward_weights(
        self,
        *,
        require_nonempty: bool = True,
    ) -> dict[str, float]:
        """Return supported non-zero reward weights from grpo.reward_fn."""
        reward_fn = getattr(self.config.grpo, "reward_fn", {}) or {}
        supported = self._supported_bp_training_rewards()
        weights: dict[str, float] = {}
        unsupported_nonzero = []
        for key, value in reward_fn.items():
            w = float(value)
            if abs(w) <= 0.0:
                continue
            if key in supported:
                weights[key] = w
            else:
                unsupported_nonzero.append(key)

        if unsupported_nonzero and not self._bp_warned_unsupported_rewards:
            logger.warning(
                "BP training-reward mode ignores unsupported rewards: %s. "
                "Supported now: %s",
                unsupported_nonzero,
                sorted(supported),
            )
            self._bp_warned_unsupported_rewards = True

        if require_nonempty and not weights:
            raise ValueError(
                "BP training-reward mode enabled, but grpo.reward_fn has "
                "no supported non-zero reward. Supported keys: "
                f"{sorted(supported)}"
            )
        return weights

    def _use_training_rewards_for_bp(self) -> bool:
        if not self._bp_enabled():
            return False
        ls_cfg = self.config.grpo.last_step_loss
        if not bool(getattr(ls_cfg, "bp_use_training_reward_fn", True)):
            return False
        return bool(self._get_bp_training_reward_weights(require_nonempty=False))

    def _init_bp_training_reward_tokenizers(self) -> None:
        """Initialize lightweight text tokenizers for BP training-reward mode."""
        if not self._use_training_rewards_for_bp():
            return
        if self._bp_train_tokenizers:
            return
        weights = self._get_bp_training_reward_weights(require_nonempty=True)

        tokenizers: dict[str, Any] = {}
        if "clipscore" in weights:
            backend = self._get_grpo_reward_backend("clipscore")
            if backend is not None and getattr(backend, "processor", None) is not None:
                tokenizers["clipscore"] = backend.processor
                logger.info("  BP reused GRPO clipscore tokenizer")
            else:
                from transformers import CLIPProcessor
                tokenizers["clipscore"] = CLIPProcessor.from_pretrained(
                    "openai/clip-vit-large-patch14"
                )
        if "pickscore" in weights:
            backend = self._get_grpo_reward_backend("pickscore")
            if backend is not None and getattr(backend, "processor", None) is not None:
                tokenizers["pickscore"] = backend.processor
                logger.info("  BP reused GRPO pickscore tokenizer")
            else:
                from transformers import AutoProcessor
                tokenizers["pickscore"] = AutoProcessor.from_pretrained(
                    "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
                )
        if "hpsv2" in weights:
            backend = self._get_grpo_reward_backend("hpsv2")
            if backend is not None and getattr(backend, "processor", None) is not None:
                tokenizers["hpsv2"] = backend.processor
                logger.info("  BP reused GRPO hpsv2 tokenizer")
            else:
                from hpsv2.src.open_clip import get_tokenizer
                tokenizers["hpsv2"] = get_tokenizer("ViT-H-14")

        self._bp_train_tokenizers = tokenizers
        logger.info(
            "  BP training-reward tokenizers initialized: %s",
            sorted(tokenizers.keys()),
        )

    def _encode_bp_training_prompts(self, prompts: list[str]) -> dict[str, torch.Tensor]:
        """Pre-tokenize prompts for differentiable training-reward forward pass."""
        self._init_bp_training_reward_tokenizers()
        if not self._bp_train_tokenizers:
            raise RuntimeError("BP training-reward tokenizers are not initialized")

        encoded: dict[str, torch.Tensor] = {}

        if "clipscore" in self._bp_train_tokenizers:
            processor = self._bp_train_tokenizers["clipscore"]
            text_inputs = processor(
                text=prompts,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            for key, value in text_inputs.items():
                encoded[f"bp_clipscore_{key}"] = value.contiguous()

        if "pickscore" in self._bp_train_tokenizers:
            processor = self._bp_train_tokenizers["pickscore"]
            text_inputs = processor(
                text=prompts,
                padding=True,
                truncation=True,
                max_length=77,
                return_tensors="pt",
            )
            for key, value in text_inputs.items():
                encoded[f"bp_pickscore_{key}"] = value.contiguous()

        if "hpsv2" in self._bp_train_tokenizers:
            tokenizer = self._bp_train_tokenizers["hpsv2"]
            token_ids = tokenizer(prompts)
            if not isinstance(token_ids, torch.Tensor):
                token_ids = torch.as_tensor(token_ids)
            encoded["bp_hpsv2_input_ids"] = token_ids.long().contiguous()

        return encoded

    def _init_bp_training_reward_models(
        self,
        device: torch.device | str = "cpu",
    ) -> None:
        """Load differentiable reward backends used by BP training-reward mode."""
        if not self._use_training_rewards_for_bp():
            return

        target_device = torch.device(device)
        if self._bp_train_reward_models:
            self._ensure_bp_training_reward_device(target_device)
            return

        from rtdmd.rewards.reward_ckpt_path import set_ckpt_path
        reward_ckpt_path = getattr(self.config.grpo, "reward_ckpt_path", "")
        if reward_ckpt_path:
            set_ckpt_path(reward_ckpt_path)

        weights = self._get_bp_training_reward_weights(require_nonempty=True)
        models: dict[str, Any] = {}

        for name in weights:
            backend = self._get_grpo_reward_backend(name)
            if backend is not None and self._is_bp_backend_compatible(name, backend):
                models[name] = backend
                logger.info("  BP reused GRPO reward backend: %s", name)
            elif backend is not None:
                logger.warning(
                    "  BP found GRPO backend for %s but it is incompatible with "
                    "differentiable path; fallback to standalone init.",
                    name,
                )

        if "clipscore" in weights and "clipscore" not in models:
            from rtdmd.rewards.clip_scorer import ClipScorer
            models["clipscore"] = ClipScorer(device=target_device)
        if "pickscore" in weights and "pickscore" not in models:
            from rtdmd.rewards.pickscore_scorer import PickScoreScorer
            models["pickscore"] = PickScoreScorer(
                device=str(target_device),
                dtype=torch.float32,
            )
        if "hpsv2" in weights and "hpsv2" not in models:
            from rtdmd.rewards.hpsv2_scorer import HPSv2Scorer
            models["hpsv2"] = HPSv2Scorer(dtype=torch.float32, device=target_device)

        for name, model in models.items():
            model.to(target_device)
            model.eval()
            model.requires_grad_(False)
            if hasattr(model, "device"):
                model.device = target_device
            logger.info("  BP training reward backend loaded: %s", name)

        self._bp_train_reward_models = models
        self._bp_train_reward_weights = weights
        self._bp_train_reward_device = target_device
        self._bp_pickscore_image_tform = None

    def _ensure_bp_training_reward_device(self, device: torch.device) -> None:
        if not self._bp_train_reward_models:
            self._init_bp_training_reward_models(device=device)
            return

        need_move = self._bp_train_reward_device != device
        if not need_move:
            # Reused backends may also be moved by self._scorer.to(cpu/cuda) between
            # sampling and training. Validate actual runtime device, not only cache.
            for model in self._bp_train_reward_models.values():
                model_device = self._infer_module_device(model)
                if model_device is not None and model_device != device:
                    need_move = True
                    break
        if not need_move:
            return

        for model in self._bp_train_reward_models.values():
            model.to(device)
            if hasattr(model, "device"):
                model.device = device
        self._bp_train_reward_device = device

    def _init_bp_reward_model(self, device: torch.device | str = "cpu") -> None:
        """Load ImageReward model used for differentiable BP loss."""
        if not self._bp_enabled():
            return

        target_device = torch.device(device)
        if self._bp_reward_model is None:
            ls_cfg = self.config.grpo.last_step_loss
            from rtdmd.rewards.imagereward_scorer import _load_imagereward_module
            from rtdmd.rewards.reward_ckpt_path import CKPT_PATH

            rm_module = _load_imagereward_module()
            download_root = os.path.join(
                os.path.expanduser(CKPT_PATH),
                "ImageReward",
            )
            self._bp_reward_model = rm_module.load(
                ls_cfg.bp_reward_model,
                device=str(target_device),
                download_root=download_root,
            )
            self._bp_reward_model.eval()
            self._bp_reward_model.requires_grad_(False)
            if hasattr(self._bp_reward_model, "device"):
                self._bp_reward_model.device = str(target_device)
            self._bp_reward_device = target_device
            logger.info(
                "  BP reward model loaded: %s on %s",
                ls_cfg.bp_reward_model,
                str(target_device),
            )
            return

        self._ensure_bp_reward_device(target_device)

    def _ensure_bp_reward_device(self, device: torch.device) -> None:
        """Move cached BP reward model to target device when needed."""
        if self._bp_reward_model is None:
            self._init_bp_reward_model(device=device)
            return
        if self._bp_reward_device != device:
            self._bp_reward_model.to(device)
            if hasattr(self._bp_reward_model, "device"):
                self._bp_reward_model.device = str(device)
            self._bp_reward_device = device

    def _encode_bp_prompts(self, prompts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """Tokenize prompts for ImageReward.score_gard forward."""
        self._init_bp_reward_model(device="cpu")
        if self._bp_reward_model is None:
            raise RuntimeError("BP reward model is not initialized")

        tokenizer = self._bp_reward_model.blip.tokenizer
        ls_cfg = self.config.grpo.last_step_loss
        text_inputs = tokenizer(
            prompts,
            padding="max_length",
            truncation=True,
            max_length=int(ls_cfg.bp_token_max_length),
            return_tensors="pt",
        )
        return text_inputs.input_ids.contiguous(), text_inputs.attention_mask.contiguous()

    def _decode_latents_to_tensor_with_grad(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latent tensors to [0,1] image tensors while keeping gradients."""
        device = latents.device
        if next(self.vae.parameters()).device != device:
            self.vae.to(device)

        def _decode_once(latents_chunk: torch.Tensor) -> torch.Tensor:
            if self._is_flux2_klein:
                latents_bn_mean = self.vae.bn.running_mean.view(1, -1, 1, 1).to(
                    latents_chunk.device,
                    latents_chunk.dtype,
                )
                latents_bn_std = torch.sqrt(
                    self.vae.bn.running_var.view(1, -1, 1, 1) + self.vae.config.batch_norm_eps
                ).to(latents_chunk.device, latents_chunk.dtype)

                scaled_latents = latents_chunk * latents_bn_std + latents_bn_mean
                scaled_latents = unpatchify_flux2_latents(scaled_latents)
                images_chunk = self.vae.decode(scaled_latents.to(self.vae.dtype)).sample
            else:
                scaling_factor = float(self.vae.config.scaling_factor)
                shift_factor = float(getattr(self.vae.config, "shift_factor", 0.0))
                scaled_latents = latents_chunk / scaling_factor + shift_factor
                images_chunk = self.vae.decode(scaled_latents.to(self.vae.dtype)).sample
            return (images_chunk / 2 + 0.5).clamp(0, 1)

        chunk_bs = self._get_bp_batch_size(latents.shape[0])
        if chunk_bs >= latents.shape[0]:
            return _decode_once(latents)

        image_chunks = []
        for s in range(0, latents.shape[0], chunk_bs):
            e = min(s + chunk_bs, latents.shape[0])
            image_chunks.append(_decode_once(latents[s:e]))
        return torch.cat(image_chunks, dim=0)

    def _get_bp_batch_size(self, total_batch_size: int) -> int:
        """Resolve BP chunk size (0/large means full batch)."""
        if total_batch_size <= 0:
            return total_batch_size
        ls_cfg = self.config.grpo.last_step_loss
        cfg_bs = int(getattr(ls_cfg, "bp_batch_size", 0))
        if cfg_bs <= 0 or cfg_bs >= total_batch_size:
            return total_batch_size
        return max(1, cfg_bs)

    @staticmethod
    def _preprocess_bp_images(images: torch.Tensor) -> torch.Tensor:
        """Apply BP image preprocessing for ImageReward-like backends."""
        images = F.interpolate(
            images.float(),
            size=(224, 224),
            mode="bicubic",
            align_corners=False,
        )
        mean = torch.tensor(
            [0.48145466, 0.4578275, 0.40821073],
            device=images.device,
            dtype=images.dtype,
        ).view(1, 3, 1, 1)
        std = torch.tensor(
            [0.26862954, 0.26130258, 0.27577711],
            device=images.device,
            dtype=images.dtype,
        ).view(1, 3, 1, 1)
        return (images - mean) / std

    def _compute_bp_loss(
        self,
        batch: dict,
        j: int,
        noise_pred: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute BP-style differentiable reward loss on one pipeline step."""
        ls_cfg = self.config.grpo.last_step_loss
        device = noise_pred.device
        zero = torch.tensor(0.0, device=device)

        if not self._use_training_rewards_for_bp():
            if "bp_input_ids" not in batch or "bp_attention_mask" not in batch:
                raise RuntimeError(
                    "Missing BP prompt tensors in batch: "
                    "expected keys 'bp_input_ids' and 'bp_attention_mask'."
                )

        _, sigma_next_value = self._lookup_step_sigmas(batch["timesteps"][0, j])
        is_deterministic = abs(float(sigma_next_value)) < 1e-12
        if ls_cfg.bp_only_deterministic and not is_deterministic:
            return zero, {
                "loss_bp_raw": 0.0,
                "loss_bp": 0.0,
                "bp_reward_mean": 0.0,
                "bp_reward_min": 0.0,
                "bp_reward_max": 0.0,
                "bp_applied": 0.0,
                "bp_is_deterministic_step": 0.0,
            }

        step_idx = self.sde_scheduler.index_for_timestep(batch["timesteps"][0, j])
        sigma_j = self.sde_scheduler.sigmas[step_idx].view(1, 1, 1, 1).to(device)
        x0 = batch["latents"][:, j].float() - sigma_j * noise_pred.float()

        images = self._decode_latents_to_tensor_with_grad(x0)
        if self._use_training_rewards_for_bp():
            weighted_reward, reward_metrics = self._compute_bp_training_weighted_reward(
                images=images,
                batch=batch,
            )
            # Keep objective form aligned with ImageReward fallback path.
            loss_bp_raw = F.relu(
                float(ls_cfg.bp_reward_margin) - weighted_reward
            ).mean()
            metrics = {
                "loss_bp_raw": float(loss_bp_raw.detach().item()),
                "bp_reward_weighted_mean": float(weighted_reward.detach().mean().item()),
                "bp_reward_weighted_min": float(weighted_reward.detach().min().item()),
                "bp_reward_weighted_max": float(weighted_reward.detach().max().item()),
                "bp_applied": 1.0,
                "bp_is_deterministic_step": float(is_deterministic),
            }
            metrics.update(reward_metrics)
        else:
            bp_images = self._preprocess_bp_images(images)

            self._ensure_bp_reward_device(device)
            if self._bp_reward_model is None:
                raise RuntimeError("BP reward model is not initialized")

            rm_input_ids = batch["bp_input_ids"].to(device=device, dtype=torch.long)
            rm_attention_mask = batch["bp_attention_mask"].to(device=device, dtype=torch.long)
            model_dtype = next(self._bp_reward_model.parameters()).dtype
            chunk_bs = self._get_bp_batch_size(bp_images.shape[0])
            if chunk_bs >= bp_images.shape[0]:
                rewards = self._bp_reward_model.score_gard(
                    rm_input_ids,
                    rm_attention_mask,
                    bp_images.to(dtype=model_dtype),
                )
            else:
                reward_chunks = []
                for s in range(0, bp_images.shape[0], chunk_bs):
                    e = min(s + chunk_bs, bp_images.shape[0])
                    reward_chunks.append(
                        self._bp_reward_model.score_gard(
                            rm_input_ids[s:e],
                            rm_attention_mask[s:e],
                            bp_images[s:e].to(dtype=model_dtype),
                        )
                    )
                rewards = torch.cat(reward_chunks, dim=0)

            loss_bp_raw = F.relu(float(ls_cfg.bp_reward_margin) - rewards).mean()
            metrics = {
                "loss_bp_raw": float(loss_bp_raw.detach().item()),
                "bp_reward_mean": float(rewards.detach().mean().item()),
                "bp_reward_min": float(rewards.detach().min().item()),
                "bp_reward_max": float(rewards.detach().max().item()),
                "bp_applied": 1.0,
                "bp_is_deterministic_step": float(is_deterministic),
            }

        loss_bp = (
            loss_bp_raw
            * float(ls_cfg.bp_grad_scale)
        )
        metrics["loss_bp"] = float(loss_bp.detach().item())
        return loss_bp, metrics

    def _compute_bp_training_weighted_reward(
        self,
        *,
        images: torch.Tensor,
        batch: dict,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute weighted training reward for differentiable BP objective."""
        device = images.device
        self._init_bp_training_reward_models(device=device)
        self._ensure_bp_training_reward_device(device)
        if not self._bp_train_reward_models:
            raise RuntimeError("BP training-reward models are not initialized")

        scores: dict[str, torch.Tensor] = {}
        if "clipscore" in self._bp_train_reward_models:
            scores["clipscore"] = self._compute_bp_reward_clipscore(images, batch)
        if "pickscore" in self._bp_train_reward_models:
            scores["pickscore"] = self._compute_bp_reward_pickscore(images, batch)
        if "hpsv2" in self._bp_train_reward_models:
            scores["hpsv2"] = self._compute_bp_reward_hpsv2(images, batch)

        if not scores:
            raise RuntimeError(
                "No reward scores computed in BP training-reward mode."
            )

        weighted = None
        metrics: dict[str, float] = {}
        for name, score in scores.items():
            w = float(self._bp_train_reward_weights.get(name, 0.0))
            if weighted is None:
                weighted = w * score
            else:
                weighted = weighted + w * score
            metrics[f"bp_reward/{name}_mean"] = float(score.detach().mean().item())
            metrics[f"bp_reward/{name}_min"] = float(score.detach().min().item())
            metrics[f"bp_reward/{name}_max"] = float(score.detach().max().item())

        if weighted is None:
            raise RuntimeError(
                "BP training-reward weighted score is empty."
            )
        return weighted, metrics

    def _compute_bp_reward_clipscore(
        self,
        images: torch.Tensor,
        batch: dict,
    ) -> torch.Tensor:
        scorer = self._bp_train_reward_models["clipscore"]
        device = images.device
        text_inputs = {}
        prefix = "bp_clipscore_"
        for k in ("input_ids", "attention_mask", "token_type_ids"):
            key = f"{prefix}{k}"
            if key in batch:
                text_inputs[k] = batch[key].to(device=device, dtype=torch.long)
        if "input_ids" not in text_inputs:
            raise RuntimeError("Missing bp_clipscore_input_ids in batch")

        model_dtype = next(scorer.model.parameters()).dtype
        chunk_bs = self._get_bp_batch_size(images.shape[0])
        if chunk_bs >= images.shape[0]:
            pixels = scorer._process(images.to(dtype=model_dtype)).to(
                device=device, dtype=model_dtype
            )
            outputs = scorer.model(pixel_values=pixels, **text_inputs)
            return outputs.logits_per_image.diagonal() / 100.0

        score_chunks = []
        for s in range(0, images.shape[0], chunk_bs):
            e = min(s + chunk_bs, images.shape[0])
            pixels = scorer._process(images[s:e].to(dtype=model_dtype)).to(
                device=device, dtype=model_dtype
            )
            chunk_inputs = {k: v[s:e] for k, v in text_inputs.items()}
            outputs = scorer.model(pixel_values=pixels, **chunk_inputs)
            score_chunks.append(outputs.logits_per_image.diagonal() / 100.0)
        return torch.cat(score_chunks, dim=0)

    def _compute_bp_reward_pickscore(
        self,
        images: torch.Tensor,
        batch: dict,
    ) -> torch.Tensor:
        scorer = self._bp_train_reward_models["pickscore"]
        device = images.device
        text_inputs = {}
        prefix = "bp_pickscore_"
        for k in ("input_ids", "attention_mask", "token_type_ids"):
            key = f"{prefix}{k}"
            if key in batch:
                text_inputs[k] = batch[key].to(device=device, dtype=torch.long)
        if "input_ids" not in text_inputs:
            raise RuntimeError("Missing bp_pickscore_input_ids in batch")

        if self._bp_pickscore_image_tform is None:
            from rtdmd.rewards.clip_scorer import get_image_transform
            self._bp_pickscore_image_tform = get_image_transform(
                scorer.processor.image_processor
            )

        model_dtype = next(scorer.model.parameters()).dtype
        chunk_bs = self._get_bp_batch_size(images.shape[0])
        if chunk_bs >= images.shape[0]:
            pixels = self._bp_pickscore_image_tform(images.to(dtype=model_dtype)).to(
                device=device, dtype=model_dtype
            )
            image_embs = scorer.model.get_image_features(pixel_values=pixels)
            image_embs = image_embs / image_embs.norm(p=2, dim=-1, keepdim=True)

            text_embs = scorer.model.get_text_features(**text_inputs)
            text_embs = text_embs / text_embs.norm(p=2, dim=-1, keepdim=True)

            logit_scale = scorer.model.logit_scale.exp()
            scores = logit_scale * (text_embs @ image_embs.T)
            return scores.diag() / 26.0

        score_chunks = []
        for s in range(0, images.shape[0], chunk_bs):
            e = min(s + chunk_bs, images.shape[0])
            pixels = self._bp_pickscore_image_tform(
                images[s:e].to(dtype=model_dtype)
            ).to(device=device, dtype=model_dtype)
            image_embs = scorer.model.get_image_features(pixel_values=pixels)
            image_embs = image_embs / image_embs.norm(p=2, dim=-1, keepdim=True)

            chunk_inputs = {k: v[s:e] for k, v in text_inputs.items()}
            text_embs = scorer.model.get_text_features(**chunk_inputs)
            text_embs = text_embs / text_embs.norm(p=2, dim=-1, keepdim=True)

            logit_scale = scorer.model.logit_scale.exp()
            scores = logit_scale * (text_embs @ image_embs.T)
            score_chunks.append(scores.diag() / 26.0)
        return torch.cat(score_chunks, dim=0)

    def _compute_bp_reward_hpsv2(
        self,
        images: torch.Tensor,
        batch: dict,
    ) -> torch.Tensor:
        scorer = self._bp_train_reward_models["hpsv2"]
        device = images.device
        if "bp_hpsv2_input_ids" not in batch:
            raise RuntimeError("Missing bp_hpsv2_input_ids in batch")
        text = batch["bp_hpsv2_input_ids"].to(device=device, dtype=torch.long)
        chunk_bs = self._get_bp_batch_size(images.shape[0])
        if chunk_bs >= images.shape[0]:
            image = scorer.preprocess_val(
                images.to(scorer.dtype).to(device=device, non_blocking=True)
            )
            outputs = scorer.model(image, text)
            image_features = outputs["image_features"]
            text_features = outputs["text_features"]
            logits_per_image = image_features @ text_features.T
            return torch.diagonal(logits_per_image, 0).contiguous()

        score_chunks = []
        for s in range(0, images.shape[0], chunk_bs):
            e = min(s + chunk_bs, images.shape[0])
            image = scorer.preprocess_val(
                images[s:e].to(scorer.dtype).to(device=device, non_blocking=True)
            )
            text_chunk = text[s:e]
            outputs = scorer.model(image, text_chunk)
            image_features = outputs["image_features"]
            text_features = outputs["text_features"]
            logits_per_image = image_features @ text_features.T
            score_chunks.append(torch.diagonal(logits_per_image, 0).contiguous())
        return torch.cat(score_chunks, dim=0)
