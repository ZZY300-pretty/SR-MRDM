from typing import Dict, Tuple

import torch
import torch.nn as nn

from ...util import append_dims, instantiate_from_config
from .denoiser import Denoiser
from .loss import ResidualDiffusionLoss
from .semantic_feature_matching import SemanticFeatureMatchingLoss
from .sigma2st import Sigma2St


class ImportanceAwareResidualDiffusionLoss(ResidualDiffusionLoss):
    def __init__(
        self,
        mask_key: str = "M",
        importance_weight: float = 4.0,
        background_weight: float = 1.0,
        mask_power: float = 1.0,
        normalize_importance: bool = True,
        eps: float = 1e-6,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.mask_key = mask_key
        self.importance_weight = importance_weight
        self.background_weight = background_weight
        self.mask_power = mask_power
        self.normalize_importance = normalize_importance
        self.eps = eps

    def _build_importance_map(self, batch: Dict, target: torch.Tensor) -> torch.Tensor:
        mask = batch.get(self.mask_key)
        if mask is None:
            return torch.ones_like(target)

        if not isinstance(mask, torch.Tensor):
            mask = torch.as_tensor(mask, device=target.device)
        else:
            mask = mask.to(target.device)

        if mask.ndim == target.ndim - 1:
            mask = mask.unsqueeze(1)
        elif mask.ndim == target.ndim - 2:
            mask = mask.unsqueeze(0).unsqueeze(0)

        mask = mask.float().clamp(0.0, 1.0)
        if mask.shape[1] == 1 and target.shape[1] != 1:
            mask = mask.expand(-1, target.shape[1], -1, -1)

        if self.mask_power != 1.0:
            mask = mask.pow(self.mask_power)

        importance = self.background_weight + mask * (
            self.importance_weight - self.background_weight
        )
        if self.normalize_importance:
            flat = importance.reshape(importance.shape[0], -1)
            mean = flat.mean(dim=1, keepdim=True).clamp_min(self.eps)
            importance = importance / mean.view(-1, 1, 1, 1)
        return importance

    def _compute_weighted_loss(
        self, model_output: torch.Tensor, target: torch.Tensor, base_weight: torch.Tensor, batch: Dict
    ) -> torch.Tensor:
        pixel_weight = base_weight * self._build_importance_map(batch, target)

        if self.loss_type == "l2":
            per_pixel = (model_output - target) ** 2
        elif self.loss_type == "l1":
            per_pixel = (model_output - target).abs()
        elif self.loss_type == "lpips":
            return self.lpips(model_output, target).reshape(-1)
        else:
            raise NotImplementedError(f"Unknown loss type {self.loss_type}")

        weighted_error = (pixel_weight * per_pixel).reshape(target.shape[0], -1).sum(dim=1)
        normalizer = pixel_weight.reshape(target.shape[0], -1).sum(dim=1).clamp_min(self.eps)
        return weighted_error / normalizer

    def _forward(
        self,
        network: nn.Module,
        denoiser: Denoiser,
        cond: Dict,
        sigma2st: Sigma2St,
        input: torch.Tensor,
        mu: torch.Tensor,
        batch: Dict,
    ) -> Tuple[torch.Tensor, Dict]:
        additional_model_inputs = {
            key: batch[key] for key in self.batch2model_keys.intersection(batch)
        }
        sigmas = self.sigma_sampler(input.shape[0]).to(input)
        st = sigma2st(sigmas)
        noise = torch.randn_like(input)
        if self.offset_noise_level > 0.0:
            offset_shape = (
                (input.shape[0], 1, input.shape[2])
                if self.n_frames is not None
                else (input.shape[0], input.shape[1])
            )
            noise = noise + self.offset_noise_level * append_dims(
                torch.randn(offset_shape, device=input.device),
                input.ndim,
            )
        sigmas_bc = append_dims(sigmas, input.ndim)
        st_bc = append_dims(st, input.ndim)
        mu = mu * ((1.0 - st_bc) / st_bc)
        noised_input = self.get_noised_input(sigmas_bc, noise, input, mu)

        model_output = denoiser(
            network, noised_input, sigmas, cond, st, **additional_model_inputs
        )
        base_weight = append_dims(self.loss_weighting(sigmas, st), input.ndim)
        return self._compute_weighted_loss(model_output, input, base_weight, batch)


class SemanticFeatureMatchingResidualDiffusionLoss(
    ImportanceAwareResidualDiffusionLoss
):
    def __init__(
        self,
        feature_matching_weight: float = 0.0,
        feature_matching_config: Dict = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.feature_matching_weight = feature_matching_weight
        if feature_matching_config is None:
            self.feature_matching = None
        elif "target" in feature_matching_config:
            self.feature_matching = instantiate_from_config(feature_matching_config)
        else:
            self.feature_matching = SemanticFeatureMatchingLoss(
                **feature_matching_config
            )

    def _forward(
        self,
        network: nn.Module,
        denoiser: Denoiser,
        cond: Dict,
        sigma2st: Sigma2St,
        input: torch.Tensor,
        mu: torch.Tensor,
        batch: Dict,
    ) -> Tuple[torch.Tensor, Dict]:
        additional_model_inputs = {
            key: batch[key] for key in self.batch2model_keys.intersection(batch)
        }
        sigmas = self.sigma_sampler(input.shape[0]).to(input)
        st = sigma2st(sigmas)
        noise = torch.randn_like(input)
        if self.offset_noise_level > 0.0:
            offset_shape = (
                (input.shape[0], 1, input.shape[2])
                if self.n_frames is not None
                else (input.shape[0], input.shape[1])
            )
            noise = noise + self.offset_noise_level * append_dims(
                torch.randn(offset_shape, device=input.device),
                input.ndim,
            )
        sigmas_bc = append_dims(sigmas, input.ndim)
        st_bc = append_dims(st, input.ndim)
        mu = mu * ((1.0 - st_bc) / st_bc)
        noised_input = self.get_noised_input(sigmas_bc, noise, input, mu)

        model_output = denoiser(
            network, noised_input, sigmas, cond, st, **additional_model_inputs
        )
        base_weight = append_dims(self.loss_weighting(sigmas, st), input.ndim)
        pixel_loss = self._compute_weighted_loss(model_output, input, base_weight, batch)

        log_dict = {
            "loss/pixel_reconstruction": pixel_loss.mean().detach(),
        }

        if self.feature_matching is None or self.feature_matching_weight <= 0.0:
            log_dict["loss/feature_matching"] = pixel_loss.new_tensor(0.0)
            log_dict["loss/total"] = pixel_loss.mean().detach()
            return pixel_loss, log_dict

        feature_loss, feature_logs = self.feature_matching(model_output, input, batch)
        total_loss = pixel_loss + self.feature_matching_weight * feature_loss

        log_dict["loss/feature_matching"] = feature_loss.mean().detach()
        log_dict["loss/feature_matching_weighted"] = (
            self.feature_matching_weight * feature_loss.mean().detach()
        )
        log_dict["loss/total"] = total_loss.mean().detach()
        log_dict.update(feature_logs)
        return total_loss, log_dict
