from typing import Dict, List, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...modules.autoencoding.lpips.loss.lpips import (
    ScalingLayer,
    normalize_tensor,
    vgg16,
)


DEFAULT_LAYER_WEIGHTS = {
    "relu1_2": 0.10,
    "relu2_2": 0.20,
    "relu3_3": 0.30,
    "relu4_3": 0.25,
    "relu5_3": 0.15,
}


class VGGFeatureExtractor(nn.Module):
    def __init__(
        self,
        layers: Optional[Sequence[str]] = None,
        pretrained: bool = True,
        use_input_scaling: bool = True,
        normalize_features: bool = True,
    ):
        super().__init__()
        self.layers = list(layers or ["relu2_2", "relu3_3", "relu4_3"])
        invalid_layers = sorted(set(self.layers) - set(DEFAULT_LAYER_WEIGHTS))
        if invalid_layers:
            raise ValueError(f"Unsupported VGG layers: {invalid_layers}")

        self.normalize_features = normalize_features
        self.scaling_layer = ScalingLayer() if use_input_scaling else nn.Identity()
        self.backbone = vgg16(pretrained=pretrained, requires_grad=False).eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        if image.shape[1] < 3:
            image = image.repeat(1, 3, 1, 1)
        elif image.shape[1] > 3:
            image = image[:, :3, ...]

        outputs = self.backbone(self.scaling_layer(image))
        features = {}
        for layer_name in self.layers:
            feature = getattr(outputs, layer_name)
            if self.normalize_features:
                feature = normalize_tensor(feature)
            features[layer_name] = feature
        return features


class SemanticFeatureMatchingLoss(nn.Module):
    def __init__(
        self,
        layers: Optional[Sequence[str]] = None,
        layer_weights: Optional[Union[Sequence[float], Dict[str, float]]] = None,
        feature_distance: str = "l1",
        mask_mode: str = "intersection",
        semantic_mask_key: str = "semantic_mask",
        semantic_mask_available_key: Optional[str] = "semantic_mask_available",
        cloud_mask_key: str = "M",
        missing_semantic_fallback: str = "ones",
        missing_cloud_fallback: str = "ones",
        outside_mask_weight: float = 0.0,
        normalize_by_mask_area: bool = True,
        eps: float = 1e-6,
        pretrained_backbone: bool = True,
        use_input_scaling: bool = True,
        normalize_features: bool = True,
    ):
        super().__init__()
        self.layers = list(layers or ["relu2_2", "relu3_3", "relu4_3"])
        self.layer_weights = self._build_layer_weights(layer_weights)
        self.feature_distance = feature_distance
        self.mask_mode = mask_mode
        self.semantic_mask_key = semantic_mask_key
        self.semantic_mask_available_key = semantic_mask_available_key
        self.cloud_mask_key = cloud_mask_key
        self.missing_semantic_fallback = missing_semantic_fallback
        self.missing_cloud_fallback = missing_cloud_fallback
        self.outside_mask_weight = outside_mask_weight
        self.normalize_by_mask_area = normalize_by_mask_area
        self.eps = eps

        if self.feature_distance not in {"l1", "l2"}:
            raise ValueError(f"Unsupported feature distance: {self.feature_distance}")
        if self.mask_mode not in {"none", "semantic", "cloud", "intersection", "union"}:
            raise ValueError(f"Unsupported mask mode: {self.mask_mode}")

        self.feature_extractor = VGGFeatureExtractor(
            layers=self.layers,
            pretrained=pretrained_backbone,
            use_input_scaling=use_input_scaling,
            normalize_features=normalize_features,
        )

    def _build_layer_weights(
        self, layer_weights: Optional[Union[Sequence[float], Dict[str, float]]]
    ) -> Dict[str, float]:
        if layer_weights is None:
            return {
                layer_name: DEFAULT_LAYER_WEIGHTS.get(layer_name, 1.0)
                for layer_name in self.layers
            }

        if isinstance(layer_weights, dict):
            return {
                layer_name: float(layer_weights.get(layer_name, 1.0))
                for layer_name in self.layers
            }

        layer_weights = list(layer_weights)
        if len(layer_weights) != len(self.layers):
            raise ValueError("layer_weights must match the number of selected layers.")
        return {
            layer_name: float(weight)
            for layer_name, weight in zip(self.layers, layer_weights)
        }

    def _to_single_channel_mask(
        self, mask: Optional[torch.Tensor], reference: torch.Tensor
    ) -> Optional[torch.Tensor]:
        if mask is None:
            return None

        if not isinstance(mask, torch.Tensor):
            mask = torch.as_tensor(mask, device=reference.device)
        else:
            mask = mask.to(reference.device)

        if mask.ndim == reference.ndim - 1:
            mask = mask.unsqueeze(1)
        elif mask.ndim == reference.ndim - 2:
            mask = mask.unsqueeze(0).unsqueeze(0)

        mask = mask.float()
        if mask.ndim != 4:
            raise ValueError(f"Expected a 4D mask tensor, got shape {tuple(mask.shape)}")
        if mask.shape[1] > 1:
            mask = mask.amax(dim=1, keepdim=True)
        return mask.clamp(0.0, 1.0)

    def _is_mask_available(self, batch: Dict, key: Optional[str], reference: torch.Tensor) -> bool:
        if key is None or key not in batch:
            return True
        availability = batch.get(key)
        if availability is None:
            return True
        if not isinstance(availability, torch.Tensor):
            availability = torch.as_tensor(availability, device=reference.device)
        else:
            availability = availability.to(reference.device)
        return bool((availability > 0).any().item())

    def _fallback_mask(
        self,
        mask: Optional[torch.Tensor],
        other_mask: Optional[torch.Tensor],
        fallback_mode: str,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        if mask is not None:
            return mask
        if fallback_mode == "zeros":
            return torch.zeros(
                reference.shape[0], 1, reference.shape[2], reference.shape[3],
                device=reference.device, dtype=reference.dtype
            )
        if fallback_mode == "other" and other_mask is not None:
            return other_mask
        return torch.ones(
            reference.shape[0], 1, reference.shape[2], reference.shape[3],
            device=reference.device, dtype=reference.dtype
        )

    def _build_region_mask(self, batch: Dict, target: torch.Tensor) -> torch.Tensor:
        semantic_mask = None
        if self._is_mask_available(batch, self.semantic_mask_available_key, target):
            semantic_mask = self._to_single_channel_mask(
                batch.get(self.semantic_mask_key), target
            )
        cloud_mask = self._to_single_channel_mask(batch.get(self.cloud_mask_key), target)
        if self.mask_mode == "none":
            region_mask = torch.ones(
                target.shape[0], 1, target.shape[2], target.shape[3],
                device=target.device, dtype=target.dtype
            )
        elif self.mask_mode == "semantic":
            region_mask = self._fallback_mask(
                semantic_mask, cloud_mask, self.missing_semantic_fallback, target
            )
        elif self.mask_mode == "cloud":
            region_mask = self._fallback_mask(
                cloud_mask, semantic_mask, self.missing_cloud_fallback, target
            )
        else:
            semantic_mask = self._fallback_mask(
                semantic_mask, cloud_mask, self.missing_semantic_fallback, target
            )
            cloud_mask = self._fallback_mask(
                cloud_mask, semantic_mask, self.missing_cloud_fallback, target
            )
            if self.mask_mode == "intersection":
                region_mask = semantic_mask * cloud_mask
            else:
                region_mask = torch.clamp(semantic_mask + cloud_mask, 0.0, 1.0)

        if self.outside_mask_weight > 0.0:
            region_mask = self.outside_mask_weight + (
                1.0 - self.outside_mask_weight
            ) * region_mask
        return region_mask

    def _masked_feature_distance(
        self,
        prediction_feature: torch.Tensor,
        target_feature: torch.Tensor,
        region_mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.feature_distance == "l1":
            diff = (prediction_feature - target_feature).abs()
        else:
            diff = (prediction_feature - target_feature) ** 2

        mask = F.interpolate(
            region_mask,
            size=prediction_feature.shape[-2:],
            mode="area",
        ).to(diff.dtype)
        mask = mask.expand(-1, diff.shape[1], -1, -1)
        numerator = (mask * diff).reshape(diff.shape[0], -1).sum(dim=1)
        if self.normalize_by_mask_area:
            denominator = mask.reshape(mask.shape[0], -1).sum(dim=1).clamp_min(self.eps)
        else:
            denominator = diff.new_full((diff.shape[0],), diff[0].numel())
        return numerator / denominator

    def forward(
        self, prediction: torch.Tensor, target: torch.Tensor, batch: Dict
    ) -> (torch.Tensor, Dict[str, torch.Tensor]):
        with torch.no_grad():
            target_features = self.feature_extractor(target)
        prediction_features = self.feature_extractor(prediction)
        region_mask = self._build_region_mask(batch, target)

        total_loss = prediction.new_zeros(prediction.shape[0])
        log_dict = {}
        for layer_name in self.layers:
            layer_loss = self._masked_feature_distance(
                prediction_features[layer_name],
                target_features[layer_name],
                region_mask,
            )
            weighted_layer_loss = self.layer_weights[layer_name] * layer_loss
            total_loss = total_loss + weighted_layer_loss
            log_dict[f"loss/feature_matching_{layer_name}"] = layer_loss.mean().detach()

        log_dict["loss/feature_matching_region_mean"] = region_mask.mean().detach()
        return total_loss, log_dict


class MaskWeightedAsymmetricFeatureMatchingLoss(SemanticFeatureMatchingLoss):
    def __init__(
        self,
        foreground_weight: float = 1.0,
        background_weight: float = 0.1,
        cloud_foreground_boost: float = 1.0,
        cloud_background_boost: float = 0.0,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.foreground_weight = foreground_weight
        self.background_weight = background_weight
        self.cloud_foreground_boost = cloud_foreground_boost
        self.cloud_background_boost = cloud_background_boost

    def _build_region_mask(self, batch: Dict, target: torch.Tensor) -> torch.Tensor:
        semantic_mask = None
        if self._is_mask_available(batch, self.semantic_mask_available_key, target):
            semantic_mask = self._to_single_channel_mask(
                batch.get(self.semantic_mask_key), target
            )
        cloud_mask = self._to_single_channel_mask(batch.get(self.cloud_mask_key), target)

        if semantic_mask is None:
            semantic_mask = torch.ones(
                target.shape[0], 1, target.shape[2], target.shape[3],
                device=target.device, dtype=target.dtype
            )

        if cloud_mask is None:
            cloud_mask = torch.zeros_like(semantic_mask)

        foreground = semantic_mask
        background = 1.0 - foreground
        weight = (
            foreground * self.foreground_weight
            + background * self.background_weight
        )
        weight = weight + cloud_mask * (
            foreground * self.cloud_foreground_boost
            + background * self.cloud_background_boost
        )
        return weight
