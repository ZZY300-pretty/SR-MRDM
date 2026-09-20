import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...util import instantiate_from_config


def _zero_init_conv(conv: nn.Conv2d) -> nn.Conv2d:
    nn.init.zeros_(conv.weight)
    if conv.bias is not None:
        nn.init.zeros_(conv.bias)
    return conv


class ConvFeatureEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        num_layers: int,
        add_global_context: bool = False,
    ):
        super().__init__()
        assert num_layers >= 2, "num_layers must be at least 2."
        layers = [
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
        ]
        for _ in range(num_layers - 2):
            layers.extend(
                [
                    nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
                    nn.SiLU(inplace=True),
                ]
            )
        layers.append(nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1))
        self.net = nn.Sequential(*layers)
        self.add_global_context = add_global_context
        if add_global_context:
            self.global_proj = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=1),
                nn.SiLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=1),
            )
        else:
            self.global_proj = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.net(x)
        if self.global_proj is not None:
            pooled = F.adaptive_avg_pool2d(feat, output_size=1)
            feat = feat + self.global_proj(pooled)
        return feat


class LightweightBackgroundDenoiser(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int = 32,
        num_layers: int = 2,
    ):
        super().__init__()
        assert num_layers >= 2, "num_layers must be at least 2."
        layers = [
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
        ]
        for _ in range(num_layers - 2):
            layers.extend(
                [
                    nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
                    nn.SiLU(inplace=True),
                ]
            )
        layers.append(nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1))
        self.net = nn.Sequential(*layers)
        self.skip_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = x[:, : self.net[-1].out_channels, ...]
        return base * self.skip_scale.to(base.dtype) + self.net(x)


class GlobalBackgroundCompensator(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int = 48,
        num_layers: int = 4,
    ):
        super().__init__()
        assert num_layers >= 2, "num_layers must be at least 2."
        layers = [
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
        ]
        for _ in range(num_layers - 2):
            layers.extend(
                [
                    nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
                    nn.SiLU(inplace=True),
                ]
            )
        layers.append(nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1))
        self.net = nn.Sequential(*layers)
        self.skip_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = x[:, : self.net[-1].out_channels, ...]
        return base * self.skip_scale.to(base.dtype) + self.net(x)


class BackgroundFeatureFusion(nn.Module):
    def __init__(
        self,
        patch_in_channels: int,
        background_feature_channels: int,
        fusion_feature_channels: int = 64,
        local_feature_hidden_channels: int = 64,
        local_feature_num_layers: int = 2,
        fusion_mode: str = "film",
    ):
        super().__init__()
        if fusion_mode not in {"film", "gated", "concat"}:
            raise ValueError(
                "fusion_mode must be one of {'film', 'gated', 'concat'}."
            )
        self.fusion_mode = fusion_mode
        self.local_encoder = ConvFeatureEncoder(
            in_channels=patch_in_channels,
            out_channels=fusion_feature_channels,
            hidden_channels=local_feature_hidden_channels,
            num_layers=local_feature_num_layers,
        )
        self.background_proj = nn.Conv2d(
            background_feature_channels, fusion_feature_channels, kernel_size=1
        )

        if fusion_mode == "film":
            self.film_generator = nn.Sequential(
                nn.Conv2d(fusion_feature_channels, fusion_feature_channels, kernel_size=1),
                nn.SiLU(inplace=True),
                nn.Conv2d(fusion_feature_channels, fusion_feature_channels * 2, kernel_size=1),
            )
            self.gate_generator = None
            self.concat_fuser = None
        elif fusion_mode == "gated":
            self.gate_generator = nn.Sequential(
                nn.Conv2d(fusion_feature_channels * 2, fusion_feature_channels, kernel_size=1),
                nn.SiLU(inplace=True),
                nn.Conv2d(fusion_feature_channels, fusion_feature_channels, kernel_size=1),
            )
            self.film_generator = None
            self.concat_fuser = None
        else:
            self.concat_fuser = nn.Sequential(
                nn.Conv2d(fusion_feature_channels * 2, fusion_feature_channels, kernel_size=1),
                nn.SiLU(inplace=True),
                nn.Conv2d(fusion_feature_channels, fusion_feature_channels, kernel_size=3, padding=1),
            )
            self.film_generator = None
            self.gate_generator = None

        self.output_proj = _zero_init_conv(
            nn.Conv2d(fusion_feature_channels, patch_in_channels, kernel_size=1)
        )
        self.residual_scale = nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        patch_input: torch.Tensor,
        background_features: torch.Tensor,
    ) -> torch.Tensor:
        local_feat = self.local_encoder(patch_input)
        bg_feat = self.background_proj(background_features.to(local_feat.dtype))

        if self.fusion_mode == "film":
            gamma, beta = self.film_generator(bg_feat).chunk(2, dim=1)
            fused = local_feat * (1.0 + torch.tanh(gamma)) + beta
        elif self.fusion_mode == "gated":
            gate = torch.sigmoid(self.gate_generator(torch.cat([local_feat, bg_feat], dim=1)))
            fused = gate * local_feat + (1.0 - gate) * bg_feat
        else:
            fused = self.concat_fuser(torch.cat([local_feat, bg_feat], dim=1))

        delta = self.output_proj(fused)
        return patch_input + self.residual_scale.to(delta.dtype) * delta


class SemanticRoutedPatchDenoiser(nn.Module):
    def __init__(
        self,
        backbone_config: Dict,
        in_channels: int,
        out_channels: int,
        route_patch_size: int = 128,
        route_threshold: float = 0.01,
        route_mode: str = "mean",
        semantic_mask_key: str = "semantic_mask",
        semantic_mask_available_key: str = "semantic_mask_available",
        cloud_mask_key: str = "M",
        route_with_cloud_mask: bool = True,
        route_mask_combine_mode: str = "union",
        light_branch_hidden_channels: int = 32,
        light_branch_num_layers: int = 2,
        use_shared_background: bool = True,
        use_independent_background_branch: bool = False,
        background_branch_in_channels: Optional[int] = None,
        background_branch_hidden_channels: int = 48,
        background_branch_num_layers: int = 4,
        background_input_mode: str = "cond_only",
        use_background_feature_fusion: bool = False,
        background_feature_channels: int = 64,
        background_feature_hidden_channels: int = 64,
        background_feature_num_layers: int = 4,
        heavy_feature_fusion_mode: str = "film",
        heavy_feature_hidden_channels: int = 64,
        heavy_feature_num_layers: int = 2,
        use_transformer_background_context: bool = False,
        transformer_background_context_channels: int = 64,
        transformer_background_context_hidden_channels: int = 64,
        transformer_background_context_num_layers: int = 4,
        merge_stride: Optional[int] = None,
        heavy_patch_blend_margin: int = 16,
        force_all_patches_to_heavy: bool = False,
        fallback_to_backbone_without_mask: bool = True,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.backbone = instantiate_from_config(backbone_config)
        self.light_branch = LightweightBackgroundDenoiser(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=light_branch_hidden_channels,
            num_layers=light_branch_num_layers,
        )
        self.use_independent_background_branch = use_independent_background_branch
        self.background_input_mode = background_input_mode
        self.background_branch_in_channels = background_branch_in_channels or out_channels
        self.use_background_feature_fusion = use_background_feature_fusion
        self.use_transformer_background_context = use_transformer_background_context
        if self.use_independent_background_branch:
            self.background_branch = GlobalBackgroundCompensator(
                in_channels=self.background_branch_in_channels,
                out_channels=out_channels,
                hidden_channels=background_branch_hidden_channels,
                num_layers=background_branch_num_layers,
            )
        else:
            self.background_branch = None
        if self.use_background_feature_fusion:
            self.background_feature_encoder = ConvFeatureEncoder(
                in_channels=self.background_branch_in_channels,
                out_channels=background_feature_channels,
                hidden_channels=background_feature_hidden_channels,
                num_layers=background_feature_num_layers,
                add_global_context=True,
            )
            self.heavy_feature_fusion = BackgroundFeatureFusion(
                patch_in_channels=in_channels,
                background_feature_channels=background_feature_channels,
                fusion_feature_channels=background_feature_channels,
                local_feature_hidden_channels=heavy_feature_hidden_channels,
                local_feature_num_layers=heavy_feature_num_layers,
                fusion_mode=heavy_feature_fusion_mode,
            )
        else:
            self.background_feature_encoder = None
            self.heavy_feature_fusion = None
        if self.use_transformer_background_context:
            backbone_context_channels = getattr(
                self.backbone, "background_context_in_channels", None
            )
            self.transformer_background_context_channels = (
                backbone_context_channels
                if backbone_context_channels is not None
                else transformer_background_context_channels
            )
            self.transformer_background_context_encoder = ConvFeatureEncoder(
                in_channels=self.background_branch_in_channels,
                out_channels=self.transformer_background_context_channels,
                hidden_channels=transformer_background_context_hidden_channels,
                num_layers=transformer_background_context_num_layers,
                add_global_context=True,
            )
        else:
            self.transformer_background_context_channels = None
            self.transformer_background_context_encoder = None
        self.route_patch_size = route_patch_size
        self.route_threshold = route_threshold
        self.route_mode = route_mode
        self.semantic_mask_key = semantic_mask_key
        self.semantic_mask_available_key = semantic_mask_available_key
        self.cloud_mask_key = cloud_mask_key
        self.route_with_cloud_mask = route_with_cloud_mask
        self.route_mask_combine_mode = route_mask_combine_mode
        self.use_shared_background = use_shared_background
        self.merge_stride = merge_stride or max(route_patch_size // 2, 1)
        self.heavy_patch_blend_margin = heavy_patch_blend_margin
        self.force_all_patches_to_heavy = force_all_patches_to_heavy
        self.fallback_to_backbone_without_mask = fallback_to_backbone_without_mask
        if self.route_mode not in {"mean", "max"}:
            raise ValueError(f"Unsupported route_mode: {self.route_mode}")
        if self.route_mask_combine_mode not in {"union", "intersection", "cloud_only"}:
            raise ValueError(
                f"Unsupported route_mask_combine_mode: {self.route_mask_combine_mode}"
            )
        if self.heavy_patch_blend_margin < 0:
            raise ValueError("heavy_patch_blend_margin must be non-negative.")
        if self.merge_stride <= 0 or self.merge_stride > self.route_patch_size:
            raise ValueError("merge_stride must be in the range [1, route_patch_size].")
        if self.background_input_mode not in {"cond_only", "concat_input"}:
            raise ValueError(
                "background_input_mode must be one of {'cond_only', 'concat_input'}."
            )
        if self.use_transformer_background_context and (
            self.use_background_feature_fusion
            or self.use_independent_background_branch
            or self.use_shared_background
        ):
            raise ValueError(
                "use_transformer_background_context should be the only active "
                "background path."
            )
        if (
            self.use_background_feature_fusion
            and self.use_independent_background_branch
            and self.use_shared_background
        ):
            raise ValueError(
                "Feature fusion should not be enabled together with both image-level "
                "background branches. Disable use_shared_background or "
                "use_independent_background_branch when using feature fusion."
            )
        self._last_debug_state: Dict[str, torch.Tensor] = {}

    def _extract_background_input(self, x: torch.Tensor) -> torch.Tensor:
        if self.background_input_mode == "concat_input":
            return x[:, : self.background_branch_in_channels, ...]
        if x.shape[1] < self.background_branch_in_channels:
            raise ValueError(
                "Input channels are fewer than background_branch_in_channels."
            )
        return x[:, -self.background_branch_in_channels :, ...]

    def _compute_transformer_background_context(
        self, x: torch.Tensor
    ) -> Optional[torch.Tensor]:
        if not self.use_transformer_background_context:
            return None
        background_input = self._extract_background_input(x)
        return self.transformer_background_context_encoder(background_input)

    def _mask_available(
        self, semantic_mask: Optional[torch.Tensor], semantic_mask_available: Optional[torch.Tensor]
    ) -> bool:
        if semantic_mask is None:
            return False
        if semantic_mask_available is None:
            return True
        if not isinstance(semantic_mask_available, torch.Tensor):
            semantic_mask_available = torch.as_tensor(
                semantic_mask_available, device=semantic_mask.device
            )
        else:
            semantic_mask_available = semantic_mask_available.to(semantic_mask.device)
        return bool((semantic_mask_available > 0).any().item())

    def _prepare_mask(self, semantic_mask: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if semantic_mask.ndim == 3:
            semantic_mask = semantic_mask.unsqueeze(1)
        elif semantic_mask.ndim == 2:
            semantic_mask = semantic_mask.unsqueeze(0).unsqueeze(0)
        semantic_mask = semantic_mask.to(device=x.device, dtype=x.dtype)
        if semantic_mask.shape[1] > 1:
            semantic_mask = semantic_mask.amax(dim=1, keepdim=True)
        return semantic_mask.clamp(0.0, 1.0)

    def _resolve_cloud_mask(
        self, cloud_mask: Optional[torch.Tensor], kwargs: Dict
    ) -> Optional[torch.Tensor]:
        if cloud_mask is not None:
            return cloud_mask
        return kwargs.get(self.cloud_mask_key)

    def _combine_route_masks(
        self,
        semantic_mask: Optional[torch.Tensor],
        cloud_mask: Optional[torch.Tensor],
        x: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        prepared_semantic = None
        prepared_cloud = None

        if semantic_mask is not None:
            prepared_semantic = self._prepare_mask(semantic_mask, x)
        if cloud_mask is not None:
            prepared_cloud = self._prepare_mask(cloud_mask, x)

        if prepared_semantic is None:
            return prepared_cloud
        if prepared_cloud is None or not self.route_with_cloud_mask:
            return prepared_semantic

        if self.route_mask_combine_mode == "intersection":
            return prepared_semantic * prepared_cloud
        if self.route_mask_combine_mode == "cloud_only":
            return prepared_cloud
        return torch.clamp(prepared_semantic + prepared_cloud, 0.0, 1.0)

    def _pad_to_multiple(
        self, tensor: torch.Tensor, multiple: int
    ) -> Tuple[torch.Tensor, Tuple[int, int]]:
        h, w = tensor.shape[-2:]
        pad_h = (multiple - h % multiple) % multiple
        pad_w = (multiple - w % multiple) % multiple
        if pad_h == 0 and pad_w == 0:
            return tensor, (0, 0)
        return F.pad(tensor, (0, pad_w, 0, pad_h)), (pad_h, pad_w)

    def _pad_for_patch_grid(
        self, tensor: torch.Tensor, patch_size: int, stride: int
    ) -> Tuple[torch.Tensor, Tuple[int, int]]:
        h, w = tensor.shape[-2:]
        target_h = patch_size if h <= patch_size else math.ceil((h - patch_size) / stride) * stride + patch_size
        target_w = patch_size if w <= patch_size else math.ceil((w - patch_size) / stride) * stride + patch_size
        pad_h = target_h - h
        pad_w = target_w - w
        if pad_h == 0 and pad_w == 0:
            return tensor, (0, 0)
        return F.pad(tensor, (0, pad_w, 0, pad_h)), (pad_h, pad_w)

    def _patchify(self, tensor: torch.Tensor, stride: Optional[int] = None) -> torch.Tensor:
        p = self.route_patch_size
        stride = p if stride is None else stride
        unfolded = F.unfold(tensor, kernel_size=p, stride=stride)
        b, cp2, num_patches = unfolded.shape
        c = tensor.shape[1]
        return unfolded.transpose(1, 2).reshape(b, num_patches, c, p, p)

    def _unpatchify(
        self,
        patches: torch.Tensor,
        batch_size: int,
        out_h: int,
        out_w: int,
        stride: Optional[int] = None,
    ) -> torch.Tensor:
        p = self.route_patch_size
        stride = p if stride is None else stride
        unfolded = patches.reshape(batch_size, -1, patches.shape[2] * p * p).transpose(1, 2)
        folded = F.fold(unfolded, output_size=(out_h, out_w), kernel_size=p, stride=stride)
        if stride == p:
            return folded

        # `patches` have already been fused with the heavy/background blend mask.
        # During overlap merge we only need a stable average over coverage counts.
        # Reusing the heavy blend window here would make the denominator approach
        # zero at image/patch borders and can catastrophically amplify outputs.
        norm_patches = torch.ones(
            (batch_size, patches.shape[1], 1, p, p),
            dtype=folded.dtype,
            device=folded.device,
        )
        norm_unfold = norm_patches.reshape(batch_size, -1, p * p).transpose(1, 2)
        norm = F.fold(norm_unfold, output_size=(out_h, out_w), kernel_size=p, stride=stride)
        return folded / norm.clamp_min(1e-6)

    def _get_heavy_patch_blend_mask(
        self, dtype: torch.dtype, device: torch.device
    ) -> torch.Tensor:
        p = self.route_patch_size
        margin = min(self.heavy_patch_blend_margin, max(p // 2, 0))
        if margin <= 0:
            return torch.ones((1, 1, p, p), dtype=dtype, device=device)

        coords = torch.arange(p, dtype=dtype, device=device)
        edge_distance = torch.minimum(coords, coords.flip(0))
        alpha_1d = torch.clamp(edge_distance / float(margin), 0.0, 1.0)
        alpha_1d = alpha_1d * alpha_1d * (3.0 - 2.0 * alpha_1d)
        alpha_2d = alpha_1d[:, None] * alpha_1d[None, :]
        return alpha_2d.unsqueeze(0).unsqueeze(0)

    def _build_route_mask(
        self, semantic_mask: torch.Tensor, stride: Optional[int] = None
    ) -> torch.Tensor:
        patches = self._patchify(semantic_mask, stride=stride)
        if self.route_mode == "mean":
            route_score = patches.mean(dim=(2, 3, 4))
        else:
            route_score = patches.amax(dim=(2, 3, 4))
        return route_score > self.route_threshold

    def _update_debug_state(self, device: torch.device, **stats) -> None:
        debug_state: Dict[str, torch.Tensor] = {}
        for key, value in stats.items():
            if value is None:
                continue
            if isinstance(value, torch.Tensor):
                tensor = value.detach()
                if tensor.numel() != 1:
                    tensor = tensor.float().mean()
                else:
                    tensor = tensor.float().reshape(())
                debug_state[key] = tensor.to(device)
            else:
                debug_state[key] = torch.tensor(float(value), device=device)
        self._last_debug_state = debug_state

    def get_debug_state(self) -> Dict[str, torch.Tensor]:
        return dict(self._last_debug_state)

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        semantic_mask: Optional[torch.Tensor] = None,
        semantic_mask_available: Optional[torch.Tensor] = None,
        cloud_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        cloud_mask = self._resolve_cloud_mask(cloud_mask, kwargs)
        if not self._mask_available(semantic_mask, semantic_mask_available):
            if cloud_mask is None and self.fallback_to_backbone_without_mask:
                out = self.backbone(
                    x,
                    timesteps,
                    background_context=self._compute_transformer_background_context(x),
                )
                self._update_debug_state(
                    x.device,
                    fallback_without_route=1.0,
                    route_ratio=1.0,
                    output_abs_mean=out.abs().mean(),
                    output_zero_fraction=(out.abs() < 1e-6).float().mean(),
                )
                return out
            semantic_mask = None

        route_mask = self._combine_route_masks(semantic_mask, cloud_mask, x)
        if route_mask is None:
            if self.fallback_to_backbone_without_mask:
                out = self.backbone(
                    x,
                    timesteps,
                    background_context=self._compute_transformer_background_context(x),
                )
                self._update_debug_state(
                    x.device,
                    fallback_without_route=1.0,
                    route_ratio=1.0,
                    output_abs_mean=out.abs().mean(),
                    output_zero_fraction=(out.abs() < 1e-6).float().mean(),
                )
                return out
            route_mask = torch.ones(
                x.shape[0], 1, x.shape[2], x.shape[3], device=x.device, dtype=x.dtype
            )

        x_padded, (pad_h, pad_w) = self._pad_for_patch_grid(
            x, self.route_patch_size, self.merge_stride
        )
        route_mask_padded, _ = self._pad_for_patch_grid(
            route_mask, self.route_patch_size, self.merge_stride
        )

        route_mask = self._build_route_mask(route_mask_padded, stride=self.merge_stride)
        if self.force_all_patches_to_heavy:
            route_mask = torch.ones_like(route_mask, dtype=torch.bool)
        x_patches = self._patchify(x_padded, stride=self.merge_stride)
        batch_size, num_patches = x_patches.shape[:2]
        flat_patches = x_patches.reshape(-1, x_patches.shape[2], x_patches.shape[3], x_patches.shape[4])
        flat_route = route_mask.reshape(-1)
        patch_timesteps = timesteps[:, None].expand(batch_size, num_patches).reshape(-1)
        route_ratio = flat_route.float().mean()
        transformer_background_context_patches = None
        transformer_background_context = None
        if self.use_transformer_background_context:
            transformer_background_context = self._compute_transformer_background_context(
                x_padded
            )
            transformer_background_context_patches = self._patchify(
                transformer_background_context, stride=self.merge_stride
            ).reshape(
                -1,
                transformer_background_context.shape[1],
                self.route_patch_size,
                self.route_patch_size,
            )

        background_feature_patches = None
        if self.use_background_feature_fusion:
            background_input = self._extract_background_input(x_padded)
            background_features = self.background_feature_encoder(background_input)
            background_feature_patches = self._patchify(
                background_features, stride=self.merge_stride
            ).reshape(
                -1,
                background_features.shape[1],
                self.route_patch_size,
                self.route_patch_size,
            )

        # Metric-oriented finetuning may want to keep the routed patch pipeline
        # and overlap reconstruction while removing the lightweight branch from
        # the final restoration path entirely.
        if bool(flat_route.all().item()):
            heavy_input = flat_patches
            if background_feature_patches is not None:
                heavy_input = self.heavy_feature_fusion(
                    heavy_input,
                    background_feature_patches.to(heavy_input.dtype),
                )
            heavy_kwargs = {}
            if transformer_background_context_patches is not None:
                heavy_kwargs["background_context"] = transformer_background_context_patches.to(
                    heavy_input.dtype
                )
            heavy_out = self.backbone(heavy_input, patch_timesteps, **heavy_kwargs)
            out_patches = heavy_out.view(
                batch_size,
                num_patches,
                heavy_out.shape[1],
                self.route_patch_size,
                self.route_patch_size,
            )
            out = self._unpatchify(
                out_patches,
                batch_size=batch_size,
                out_h=x_padded.shape[-2],
                out_w=x_padded.shape[-1],
                stride=self.merge_stride,
            )
            if pad_h > 0:
                out = out[:, :, :-pad_h, :]
            if pad_w > 0:
                out = out[:, :, :, :-pad_w]
            self._update_debug_state(
                x.device,
                route_ratio=route_ratio,
                all_heavy=1.0,
                heavy_patch_count=float(flat_route.sum().item()),
                total_patch_count=float(flat_route.numel()),
                transformer_context_abs_mean=(
                    transformer_background_context.abs().mean()
                    if transformer_background_context is not None
                    else None
                ),
                heavy_input_abs_mean=heavy_input.abs().mean(),
                heavy_output_abs_mean=heavy_out.abs().mean(),
                heavy_output_zero_fraction=(heavy_out.abs() < 1e-6).float().mean(),
                output_abs_mean=out.abs().mean(),
                output_zero_fraction=(out.abs() < 1e-6).float().mean(),
            )
            return out

        local_light = self.light_branch(flat_patches)

        independent_background = None
        if self.background_branch is not None and not self.use_background_feature_fusion:
            background_input = self._extract_background_input(x_padded)
            independent_background = self.background_branch(background_input)

        shared_background = None
        if (
            self.use_shared_background
            and independent_background is None
            and not self.use_background_feature_fusion
        ):
            shared_background = self.light_branch(x_padded)

        if independent_background is not None:
            base_background = independent_background
        else:
            base_background = shared_background

        if base_background is not None:
            base_patches = self._patchify(base_background, stride=self.merge_stride).reshape(
                -1,
                base_background.shape[1],
                self.route_patch_size,
                self.route_patch_size,
            )
            out_patches = local_light.to(base_patches.dtype)
        else:
            base_patches = None
            out_patches = local_light

        if flat_route.any():
            heavy_indices = flat_route.nonzero(as_tuple=False).squeeze(1)
            heavy_input = flat_patches[heavy_indices]
            if background_feature_patches is not None:
                heavy_input = self.heavy_feature_fusion(
                    heavy_input,
                    background_feature_patches[heavy_indices].to(heavy_input.dtype),
                )
            heavy_kwargs = {}
            if transformer_background_context_patches is not None:
                heavy_kwargs["background_context"] = transformer_background_context_patches[
                    heavy_indices
                ].to(heavy_input.dtype)
            heavy_out = self.backbone(
                heavy_input,
                patch_timesteps[heavy_indices],
                **heavy_kwargs,
            )

            out_patches = out_patches.to(heavy_out.dtype)
            if base_patches is not None:
                blend_mask = self._get_heavy_patch_blend_mask(
                    heavy_out.dtype, heavy_out.device
                )
                base_selected = base_patches[heavy_indices].to(heavy_out.dtype)
                out_patches[heavy_indices] = (
                    blend_mask * heavy_out + (1.0 - blend_mask) * base_selected
                )
            else:
                out_patches[heavy_indices] = heavy_out
        elif base_background is not None and self.use_shared_background and independent_background is None:
            out = base_background
            if pad_h > 0:
                out = out[:, :, :-pad_h, :]
            if pad_w > 0:
                out = out[:, :, :, :-pad_w]
            return out

        out_patches = out_patches.view(
            batch_size,
            num_patches,
            out_patches.shape[1],
            self.route_patch_size,
            self.route_patch_size,
        )
        out = self._unpatchify(
            out_patches,
            batch_size=batch_size,
            out_h=x_padded.shape[-2],
            out_w=x_padded.shape[-1],
            stride=self.merge_stride,
        )
        if pad_h > 0:
            out = out[:, :, :-pad_h, :]
        if pad_w > 0:
            out = out[:, :, :, :-pad_w]
        self._update_debug_state(
            x.device,
            route_ratio=route_ratio,
            all_heavy=float(flat_route.all().item()),
            heavy_patch_count=float(flat_route.sum().item()),
            total_patch_count=float(flat_route.numel()),
            transformer_context_abs_mean=(
                transformer_background_context.abs().mean()
                if transformer_background_context is not None
                else None
            ),
            light_output_abs_mean=local_light.abs().mean(),
            background_output_abs_mean=(
                base_background.abs().mean() if base_background is not None else None
            ),
            heavy_output_abs_mean=heavy_out.abs().mean() if flat_route.any() else None,
            heavy_output_zero_fraction=(
                (heavy_out.abs() < 1e-6).float().mean() if flat_route.any() else None
            ),
            output_abs_mean=out.abs().mean(),
            output_zero_fraction=(out.abs() < 1e-6).float().mean(),
        )
        return out
