import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.ops.deform_conv import deform_conv2d
from mmcv.runner import BaseModule

from ..builder import ROTATED_NECKS


class CloudResidualFrequencyFilter(nn.Module):
    """Cloud-Residual Frequency Filter.

    The module uses the residual between declouded features and cloudy
    features to generate frequency-domain attention, then fuses the filtered
    declouded features back with the residual cue.
    """

    def __init__(self, channels, reduction=4, norm_cfg=None):
        super(CloudResidualFrequencyFilter, self).__init__()
        hidden_channels = max(channels // reduction, 1)
        self.freq_attention = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, 1, bias=False),
            nn.Sigmoid())
        self.fusion = ConvModule(
            channels * 2,
            channels,
            3,
            padding=1,
            norm_cfg=norm_cfg if norm_cfg is not None else dict(type='BN'))

    def forward(self, x_decloud, x_cloudy):
        residual = x_cloudy - x_decloud
        freq_residual = torch.fft.rfft2(residual, norm='ortho')
        freq_weights = self.freq_attention(torch.abs(freq_residual))
        freq_decloud = torch.fft.rfft2(x_decloud, norm='ortho')
        filtered = torch.fft.irfft2(
            freq_decloud * freq_weights,
            s=(x_decloud.size(-2), x_decloud.size(-1)),
            norm='ortho')
        return self.fusion(torch.cat([filtered, residual], dim=1))


class UncertaintyGuidedDynamicAlignment(nn.Module):
    """Uncertainty-Guided Dynamic Alignment."""

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 dilation=1,
                 groups=1,
                 deformable_groups=1):
        super(UncertaintyGuidedDynamicAlignment, self).__init__()
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.deformable_groups = deformable_groups
        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels // groups, kernel_size,
                         kernel_size))
        offset_channels = 2 * kernel_size * kernel_size * deformable_groups
        self.conv_offset = nn.Conv2d(
            in_channels,
            offset_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation)
        self.escape_factor = nn.Parameter(torch.zeros(1))
        self.init_weights()

    def init_weights(self):
        nn.init.constant_(self.conv_offset.weight, 0)
        nn.init.constant_(self.conv_offset.bias, 0)
        nn.init.kaiming_uniform_(self.weight, a=1)

    def forward(self, x, uncertainty_mask, base_offsets=None):
        offsets = base_offsets if base_offsets is not None else self.conv_offset(
            x)
        if uncertainty_mask.shape[-2:] != x.shape[-2:]:
            uncertainty_mask = F.interpolate(
                uncertainty_mask,
                size=x.shape[-2:],
                mode='bilinear',
                align_corners=False)
        if uncertainty_mask.size(1) != 1:
            uncertainty_mask = uncertainty_mask.mean(dim=1, keepdim=True)
        scale_map = 1.0 + self.escape_factor * uncertainty_mask
        guided_offsets = offsets * scale_map
        return deform_conv2d(
            x,
            guided_offsets,
            self.weight,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation)


@ROTATED_NECKS.register_module()
class CloudFeatureFusionNeck(BaseModule):
    """Insert CRFF before a downstream neck for easy ablations."""

    def __init__(self,
                 neck,
                 in_channels,
                 num_levels=None,
                 reduction=4,
                 norm_cfg=None,
                 init_cfg=None):
        super(CloudFeatureFusionNeck, self).__init__(init_cfg)
        from ..builder import build_neck

        self.neck = build_neck(neck)
        self.in_channels = list(in_channels)
        self.num_levels = num_levels or len(self.in_channels)
        self.crff_modules = nn.ModuleList([
            CloudResidualFrequencyFilter(
                channels, reduction=reduction, norm_cfg=norm_cfg)
            for channels in self.in_channels[:self.num_levels]
        ])

    def forward(self, inputs, cloudy_inputs=None):
        fused_inputs = list(inputs)
        if cloudy_inputs is not None:
            for level_idx, module in enumerate(self.crff_modules):
                fused_inputs[level_idx] = module(inputs[level_idx],
                                                 cloudy_inputs[level_idx])
        return self.neck(tuple(fused_inputs))
