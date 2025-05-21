"""Convolution Fusion modules."""
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = (
    "EarlyFusion",
    "EarlyFusionRB"
    # "LateFusionYOLOv11"
)

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class ResidualBottleneck(nn.Module):
    """
    Bottleneck with residual connection
    """
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels // 2
            print(mid_channels)

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels, eps=0.001, momentum=0.03)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels, eps=0.001, momentum=0.03)
        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.03)

        self.shortcut = nn.Identity()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.03)
            )

        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.act(self.bn1(self.conv1(x)))
        out = self.act(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        out += self.shortcut(identity)
        out = self.act(out)
        return out

class EarlyFusion(nn.Module):
    def __init__(self, c1, c2, k, s, p=None, g=1, d=1, act=True, fusion_mode=0):
        assert c2 % 2 == 0, f"Output channels must be even but got {c2} for fusion layer"

        super().__init__()
        half_filter = int(c2 / 2)
        down_filter = int(half_filter / 2)
        self.fusion_mode = fusion_mode

        self.rgb_conv1 = nn.Conv2d(3, half_filter, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        if c1 > 3:
            self.ir_conv1 = nn.Conv2d(c1 - 3, half_filter, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        else:
            self.ir_conv1 = None

        self.stem_block_rgb = nn.Sequential(
            nn.Conv2d(half_filter, down_filter, kernel_size=1, stride=1),
            nn.Conv2d(down_filter, half_filter, kernel_size=3, stride=1),
            nn.Conv2d(half_filter, half_filter, kernel_size=1, stride=1),
        )

        self.stem_block_ir = nn.Sequential(
            nn.Conv2d(half_filter, down_filter, kernel_size=1, stride=1),
            nn.Conv2d(down_filter, half_filter, kernel_size=3, stride=1),
            nn.Conv2d(half_filter, half_filter, kernel_size=1, stride=1),
        )

        self.bn_ir = nn.BatchNorm2d(half_filter, eps=0.001, momentum=0.03, affine=True)
        self.bn_rgb = nn.BatchNorm2d(half_filter, eps=0.001, momentum=0.03, affine=True)

        self.act_rgb = nn.SiLU(inplace=True)
        self.act_ir = nn.SiLU(inplace=True)

    def forward(self, x):
        rgb_features = self.rgb_conv1(x[:, :self.rgb_conv1.in_channels, :, :])
        stem_output_rgb = self.act_rgb(self.bn_rgb(self.stem_block_rgb(rgb_features)))

        if x.shape[1] > self.rgb_conv1.in_channels and self.ir_conv1 is not None:
            ir_features = self.ir_conv1(x[:, self.rgb_conv1.in_channels:, :, :])
            stem_output_ir = self.act_ir(self.bn_ir(self.stem_block_ir(ir_features)))
        else:
            stem_output_ir = torch.zeros_like(stem_output_rgb)
            if self.training:
                print(f"[Info] No IR channels detected. IR branch skipped, filled with zeros.")

        if self.fusion_mode == 0:
            fused_features = torch.cat((stem_output_rgb, stem_output_ir), dim=1)
            print(fused_features.shape)
        else:
            weights = F.softmax(self.fusion_weights, dim=0)
            fused_features = torch.cat((stem_output_rgb * weights[0], stem_output_ir * weights[1]), dim=1)

        return fused_features

    def forward_fuse(self, x):
        rgb_features = self.rgb_conv1(x[:, :self.rgb_conv1.in_channels, :, :])
        stem_output_rgb = self.act_rgb(self.stem_block_rgb(rgb_features))

        if x.shape[1] > self.rgb_conv1.in_channels and self.ir_conv1 is not None:
            ir_features = self.ir_conv1(x[:, self.rgb_conv1.in_channels:, :, :])
            stem_output_ir = self.act_ir(self.stem_block_ir(ir_features))
        else:
            stem_output_ir = torch.zeros_like(stem_output_rgb)
            if self.training:
                print(f"[Info] No IR channels detected. IR branch skipped, filled with zeros.")

        if self.fusion_mode == 0:
            fused_features = torch.cat((stem_output_rgb, stem_output_ir), dim=1)
        else:
            weights = F.softmax(self.fusion_weights, dim=0)
            fused_features = torch.cat((stem_output_rgb * weights[0], stem_output_ir * weights[1]), dim=1)

        return fused_features

class EarlyFusionRB(nn.Module):
    def __init__(self, c1, c2, k, s, p=None, g=1, d=1, act=True, fusion_mode=0):
        assert c2 % 2 == 0, f"Output channels must be even but got {c2} for fusion layer"

        super().__init__()
        half_filter = int(c2 / 2)
        down_filter = int(half_filter / 2)
        self.fusion_mode = fusion_mode

        self.rgb_conv1 = nn.Conv2d(3, half_filter, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        if c1 > 3:
            self.ir_conv1 = nn.Conv2d(c1 - 3, half_filter, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        else:
            self.ir_conv1 = None

        self.stem_block_rgb = ResidualBottleneck(half_filter, half_filter, down_filter)
        self.stem_block_ir = ResidualBottleneck(half_filter, half_filter, down_filter)

        if fusion_mode == 1:
            self.fusion_weights = nn.Parameter(torch.ones(2), requires_grad=True)

    def forward(self, x):
        rgb_features = self.rgb_conv1(x[:, :self.rgb_conv1.in_channels, :, :])
        stem_output_rgb = self.stem_block_rgb(rgb_features)

        if x.shape[1] > self.rgb_conv1.in_channels and self.ir_conv1 is not None:
            ir_features = self.ir_conv1(x[:, self.rgb_conv1.in_channels:, :, :])
            stem_output_ir = self.stem_block_ir(ir_features)
        else:
            stem_output_ir = torch.zeros_like(stem_output_rgb)
            if self.training:
                print(f"[Info] No IR channels detected. IR branch skipped, filled with zeros.")

        if self.fusion_mode == 0:
            fused_features = torch.cat((stem_output_rgb, stem_output_ir), dim=1)
            print(fused_features.shape)
        else:
            weights = F.softmax(self.fusion_weights, dim=0)
            fused_features = torch.cat((stem_output_rgb * weights[0], stem_output_ir * weights[1]), dim=1)

        return fused_features
