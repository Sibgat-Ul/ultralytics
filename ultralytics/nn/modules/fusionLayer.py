"""Convolution Fusion modules."""
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = (
    "EarlyFusion",
    # "LateFusionYOLOv11"
)

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class EarlyFusion(nn.Module):
    def __init__(self, fusion_mode, c1, c2, k, s, p=None, g=1, d=1, act=True):
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
        )

        self.stem_block_ir = nn.Sequential(
            nn.Conv2d(half_filter, down_filter, kernel_size=1, stride=1),
            nn.Conv2d(down_filter, half_filter, kernel_size=3, stride=1),
        )

        self.bn = nn.BatchNorm2d(half_filter, eps=0.001, momentum=0.03, affine=True)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        # Process RGB channels
        rgb_features = self.rgb_conv1(x[:, :self.rgb_conv1.in_channels, :, :])
        stem_output_rgb = self.act(self.bn(self.stem_block_rgb(rgb_features)))

        # Process IR channels if available
        if x.shape[1] > self.rgb_conv1.in_channels and self.ir_conv1 is not None:
            ir_features = self.ir_conv1(x[:, self.rgb_conv1.in_channels:, :, :])
            stem_output_ir = self.act(self.bn(self.stem_block_ir(ir_features)))
        else:
            stem_output_ir = torch.zeros_like(stem_output_rgb)
            if self.training:  # Only print during training to avoid log spam
                print(f"[Info] No IR channels detected. IR branch skipped, filled with zeros.")

        if self.fusion_mode == 0:
            fused_features = torch.cat((stem_output_rgb, stem_output_ir), dim=1)
        else:
            weights = F.softmax(self.fusion_weights, dim=0)
            fused_features = torch.cat((stem_output_rgb * weights[0], stem_output_ir * weights[1]), dim=1)

        return fused_features

    def forward_fuse(self, x):
        # Process RGB channels
        rgb_features = self.rgb_conv1(x[:, :self.rgb_conv1.in_channels, :, :])
        stem_output_rgb = self.act(self.stem_block_rgb(rgb_features))

        # Process IR channels if available
        if x.shape[1] > self.rgb_conv1.in_channels and self.ir_conv1 is not None:
            ir_features = self.ir_conv1(x[:, self.rgb_conv1.in_channels:, :, :])
            stem_output_ir = self.act(self.stem_block_ir(ir_features))
        else:
            stem_output_ir = torch.zeros_like(stem_output_rgb)
            if self.training:  # Only print during training to avoid log spam
                print(f"[Info] No IR channels detected in forward_fuse. IR branch skipped, filled with zeros.")

        # For forward_fuse, we always use simple concatenation as per Ultralytics standard
        fused_features = torch.cat((stem_output_rgb, stem_output_ir), dim=1)
        return fused_features

# class LateFusionYOLOv11(nn.Module):
#     def __init__(self, c_inr=3, c_ini=1, c2=64, k=1, s=1, p=None, g=1, d=1, act=True):
#         assert c2 % 2 is 0, "Output filters must be even"
#
#         super().__init__()
#         self.rgb_conv1 = nn.Conv2d(c_inr, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
#         self.ir_conv1 = nn.Conv2d(c_ini, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
#
#         half_filter = int(c2 / 2)
#
#     def forward(self, x):
#         rgb_features = self.rgb_conv1(x[:, :3, :, :])
#         ir_features = self.ir_conv1(x[:, 3:4, :, :])
#
#         stem_output_rgb = self.stem_block(rgb_features)
#         stem_output_ir = self.stem_block(ir_features)
#
#         fused_features = torch.cat((stem_output_rgb, stem_output_ir), dim=1)
#
#         return fused_features