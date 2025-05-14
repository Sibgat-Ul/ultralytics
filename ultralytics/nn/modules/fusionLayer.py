"""Convolution Fusion modules."""
import torch
import torch.nn as nn

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
    def __init__(self, c1=4, c2=64, k=1, s=1, p=None, g=1, d=1, act=True):
        assert c2 % 2 is 0, f"params: {c1, c2, k, s, p, g, d, act}"
        super().__init__()

        half_filter = int(c2/2)
        down_filter = int(half_filter/2)
        print(f"params: {c1, c2, k, s, p, g, d, act}")

        self.rgb_conv1 = nn.Conv2d(c1, half_filter, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.ir_conv1 = nn.Conv2d(c1-3, half_filter, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)

        self.stem_block = nn.Sequential(
            nn.Conv2d(half_filter, down_filter, kernel_size=1, stride=1),
            nn.Conv2d(down_filter, half_filter, kernel_size=3, stride=2, padding=1),
        )

        self.bn = nn.BatchNorm2d(half_filter, eps=0.001, momentum=0.03, affine=True)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        print(x.shape)
        rgb_features = self.rgb_conv1(x[:, :3, :, :])
        ir_features = self.ir_conv1(x[:, 3:, :, :])

        stem_output_rgb = self.act(self.bn(self.stem_block(rgb_features)))
        stem_output_ir = self.act(self.bn(self.stem_block(ir_features)))

        fused_features = torch.cat((stem_output_rgb, stem_output_ir), dim=1)

        return fused_features

    def forward_fuse(self, x):
        rgb_features = self.rgb_conv1(x[:, :3, :, :])
        ir_features = self.ir_conv1(x[:, 3:, :, :])

        stem_output_rgb = self.act(self.stem_block(rgb_features))
        stem_output_ir = self.act(self.stem_block(ir_features))

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
#         self.stem_block = nn.Sequential(
#             nn.Conv2d(c2, half_filter, kernel_size=1, stride=1),
#             nn.Conv2d(half_filter, c2, kernel_size=3, stride=2, padding=1),
#             nn.BatchNorm2d(c2, eps=0.001, momentum=0.03, affine=True),
#             nn.SiLU(inplace=True)
#         )
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