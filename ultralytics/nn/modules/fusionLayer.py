"""Convolution Fusion modules."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

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


class ResidualBottleneck(nn.Module):
    """
    Bottleneck with residual connection
    """
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if mid_channels is None:
            mid_channels = int(out_channels/2)
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

class EarlyFusionRB(nn.Module):
    def __init__(self, c1, c2, k, s, p=None, g=1, d=1, act=True, fusion_mode=1, 
                 detection_method="learned", enable_dynamic_weights=False, 
                 attention_heads=4, attention_dim=None):
        super().__init__()
        assert c2 % 2 == 0, f"Output channels must be even but got {c2}"
        
        half_filter = c2 // 2
        self.fusion_mode = fusion_mode
        self.detection_method = detection_method
        self.enable_dynamic_weights = enable_dynamic_weights
        
        # RGB path (always 3 channels)
        self.rgb_conv = nn.Conv2d(3, half_filter, k, s, autopad(k, p, d), 
                       groups=g, dilation=d, bias=False)
        self.rgb_stem = ResidualBottleneck(half_filter, half_filter, half_filter//2)
        
        # IR path (flexible channels)
        self.ir_conv = nn.Conv2d(c1-3, half_filter, k, s, autopad(k, p, d),
                      groups=g, dilation=d, bias=False) if c1 > 3 else None
        self.ir_stem = ResidualBottleneck(half_filter, half_filter, half_filter//2)
        
        # Attention mechanisms
        self.self_attention = SelfAttention(half_filter, attention_heads, attention_dim or half_filter)
        self.cross_attention = CrossAttention(half_filter, attention_heads, attention_dim or half_filter)
        
        self.modality_detector = nn.Sequential(
            nn.AdaptiveAvgPool2d(8),
            nn.Conv2d(c1, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, 16),
            nn.SiLU(),
            nn.Linear(16, 3),
            nn.Softmax(dim=1)
        )
                     
        if enable_dynamic_weights:
            self.dynamic_weight_generator = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(half_filter*2, 8, 1, bias=False),
                nn.SiLU(),
                nn.Conv2d(8, 2, 1, bias=False),
                nn.Sigmoid()
            )
        
        self.register_buffer('blank_ir', torch.full((1, half_filter, 1, 1), 255.0))
        self.register_buffer('blank_rgb', torch.full((1, 3, 1, 1), 255.0))

    def forward(self, x):
        """Optimized forward pass with minimal conditionals"""
        B, C, H, W = x.shape
        
        # Process RGB (always present)
        rgb_in = x[:, :3] if C >= 3 else self.blank_rgb.expand(B, 3, H, W)
        rgb_features = self.rgb_stem(self.rgb_conv(rgb_in))
        
        # Process IR (conditional)
        if C > 3 and self.ir_conv is not None:
            ir_in = x[:, 3:]
            ir_features = self.ir_stem(self.ir_conv(ir_in))
        else:
            s = self.ir_conv.stride[0] if self.ir_conv is not None else self.rgb_conv.stride[0]
            ir_features = self.blank_ir.expand(B, -1, H//s, W//s)  # Match downsampled size
            if self.training and C <= 3:
                print("[Info] No IR channels detected. IR branch filled with 255 (black)")
        
        # Modality detection and fusion
        modality, confidence = self._detect_modality(x)
        
        if modality == "both":
            with torch.cuda.amp.autocast():
                rgb_features, ir_features = self.cross_attention(rgb_features, ir_features)
        elif modality == "rgb":
            rgb_features = self.self_attention(rgb_features)
        else:  # IR
            ir_features = self.self_attention(ir_features)
        
        # Dynamic or confidence-based fusion
        if self.enable_dynamic_weights:
            weights = self.dynamic_weight_generator(
                torch.cat([rgb_features, ir_features], dim=1)
            ).view(B, 2, 1, 1)
            rgb_features = rgb_features * weights[:, 0]
            ir_features = ir_features * weights[:, 1] 
        else:
            if modality != "both":
                rgb_features = rgb_features * confidence
                ir_features = ir_features * (1 - confidence)
        
        return torch.cat([rgb_features, ir_features], dim=1)

    def _detect_modality(self, x):
        """Efficient modality detection"""
        if self.detection_method == "learned":
            probs = self.modality_detector(x)
            conf, pred = torch.max(probs.mean(dim=0), dim=0)
            return ["rgb", "ir", "both"][pred.item()], conf.item()
        
        # Fallback to statistical detection
        if x.shape[1] == 3:
            return ("rgb", 0.9) if x.std() > 0.1 else ("ir", 0.7)
        elif x.shape[1] == 1:
            return ("ir", 0.95)
        else:  # 4+ channels
            return ("both", 0.8)


class SelfAttention(nn.Module):
    """Memory-optimized self-attention for single-modality features"""
    def __init__(self, in_channels, heads=8, dim_head=64):
        super().__init__()
        assert in_channels == heads * dim_head, "in_channels must be equal to heads * dim_head"
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5

        inner_dim = heads * dim_head

        # Shared QKV projection
        self.to_qkv = nn.Conv2d(in_channels, inner_dim * 3, kernel_size=1, bias=False)
        self.to_out = nn.Conv2d(inner_dim, in_channels, kernel_size=1, bias=False)

        self.norm = nn.GroupNorm(1, in_channels)
        self.use_flash = hasattr(F, "scaled_dot_product_attention")

    def forward(self, x):
        """
        Args:
            x: [B, C, H, W]
        Returns:
            out: [B, C, H, W]
        """
        residual = x
        x = self.norm(x)
        B, C, H, W = x.shape
        qkv = self.to_qkv(x)  # [B, 3*inner_dim, H, W]
        q, k, v = qkv.chunk(3, dim=1)

        # Reshape to [B, heads, H*W, dim_head]
        q = q.view(B, self.heads, self.dim_head, H * W).transpose(-2, -1)
        k = k.view(B, self.heads, self.dim_head, H * W).transpose(-2, -1)
        v = v.view(B, self.heads, self.dim_head, H * W).transpose(-2, -1)

        if self.use_flash:
            out = F.scaled_dot_product_attention(q, k, v, scale=self.scale)
        else:
            # Fallback: chunked attention (optional if memory tight)
            chunk_size = max(1, H * W // 4)
            out_chunks = []
            for i in range(0, H * W, chunk_size):
                q_chunk = q[:, :, i:i + chunk_size, :]
                attn = torch.matmul(q_chunk, k.transpose(-2, -1)) * self.scale
                attn = F.softmax(attn, dim=-1)
                out_chunk = torch.matmul(attn, v)
                out_chunks.append(out_chunk)
            out = torch.cat(out_chunks, dim=-2)

        out = out.transpose(-2, -1).contiguous().view(B, -1, H, W)
        out = self.to_out(out)
        return out + residual

class CrossAttention(nn.Module):
    def __init__(self, in_channels, heads=8, dim_head=64):
        super().__init__()
        assert in_channels % heads == 0, "in_channels must be divisible by heads"
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads
        
        # Shared projections for both modalities (50% memory reduction)
        self.to_qkv = nn.Conv2d(in_channels, inner_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(inner_dim, in_channels, 1, bias=False)
        
        # Memory-efficient group norms
        self.norm = nn.GroupNorm(1, in_channels)
        
        # Flash attention if available
        self.use_flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        
    def forward(self, x1, x2):
        """
        Args:
            x1: [B, C, H, W] - first modality features
            x2: [B, C, H, W] - second modality features
        Returns:
            Tuple of attended features (x1, x2)
        """
        residual1, residual2 = x1, x2
        x1, x2 = self.norm(x1), self.norm(x2)
        
        # Process both modalities through shared projections
        qkv1 = self.to_qkv(x1)  # [B, 3*inner_dim, H, W]
        qkv2 = self.to_qkv(x2)  # [B, 3*inner_dim, H, W]
        
        # Split into q,k,v chunks (memory efficient view operations)
        q1, k1, v1 = qkv1.chunk(3, dim=1)
        q2, k2, v2 = qkv2.chunk(3, dim=1)
        
        # Reshape for attention - using view instead of reshape
        B, _, H, W = x1.shape
        q1 = q1.view(B, self.heads, -1, H*W).transpose(-1, -2)
        k1 = k1.view(B, self.heads, -1, H*W).transpose(-1, -2)
        v1 = v1.view(B, self.heads, -1, H*W).transpose(-1, -2)
        
        q2 = q2.view(B, self.heads, -1, H*W).transpose(-1, -2)
        k2 = k2.view(B, self.heads, -1, H*W).transpose(-1, -2)
        v2 = v2.view(B, self.heads, -1, H*W).transpose(-1, -2)
        
        # Memory-efficient attention computation
        if self.use_flash:
            # Use PyTorch 2.0's optimized attention
            out1 = F.scaled_dot_product_attention(q1, k2, v2, scale=self.scale)
            out2 = F.scaled_dot_product_attention(q2, k1, v1, scale=self.scale)
        else:
            # Manual chunked attention to reduce peak memory
            chunk_size = max(1, H*W // 4)  # Process in 4 chunks
            out1, out2 = [], []
            
            for i in range(0, H*W, chunk_size):
                q1_chunk = q1[:, :, i:i+chunk_size, :]
                attn1 = (q1_chunk @ k2.transpose(-2,-1)) * self.scale
                attn1 = F.softmax(attn1, dim=-1)
                out1.append(attn1 @ v2)
                
                q2_chunk = q2[..., i:i+chunk_size, :]
                attn2 = (q2_chunk @ k1.transpose(-2,-1)) * self.scale
                attn2 = F.softmax(attn2, dim=-1)
                out2.append(attn2 @ v1)
            
            out1 = torch.cat(out1, dim=-2)
            out2 = torch.cat(out2, dim=-2)
        
        # Restore original shape
        out1 = out1.transpose(-1,-2).contiguous().view(B, -1, H, W)
        out2 = out2.transpose(-1,-2).contiguous().view(B, -1, H, W)
        
        # Final projection
        out1 = self.to_out(out1) + residual1
        out2 = self.to_out(out2) + residual2
        
        return out1, out2

# # Helper functions
# def autopad(k, p=None, d=1):
#     if d > 1:
#         k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
#     if p is None:
#         p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
#     return p


# class ResidualBottleneck(nn.Module):
#     def __init__(self, in_channels, out_channels, bottleneck_channels):
#         super().__init__()
#         self.conv1 = nn.Conv2d(in_channels, bottleneck_channels, 1, bias=False)
#         self.conv2 = nn.Conv2d(bottleneck_channels, bottleneck_channels, 3, 1, 1, bias=False)
#         self.conv3 = nn.Conv2d(bottleneck_channels, out_channels, 1, bias=False)
#         self.bn1 = nn.BatchNorm2d(bottleneck_channels)
#         self.bn2 = nn.BatchNorm2d(bottleneck_channels)
#         self.bn3 = nn.BatchNorm2d(out_channels)
#         self.relu = nn.ReLU(inplace=True)
        
#         if in_channels != out_channels:
#             self.shortcut = nn.Conv2d(in_channels, out_channels, 1, bias=False)
#         else:
#             self.shortcut = nn.Identity()
    
#     def forward(self, x):
#         identity = self.shortcut(x)
        
#         out = self.relu(self.bn1(self.conv1(x)))
#         out = self.relu(self.bn2(self.conv2(out)))
#         out = self.bn3(self.conv3(out))
        
#         out += identity
#         return self.relu(out)


# class UniversalEarlyFusionRB(nn.Module):
#     def __init__(self, c1, c2, k, s, p=None, g=1, d=1, act=True, fusion_mode=1, 
#                  detection_method="learned", enable_dynamic_weights=True, 
#                  attention_heads=8, attention_dim=None):
#         """
#         Universal fusion layer that can handle RGB, IR, or RGB+IR inputs automatically
        
#         Args:
#             detection_method: "learned", "statistical", or "explicit"
#                 - "learned": Use a small network to classify modality type
#                 - "statistical": Use channel statistics to guess modality
#                 - "explicit": Require external modality specification
#             attention_heads: Number of attention heads
#             attention_dim: Attention dimension (defaults to half_filter)
#         """
#         assert c2 % 2 == 0, f"Output channels must be even but got {c2} for fusion layer"

#         super().__init__()
#         half_filter = int(c2 / 2)
#         down_filter = int(half_filter / 2)
#         self.fusion_mode = fusion_mode
#         self.detection_method = detection_method
#         self.enable_dynamic_weights = enable_dynamic_weights
#         self.c2 = c2
#         self.attention_heads = attention_heads
#         self.attention_dim = attention_dim or half_filter
        
#         # Universal convolution layers - can handle any input
#         self.universal_conv1 = nn.Conv2d(c1, half_filter, k, s, autopad(k, p, d), 
#                                        groups=g, dilation=d, bias=False)
#         self.universal_conv2 = nn.Conv2d(c1, half_filter, k, s, autopad(k, p, d), 
#                                        groups=g, dilation=d, bias=False)
        
#         # Residual bottleneck blocks
#         self.stem_block_1 = ResidualBottleneck(half_filter, half_filter, down_filter)
#         self.stem_block_2 = ResidualBottleneck(half_filter, half_filter, down_filter)
        
#         # Attention modules
#         self.self_attention = SelfAttention(half_filter, self.attention_heads, self.attention_dim)
#         self.cross_attention = CrossAttention(half_filter, self.attention_heads, self.attention_dim)
        
#         # Modality detection networks
#         if detection_method == "learned":
#             self.modality_detector = nn.Sequential(
#                 nn.AdaptiveAvgPool2d(8),  # Reduce spatial dimensions
#                 nn.Conv2d(c1, 32, 3, 1, 1, bias=False),
#                 nn.BatchNorm2d(32),
#                 nn.ReLU(inplace=True),
#                 nn.AdaptiveAvgPool2d(1),
#                 nn.Conv2d(32, 16, 1, bias=False),
#                 nn.ReLU(inplace=True),
#                 nn.Conv2d(16, 3, 1, bias=False),  # 3 classes: RGB, IR, RGB+IR
#                 nn.Flatten()
#             )
            
#         # Dynamic weight generation
#         if enable_dynamic_weights:
#             self.dynamic_weight_generator = nn.Sequential(
#                 nn.AdaptiveAvgPool2d(1),
#                 nn.Conv2d(half_filter * 2, 32, 1, bias=False),
#                 nn.ReLU(inplace=True),
#                 nn.Conv2d(32, 2, 1, bias=False),
#                 nn.Sigmoid()
#             )
            
#         # Single modality projections
#         self.single_modality_projection = nn.Conv2d(half_filter, c2, 1, 1, 0, bias=False)
        
#         # Fusion weights
#         self.fusion_weights = nn.Parameter(torch.ones(2), requires_grad=True)

#     def detect_modality_type(self, x):
#         """
#         Detect what type of modality the input represents
        
#         Returns:
#             modality_type: "rgb", "ir", or "both" 
#             confidence: confidence score
#         """
#         if self.detection_method == "learned":
#             # Use learned detector
#             logits = self.modality_detector(x)  # [B, 3]
#             probs = F.softmax(logits, dim=1)  # [B, 3]
            
#             # Get batch-wise decision (majority vote or mean)
#             mean_probs = probs.mean(dim=0)  # [3]
#             max_idx = mean_probs.argmax().item()
#             confidence = mean_probs[max_idx].item()
            
#             modality_types = ["rgb", "ir", "both"]
#             return modality_types[max_idx], confidence
            
#         elif self.detection_method == "statistical":
#             # Use statistical heuristics
#             return self._statistical_detection(x)
            
#         else:
#             raise ValueError("Unknown detection method or explicit mode requires modality_type parameter")

#     def _statistical_detection(self, x):
#         """Statistical heuristics to guess modality type"""
#         # Calculate channel statistics
#         channel_means = x.mean(dim=[0, 2, 3])  # [C]
#         channel_stds = x.std(dim=[0, 2, 3])    # [C]
        
#         if x.shape[1] == 3:
#             # 3 channels - could be RGB or IR
#             # RGB typically has more variation between channels
#             # IR (grayscale replicated) has similar values across channels
#             channel_variation = channel_stds.std().item()  # Variation of channel std devs
            
#             if channel_variation > 0.1:  # Threshold - tune based on your data
#                 return "rgb", 0.7
#             else:
#                 return "ir", 0.6
                
#         elif x.shape[1] == 1:
#             return "ir", 0.9  # Single channel is likely IR
            
#         elif x.shape[1] == 6:
#             # Assume first 3 are RGB, last 3 are IR
#             return "both", 0.8
            
#         else:
#             # Unknown configuration - default to RGB
#             return "rgb", 0.5

#     def forward(self, x, modality_type=None):
#         """
#         Universal forward pass
        
#         Args:
#             x: Input tensor [B, C, H, W]
#             modality_type: Optional explicit modality specification
#                           "rgb", "ir", "both", or None for auto-detection
#         """
        
#         # Determine modality type
#         if modality_type is None:
#             detected_type, confidence = self.detect_modality_type(x)
#             if confidence < 0.6:
#                 print(f"[Warning] Low confidence ({confidence:.2f}) in modality detection: {detected_type}")
#         else:
#             detected_type = modality_type
            
#         # Process based on detected/specified modality
#         if detected_type == "both":
#             return self._handle_both_modalities(x)
#         elif detected_type == "rgb":
#             return self._handle_single_modality(x, "rgb")
#         elif detected_type == "ir":
#             return self._handle_single_modality(x, "ir")
#         else:
#             raise ValueError(f"Unknown modality type: {detected_type}")

#     def _handle_both_modalities(self, x):
#         """Handle input with both RGB and IR using cross-attention"""
#         # Assume first half channels are one modality, second half are another
#         mid_channel = x.shape[1] // 2
        
#         modality1_data = x[:, :mid_channel, :, :]
#         modality2_data = x[:, mid_channel:, :, :]
        
#         # Process both
#         features1 = self.stem_block_1(self.universal_conv1(modality1_data))
#         features2 = self.stem_block_2(self.universal_conv2(modality2_data))
        
#         # Apply cross-attention between modalities
#         attended_features1, attended_features2 = self.cross_attention(features1, features2)
        
#         # Dynamic fusion
#         if self.enable_dynamic_weights:
#             combined = torch.cat([attended_features1, attended_features2], dim=1)
#             weights = self.dynamic_weight_generator(combined).squeeze(-1).squeeze(-1)
#             weights = F.softmax(weights, dim=1)  # [B, 2]
            
#             # Apply per-sample weights
#             weighted_features1 = attended_features1 * weights[:, 0:1, None, None]
#             weighted_features2 = attended_features2 * weights[:, 1:2, None, None]
            
#             return torch.cat([weighted_features1, weighted_features2], dim=1)
#         else:
#             # Simple fusion
#             fusion_weights = F.softmax(self.fusion_weights, dim=0)
#             weighted_features1 = attended_features1 * fusion_weights[0]
#             weighted_features2 = attended_features2 * fusion_weights[1]
            
#             return torch.cat([weighted_features1, weighted_features2], dim=1)

#     def _handle_single_modality(self, x, modality_name):
#         """Handle single modality input (RGB or IR) using self-attention"""
#         # Process through one branch
#         features = self.stem_block_1(self.universal_conv1(x))
        
#         attended_features = self.self_attention(features)
        
#         if not self.enable_dynamic_weights:
#             return self.single_modality_projection(attended_features)
        
#         synthetic_features = self.stem_block_2(self.universal_conv2(x))
        
#         attended_synthetic = self.self_attention(synthetic_features)
        
#         combined = torch.cat([attended_features, attended_synthetic], dim=1)
#         weights = self.dynamic_weight_generator(combined).squeeze(-1).squeeze(-1)
        
#         if modality_name == "rgb":
#             bias = torch.tensor([0.8, 0.2], device=x.device)
#         else:  # IR
#             bias = torch.tensor([0.7, 0.3], device=x.device)
            
#         weights = F.softmax(weights + bias.log(), dim=1)
        
#         weighted_real = attended_features * weights[:, 0:1, None, None]
#         weighted_synthetic = attended_synthetic * weights[:, 1:2, None, None]
        
#         return torch.cat([weighted_real, weighted_synthetic], dim=1)

#     def get_modality_prediction(self, x):
#         """Get modality prediction for analysis"""
#         if self.detection_method == "learned":
#             with torch.no_grad():
#                 logits = self.modality_detector(x)
#                 probs = F.softmax(logits, dim=1)
#                 return {
#                     "predictions": probs,
#                     "batch_mean": probs.mean(dim=0),
#                     "predicted_class": ["rgb", "ir", "both"][probs.mean(dim=0).argmax()]
#                 }
#         else:
#             modality_type, confidence = self._statistical_detection(x)
#             return {
#                 "predicted_class": modality_type,
#                 "confidence": confidence
#             }


# class SelfAttention(nn.Module):
#     """Self-attention module for single modality features"""
#     def __init__(self, in_channels, heads=8, dim_head=64):
#         super().__init__()
#         self.heads = heads
#         self.dim_head = dim_head
#         self.scale = dim_head ** -0.5
        
#         inner_dim = dim_head * heads
#         self.to_qkv = nn.Conv2d(in_channels, inner_dim * 3, 1, bias=False)
#         self.to_out = nn.Conv2d(inner_dim, in_channels, 1, bias=False)
        
#         self.norm = nn.GroupNorm(1, in_channels)
        
#     def forward(self, x):
#         """
#         Args:
#             x: [B, C, H, W]
#         Returns:
#             attended_x: [B, C, H, W]
#         """
#         residual = x
#         x = self.norm(x)
        
#         b, c, h, w = x.shape
        
#         # Generate Q, K, V
#         qkv = self.to_qkv(x).chunk(3, dim=1)  # 3 * [B, inner_dim, H, W]
#         q, k, v = map(lambda t: t.view(b, self.heads, self.dim_head, h * w).transpose(-2, -1), qkv)
#         # q, k, v: [B, heads, H*W, dim_head]
        
#         # Compute attention
#         attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, heads, H*W, H*W]
#         attn = F.softmax(attn, dim=-1)
        
#         # Apply attention to values
#         out = torch.matmul(attn, v)  # [B, heads, H*W, dim_head]
#         out = out.transpose(-2, -1).contiguous().view(b, -1, h, w)  # [B, inner_dim, H, W]
        
#         # Project back
#         out = self.to_out(out)
        
#         return out + residual


# class CrossAttention(nn.Module):
#     """Cross-attention module for dual modality features"""
#     def __init__(self, in_channels, heads=8, dim_head=64):
#         super().__init__()
#         self.heads = heads
#         self.dim_head = dim_head
#         self.scale = dim_head ** -0.5
        
#         inner_dim = dim_head * heads
        
#         self.to_q1 = nn.Conv2d(in_channels, inner_dim, 1, bias=False)
#         self.to_k1 = nn.Conv2d(in_channels, inner_dim, 1, bias=False)
#         self.to_v1 = nn.Conv2d(in_channels, inner_dim, 1, bias=False)
        
#         self.to_q2 = nn.Conv2d(in_channels, inner_dim, 1, bias=False)
#         self.to_k2 = nn.Conv2d(in_channels, inner_dim, 1, bias=False)
#         self.to_v2 = nn.Conv2d(in_channels, inner_dim, 1, bias=False)
        
#         self.to_out1 = nn.Conv2d(inner_dim, in_channels, 1, bias=False)
#         self.to_out2 = nn.Conv2d(inner_dim, in_channels, 1, bias=False)
        
#         self.norm1 = nn.GroupNorm(1, in_channels)
#         self.norm2 = nn.GroupNorm(1, in_channels)
        
#     def forward(self, x1, x2):
#         """
#         Args:
#             x1: [B, C, H, W] - first modality features
#             x2: [B, C, H, W] - second modality features
#         Returns:
#             attended_x1: [B, C, H, W]
#             attended_x2: [B, C, H, W]
#         """
#         residual1, residual2 = x1, x2
#         x1, x2 = self.norm1(x1), self.norm2(x2)
        
#         b, c, h, w = x1.shape
        
#         # Generate Q, K, V for both modalities
#         q1 = self.to_q1(x1).view(b, self.heads, self.dim_head, h * w).transpose(-2, -1)
#         k1 = self.to_k1(x1).view(b, self.heads, self.dim_head, h * w).transpose(-2, -1)
#         v1 = self.to_v1(x1).view(b, self.heads, self.dim_head, h * w).transpose(-2, -1)
        
#         q2 = self.to_q2(x2).view(b, self.heads, self.dim_head, h * w).transpose(-2, -1)
#         k2 = self.to_k2(x2).view(b, self.heads, self.dim_head, h * w).transpose(-2, -1)
#         v2 = self.to_v2(x2).view(b, self.heads, self.dim_head, h * w).transpose(-2, -1)
        
#         # Cross-attention: x1 attends to x2, x2 attends to x1
#         attn1_to_2 = torch.matmul(q1, k2.transpose(-2, -1)) * self.scale
#         attn2_to_1 = torch.matmul(q2, k1.transpose(-2, -1)) * self.scale
        
#         attn1_to_2 = F.softmax(attn1_to_2, dim=-1)
#         attn2_to_1 = F.softmax(attn2_to_1, dim=-1)
        
#         out1 = torch.matmul(attn1_to_2, v2) 
#         out2 = torch.matmul(attn2_to_1, v1)
        
#         # Reshape and project
#         out1 = out1.transpose(-2, -1).contiguous().view(b, -1, h, w)
#         out2 = out2.transpose(-2, -1).contiguous().view(b, -1, h, w)
        
#         out1 = self.to_out1(out1)
#         out2 = self.to_out2(out2)
        
#         return out1 + residual1, out2 + residual2
