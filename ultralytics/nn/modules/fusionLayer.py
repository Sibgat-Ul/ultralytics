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
                 enable_adaptive_weights=True, enable_channel_duplication=True, 
                 enable_dynamic_weights=True):
        assert c2 % 2 == 0, f"Output channels must be even but got {c2} for fusion layer"

        super().__init__()
        half_filter = int(c2 / 2)
        down_filter = int(half_filter / 2)
        self.fusion_mode = fusion_mode
        self.enable_adaptive_weights = enable_adaptive_weights
        self.enable_channel_duplication = enable_channel_duplication
        self.enable_dynamic_weights = enable_dynamic_weights
        self.c2 = c2
        
        # Core convolution layers
        self.rgb_conv1 = nn.Conv2d(3, half_filter, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        if c1 > 3:
            self.ir_conv1 = nn.Conv2d(c1 - 3, half_filter, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        else:
            self.ir_conv1 = None

        # Residual bottleneck blocks
        self.stem_block_rgb = ResidualBottleneck(half_filter, half_filter, down_filter)
        self.stem_block_ir = ResidualBottleneck(half_filter, half_filter, down_filter)
        
        # Dynamic architecture components
        self.rgb_only_projection = nn.Conv2d(half_filter, c2, 1, 1, 0, bias=False)  # 1x1 conv for channel expansion
        self.ir_only_projection = nn.Conv2d(half_filter, c2, 1, 1, 0, bias=False)
        
        # Channel duplication/adaptation layers
        if enable_channel_duplication:
            self.rgb_to_ir_adapter = nn.Sequential(
                nn.Conv2d(half_filter, half_filter, 3, 1, 1, bias=False),
                nn.BatchNorm2d(half_filter),
                nn.ReLU(inplace=True),
                nn.Conv2d(half_filter, half_filter, 1, 1, 0, bias=False)
            )
            self.ir_to_rgb_adapter = nn.Sequential(
                nn.Conv2d(half_filter, half_filter, 3, 1, 1, bias=False),
                nn.BatchNorm2d(half_filter),
                nn.ReLU(inplace=True),
                nn.Conv2d(half_filter, half_filter, 1, 1, 0, bias=False)
            )
        
        # Adaptive fusion weights
        if fusion_mode == 1 or enable_dynamic_weights:
            self.fusion_weights = nn.Parameter(torch.ones(2), requires_grad=True)
            
        # Modality presence detection weights (learnable)
        if enable_adaptive_weights:
            self.modality_importance = nn.Parameter(torch.tensor([1.0, 1.0]), requires_grad=True)
            
        # Dynamic weight generation network
        if enable_dynamic_weights:
            # Attention-based dynamic weight generator
            self.dynamic_weight_generator = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),  # Global average pooling
                nn.Conv2d(half_filter * 2, 32, 1, bias=False),  # Will handle concatenated features
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 2, 1, bias=False),  # Output 2 weights
                nn.Sigmoid()  # Ensure positive weights
            )
            
            # For single modality scenarios
            self.single_modality_weight_gen = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(half_filter, 16, 1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 2, 1, bias=False),
                nn.Sigmoid()
            )

    def detect_modalities(self, x):
        """Detect which modalities are present in the input"""
        has_rgb = x.shape[1] >= 3
        has_ir = x.shape[1] > 3 and self.ir_conv1 is not None
        return has_rgb, has_ir

    def forward(self, x):
        has_rgb, has_ir = self.detect_modalities(x)
        
        # Process RGB branch if available
        if has_rgb:
            rgb_features = self.rgb_conv1(x[:, :3, :, :])
            stem_output_rgb = self.stem_block_rgb(rgb_features)
        else:
            stem_output_rgb = None
            
        # Process IR branch if available
        if has_ir:
            ir_features = self.ir_conv1(x[:, 3:, :, :])
            stem_output_ir = self.stem_block_ir(ir_features)
        else:
            stem_output_ir = None

        # Handle different modality scenarios
        if has_rgb and has_ir:
            # Both modalities present - full fusion
            return self._fuse_both_modalities(stem_output_rgb, stem_output_ir)
            
        elif has_rgb and not has_ir:
            # Only RGB present
            return self._handle_rgb_only(stem_output_rgb)
            
        elif not has_rgb and has_ir:
            # Only IR present
            return self._handle_ir_only(stem_output_ir)
            
        else:
            raise ValueError("No valid modalities detected in input")

    def _compute_dynamic_weights(self, rgb_features, ir_features, scenario="both"):
        """Compute dynamic weights based on feature content"""
        if not self.enable_dynamic_weights:
            return torch.tensor([0.5, 0.5], device=rgb_features.device)
            
        if scenario == "both":
            # Concatenate features for joint analysis
            combined_features = torch.cat([rgb_features, ir_features], dim=1)
            raw_weights = self.dynamic_weight_generator(combined_features)  # [B, 2, 1, 1]
            
        elif scenario == "rgb_only" and rgb_features is not None:
            raw_weights = self.single_modality_weight_gen(rgb_features)
            # Bias towards RGB for this scenario
            raw_weights = raw_weights * torch.tensor([1.5, 0.5], device=rgb_features.device).view(1, 2, 1, 1)
            
        elif scenario == "ir_only" and ir_features is not None:
            raw_weights = self.single_modality_weight_gen(ir_features)
            # Bias towards IR for this scenario
            raw_weights = raw_weights * torch.tensor([0.5, 1.5], device=ir_features.device).view(1, 2, 1, 1)
            
        else:
            return torch.tensor([0.5, 0.5], device=rgb_features.device if rgb_features is not None else ir_features.device)
        
        # Normalize weights to sum to 1
        weights = raw_weights.squeeze(-1).squeeze(-1)  # [B, 2]
        weights = F.softmax(weights, dim=1)  # Ensure they sum to 1
        
        # Return mean across batch for simplicity (or you could use per-sample weights)
        return weights.mean(dim=0)

    def _fuse_both_modalities(self, rgb_features, ir_features):
        """Handle fusion when both modalities are present"""
        if self.enable_dynamic_weights:
            # Compute dynamic weights based on feature content
            dynamic_weights = self._compute_dynamic_weights(rgb_features, ir_features, "both")
            
            if self.fusion_mode == 1 and self.enable_adaptive_weights:
                # Combine dynamic weights with learned parameters
                importance_weights = F.softmax(self.modality_importance, dim=0)
                fusion_weights = F.softmax(self.fusion_weights, dim=0)
                
                # Multi-level weight combination
                final_weights = dynamic_weights * fusion_weights * importance_weights
                final_weights = final_weights / final_weights.sum()  # Renormalize
            else:
                final_weights = dynamic_weights
                
            # Apply dynamic weighted fusion
            weighted_rgb = rgb_features * final_weights[0]
            weighted_ir = ir_features * final_weights[1]
            return torch.cat((weighted_rgb, weighted_ir), dim=1)
            
        elif self.fusion_mode == 0:
            # Even for "simple concatenation", apply dynamic weighting
            if self.enable_dynamic_weights:
                dynamic_weights = self._compute_dynamic_weights(rgb_features, ir_features, "both")
                weighted_rgb = rgb_features * dynamic_weights[0]
                weighted_ir = ir_features * dynamic_weights[1]
                return torch.cat((weighted_rgb, weighted_ir), dim=1)
            else:
                # True simple concatenation (no weighting)
                return torch.cat((rgb_features, ir_features), dim=1)
        else:
            # Original weighted fusion logic
            if self.enable_adaptive_weights:
                importance_weights = F.softmax(self.modality_importance, dim=0)
                fusion_weights = F.softmax(self.fusion_weights, dim=0)
                final_weights = fusion_weights * importance_weights
                final_weights = final_weights / final_weights.sum()
            else:
                final_weights = F.softmax(self.fusion_weights, dim=0)
                
            weighted_rgb = rgb_features * final_weights[0]
            weighted_ir = ir_features * final_weights[1]
            return torch.cat((weighted_rgb, weighted_ir), dim=1)

    def _handle_rgb_only(self, rgb_features):
        """Handle scenario with only RGB modality"""
        if self.enable_channel_duplication:
            # Create synthetic IR features from RGB
            synthetic_ir = self.rgb_to_ir_adapter(rgb_features)
            
            if self.enable_dynamic_weights:
                # Compute dynamic weights for RGB + synthetic IR
                dynamic_weights = self._compute_dynamic_weights(rgb_features, synthetic_ir, "rgb_only")
                weighted_rgb = rgb_features * dynamic_weights[0]
                weighted_synthetic_ir = synthetic_ir * dynamic_weights[1]
                return torch.cat((weighted_rgb, weighted_synthetic_ir), dim=1)
            else:
                # Static adaptive weights
                adaptive_weights = torch.tensor([0.8, 0.2], device=rgb_features.device)
                weighted_rgb = rgb_features * adaptive_weights[0]
                weighted_synthetic_ir = synthetic_ir * adaptive_weights[1]
                return torch.cat((weighted_rgb, weighted_synthetic_ir), dim=1)
        else:
            # Dynamic architecture - use full capacity for RGB
            return self.rgb_only_projection(rgb_features)

    def _handle_ir_only(self, ir_features):
        """Handle scenario with only IR modality"""
        if self.enable_channel_duplication:
            # Create synthetic RGB features from IR
            synthetic_rgb = self.ir_to_rgb_adapter(ir_features)
            
            if self.enable_dynamic_weights:
                # Compute dynamic weights for synthetic RGB + IR
                dynamic_weights = self._compute_dynamic_weights(synthetic_rgb, ir_features, "ir_only")
                weighted_synthetic_rgb = synthetic_rgb * dynamic_weights[0]
                weighted_ir = ir_features * dynamic_weights[1]
                return torch.cat((weighted_synthetic_rgb, weighted_ir), dim=1)
            else:
                # Static adaptive weights
                adaptive_weights = torch.tensor([0.2, 0.8], device=ir_features.device)
                weighted_synthetic_rgb = synthetic_rgb * adaptive_weights[0]
                weighted_ir = ir_features * adaptive_weights[1]
                return torch.cat((weighted_synthetic_rgb, weighted_ir), dim=1)
        else:
            # Dynamic architecture - use full capacity for IR
            return self.ir_only_projection(ir_features)

    def get_modality_weights(self):
        """Return current learned weights for analysis"""
        weights_info = {}
        
        if hasattr(self, 'fusion_weights'):
            weights_info['fusion_weights'] = F.softmax(self.fusion_weights, dim=0).detach()
            
        if hasattr(self, 'modality_importance'):
            weights_info['modality_importance'] = F.softmax(self.modality_importance, dim=0).detach()
            
        return weights_info
