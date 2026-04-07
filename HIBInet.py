# HIBInet.py
"""
HIBI-Net: A Lightweight Hybrid Framework with Multi-Scale Context Aggregation and Boundary Regularization for Polyp Segmentation
A complete, self-contained model file combining:
- PVTv2 backbone
- SDI (Scale-Discriminative Integration) modules
- KAN (Kernel Attention Network) blocks
- Boundary prediction head
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging
from functools import partial


# ============================================================================
# PART 1: PVTv2 Backbone Components
# ============================================================================

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.sr_ratio = sr_ratio

        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0.,
                 attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        self.drop_path = nn.Identity() if drop_path <= 0. else nn.Dropout(drop_path)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x


class OverlapPatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        # Convert patch_size to tuple if it's an integer
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class PyramidVisionTransformerV2(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000,
                 embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8],
                 mlp_ratios=[8, 8, 4, 4], qkv_bias=True, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1]):
        super().__init__()

        # patch_embed
        self.patch_embed1 = OverlapPatchEmbed(img_size=img_size, patch_size=7, stride=4,
                                              in_chans=in_chans, embed_dim=embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2,
                                              in_chans=embed_dims[0], embed_dim=embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2,
                                              in_chans=embed_dims[1], embed_dim=embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 16, patch_size=3, stride=2,
                                              in_chans=embed_dims[2], embed_dim=embed_dims[3])

        # transformer encoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0

        self.block1 = nn.ModuleList([
            Block(dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0],
                  qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate,
                  drop_path=dpr[cur + i], norm_layer=norm_layer, sr_ratio=sr_ratios[0])
            for i in range(depths[0])
        ])
        self.norm1 = norm_layer(embed_dims[0])

        cur += depths[0]
        self.block2 = nn.ModuleList([
            Block(dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1],
                  qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate,
                  drop_path=dpr[cur + i], norm_layer=norm_layer, sr_ratio=sr_ratios[1])
            for i in range(depths[1])
        ])
        self.norm2 = norm_layer(embed_dims[1])

        cur += depths[1]
        self.block3 = nn.ModuleList([
            Block(dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2],
                  qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate,
                  drop_path=dpr[cur + i], norm_layer=norm_layer, sr_ratio=sr_ratios[2])
            for i in range(depths[2])
        ])
        self.norm3 = norm_layer(embed_dims[2])

        cur += depths[2]
        self.block4 = nn.ModuleList([
            Block(dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3],
                  qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate,
                  drop_path=dpr[cur + i], norm_layer=norm_layer, sr_ratio=sr_ratios[3])
            for i in range(depths[3])
        ])
        self.norm4 = norm_layer(embed_dims[3])

    def forward_features(self, x):
        B = x.shape[0]
        outs = []

        # stage 1
        x, H, W = self.patch_embed1(x)
        for blk in self.block1:
            x = blk(x, H, W)
        x = self.norm1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 2
        x, H, W = self.patch_embed2(x)
        for blk in self.block2:
            x = blk(x, H, W)
        x = self.norm2(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 3
        x, H, W = self.patch_embed3(x)
        for blk in self.block3:
            x = blk(x, H, W)
        x = self.norm3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 4
        x, H, W = self.patch_embed4(x)
        for blk in self.block4:
            x = blk(x, H, W)
        x = self.norm4(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        return outs

    def forward(self, x):
        return self.forward_features(x)


def pvt_v2_b2(pretrained=False, **kwargs):
    model = PyramidVisionTransformerV2(
        patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8],
        mlp_ratios=[8, 8, 4, 4], qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
        depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], drop_rate=0.0, drop_path_rate=0.1, **kwargs)

    if pretrained:
        try:
            state_dict = torch.load('pretrained/pvt_v2_b2.pth', map_location='cpu')
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            model.load_state_dict(state_dict, strict=False)
            logging.info("Loaded pretrained PVTv2-B2 weights")
        except:
            logging.warning("Pretrained weights not found, using random initialization")

    return model


# ============================================================================
# PART 2: SDI Components
# ============================================================================

class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block for channel-wise feature recalibration."""

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SDIModule(nn.Module):
    """Scale-Discriminative Integration module with Hadamard attention fusion."""

    def __init__(self, low_channels, high_channels, out_channels):
        super().__init__()
        # Low-level projection
        self.conv_low = nn.Sequential(
            nn.Conv2d(low_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        # High-level processing
        self.conv_high = nn.Sequential(
            nn.Conv2d(high_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        # SE Gating Block
        self.se = SEBlock(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, f_low, f_high):
        # 1. Align resolution
        f_low_upsampled = F.interpolate(f_low, size=f_high.shape[2:], mode='bilinear', align_corners=False)
        tilde_f_low = self.conv_low(f_low_upsampled)
        tilde_f_high = self.conv_high(f_high)

        # 2. Hadamard Attention
        A = tilde_f_low * tilde_f_high

        # 3. SE Gating + Residual
        refined_A = self.se(A)
        R = self.relu(refined_A + tilde_f_high)
        return R


# ============================================================================
# PART 3: KAN Components
# ============================================================================

class SplineActivation(nn.Module):
    """GPU-optimized spline-based activation function using pure PyTorch."""

    def __init__(self, num_splines=10, spline_order=3, grid_min=-2, grid_max=2):
        super().__init__()
        self.num_splines = num_splines
        self.spline_order = spline_order
        self.grid_min = grid_min
        self.grid_max = grid_max
        num_knots = num_splines + spline_order + 1
        self.register_buffer('knots', torch.linspace(grid_min, grid_max, num_knots))
        self.coeffs = nn.Parameter(torch.empty(num_splines).uniform_(-0.1, 0.1))

    def _compute_b_spline_basis_pytorch(self, x, i, k):
        knots = self.knots
        if k == 0:
            return ((x >= knots[i]) & (x < knots[i + 1])).float()

        term1 = torch.zeros_like(x)
        term2 = torch.zeros_like(x)

        denom1 = knots[i + k] - knots[i]
        if denom1 != 0:
            term1 = (x - knots[i]) / denom1 * self._compute_b_spline_basis_pytorch(x, i, k - 1)

        denom2 = knots[i + k + 1] - knots[i + 1]
        if denom2 != 0:
            term2 = (knots[i + k + 1] - x) / denom2 * self._compute_b_spline_basis_pytorch(x, i + 1, k - 1)

        return term1 + term2

    def _vectorized_b_spline_basis(self, x):
        original_shape = x.shape
        x_flat = x.reshape(-1)
        basis = torch.zeros(x_flat.shape[0], self.num_splines, device=x.device, dtype=x.dtype)

        for i in range(self.num_splines):
            basis[:, i] = self._compute_b_spline_basis_pytorch(x_flat, i, self.spline_order)

        basis[:, -1] += (x_flat == self.knots[-1]).float()
        return basis.reshape(*original_shape, self.num_splines)

    def forward(self, x):
        if self.knots.device != x.device:
            self.knots = self.knots.to(x.device)

        x_clamped = torch.clamp(x, self.grid_min, self.grid_max)
        basis = self._vectorized_b_spline_basis(x_clamped)
        activated = torch.sum(basis * self.coeffs, dim=-1)
        return activated


class ConvKANLayer(nn.Module):
    """Core KAN block: Z_{k+1} = LayerNorm(Z_k + DwConv(phi(Z_k)))"""

    def __init__(self, channels=512):
        super().__init__()
        self.spline_activation = SplineActivation(num_splines=10)
        self.depthwise_conv = nn.Conv2d(
            in_channels=channels, out_channels=channels,
            kernel_size=3, padding=1, groups=channels, bias=False
        )
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        activated_x = self.spline_activation(x)
        dw_conv_out = self.depthwise_conv(activated_x)
        out = x + dw_conv_out

        out = out.permute(0, 2, 3, 1)
        out = self.norm(out)
        out = out.permute(0, 3, 1, 2)
        return out


class KANBottleneck(nn.Module):
    """KAN Bottleneck module processing feature maps with 3 stacked layers."""

    def __init__(self, in_channels=512, depth=3):
        super().__init__()
        self.layers = nn.ModuleList([
            ConvKANLayer(in_channels) for _ in range(depth)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


# KAN Block for decoder integration
class SEConv(nn.Module):
    """Squeeze-and-Excitation for KAN blocks."""

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class DepthwiseConv(nn.Module):
    """Depthwise convolution for KAN blocks."""

    def __init__(self, ch, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, kernel_size, padding=kernel_size // 2, groups=ch, bias=False)
        self.bn = nn.BatchNorm2d(ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class KANBlock(nn.Module):
    """Kernel Attention Network block for decoder integration."""

    def __init__(self, in_ch, out_ch=None, reduction=16, use_gate=True, mid_ch=None):
        super().__init__()
        out_ch = out_ch or in_ch
        mid_ch = mid_ch or max(in_ch // 2, 8)

        self.pre = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True)
        )

        self.dw3 = DepthwiseConv(mid_ch, kernel_size=3)
        self.dw5 = DepthwiseConv(mid_ch, kernel_size=5)
        self.dw7 = DepthwiseConv(mid_ch, kernel_size=7)

        self.fuse = nn.Sequential(
            nn.Conv2d(mid_ch * 3, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

        self.se = SEConv(out_ch, reduction=reduction)
        self.use_gate = use_gate

        if use_gate:
            self.gate = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(out_ch, out_ch // 4, 1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch // 4, out_ch, 1),
                nn.Sigmoid()
            )

    def forward(self, x, skip=None):
        input_dtype = x.dtype
        try:
            param_dtype = next(self.parameters()).dtype
        except StopIteration:
            param_dtype = torch.float32

        if input_dtype != param_dtype:
            x = x.to(param_dtype)
            if skip is not None:
                skip = skip.to(param_dtype)

        z = self.pre(x)
        b1 = self.dw3(z)
        b2 = self.dw5(z)
        b3 = self.dw7(z)
        cat = torch.cat([b1, b2, b3], dim=1)
        out = self.fuse(cat)
        out = self.se(out)

        if skip is not None:
            if skip.shape[2:] != out.shape[2:]:
                skip = F.interpolate(skip, size=out.shape[2:], mode='bilinear', align_corners=False)
            out = out + skip

        if self.use_gate:
            g = self.gate(out)
            out = out * g

        if out.dtype != input_dtype:
            out = out.to(input_dtype)

        return out


# ============================================================================
# PART 4: Decoder Components
# ============================================================================

class ConvBlock(nn.Module):
    """Basic convolutional block with BN and ReLU."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class BaselineDecoder(nn.Module):
    """Progressive decoder with optional boundary prediction."""

    def __init__(self, num_classes=1, return_boundary=False):
        super().__init__()
        self.return_boundary = return_boundary

        # Stage 4 (f4 -> d4)
        self.conv_f4 = ConvBlock(512, 256)
        self.up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_d4 = ConvBlock(256 + 320, 256)

        # Stage 3 (d4 -> d3)
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_d3 = ConvBlock(256 + 128, 128)

        # Stage 2 (d3 -> d2)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_d2 = ConvBlock(128 + 64, 64)

        # Final Segmentation Head
        self.out_conv = nn.Conv2d(64, num_classes, kernel_size=1)

        # Boundary Head (optional)
        if self.return_boundary:
            self.boundary_head = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, f4, f3, f2, f1, original_size):
        d4 = self.conv_f4(f4)
        d4 = self.up4(d4)
        d4 = torch.cat([d4, f3], dim=1)
        d3 = self.conv_d4(d4)

        d3 = self.up3(d3)
        d3 = torch.cat([d3, f2], dim=1)
        d2 = self.conv_d3(d3)

        d2 = self.up2(d2)
        d2 = torch.cat([d2, f1], dim=1)
        d1 = self.conv_d2(d2)

        # Segmentation output
        out = self.out_conv(d1)
        seg_out = F.interpolate(out, size=original_size, mode='bilinear', align_corners=False)

        # Boundary output (if enabled)
        if self.return_boundary:
            bnd = self.boundary_head(d1)
            bnd_out = F.interpolate(bnd, size=original_size, mode='bilinear', align_corners=False)
            return seg_out, bnd_out

        return seg_out


def make_kan_modules(decoder_channels, kan_cfg):
    """Create KAN modules for specific decoder stages."""
    modules = nn.ModuleDict()
    for idx, ch in enumerate(decoder_channels):
        if idx in kan_cfg.get('apply_at', []):
            modules[f'kan_{idx}'] = KANBlock(
                in_ch=ch, out_ch=ch,
                reduction=kan_cfg.get('reduction', 16),
                use_gate=kan_cfg.get('use_gate', True)
            )
    return modules


class DecoderWithKAN(nn.Module):
    """Wrapper that applies KAN modules to decoder features."""

    def __init__(self, base_decoder: nn.Module, kan_modules: nn.ModuleDict, apply_before: bool = True):
        super().__init__()
        self.base = base_decoder
        self.kan_modules = kan_modules or nn.ModuleDict()
        self.apply_before = apply_before

    def forward(self, f4, f3, f2, f1, original_size, **kwargs):
        features = [f4, f3, f2, f1]
        processed_features = []

        for i, feat in enumerate(features):
            if feat is None:
                processed_features.append(None)
                continue

            key = f'kan_{i}'
            if key in self.kan_modules:
                kan_module = self.kan_modules[key]
                feat_device = feat.device
                kan_module = kan_module.to(feat_device)

                # Handle dtype compatibility
                orig_dtype = feat.dtype
                try:
                    param_dtype = next(kan_module.parameters()).dtype
                except StopIteration:
                    param_dtype = torch.float32

                feat_for_kan = feat.to(device=feat_device, dtype=param_dtype, copy=False)

                # Run KAN without autocast
                processed = kan_module(feat_for_kan)

                if processed.dtype != orig_dtype:
                    processed_feat = processed.to(device=feat_device, dtype=orig_dtype)
                else:
                    processed_feat = processed

                processed_features.append(processed_feat)
            else:
                processed_features.append(feat)

        return self.base(
            processed_features[0],
            processed_features[1],
            processed_features[2],
            processed_features[3],
            original_size=original_size,
            **kwargs
        )


# ============================================================================
# PART 5: Main HIBINet Model
# ============================================================================

class HIBINet(nn.Module):
    """
    HIBI-Net: A Lightweight Hybrid Framework with Multi-Scale Context Aggregation and Boundary Regularization for Polyp Segmentation
    
    Args:
        num_classes (int): Number of output classes
        use_sdi (bool): Enable SDI modules for feature fusion
        use_kan (bool): Enable KAN modules in decoder
        use_boundary (bool): Enable boundary prediction head
        backbone_pretrained (bool): Load pretrained PVTv2-B2 weights
        kan_cfg (dict): Configuration for KAN modules
    """

    def __init__(self, num_classes=1, use_sdi=False, use_kan=False,
                 use_boundary=True, backbone_pretrained=True, kan_cfg=None):
        super().__init__()

        self.use_sdi = use_sdi
        self.use_kan = use_kan
        self.use_boundary = use_boundary

        # Backbone
        self.backbone = pvt_v2_b2(pretrained=backbone_pretrained)

        # SDI modules (if enabled)
        if use_sdi:
            self.sdi2 = SDIModule(low_channels=512, high_channels=320, out_channels=512)
            self.sdi1 = SDIModule(low_channels=320, high_channels=128, out_channels=320)
            self.r2_to_f3 = nn.Conv2d(512, 320, kernel_size=1, bias=False)
            self.r1_to_f2 = nn.Conv2d(320, 128, kernel_size=1, bias=False)

            # Initialize adapters
            nn.init.kaiming_normal_(self.r2_to_f3.weight, mode='fan_out', nonlinearity='relu')
            nn.init.kaiming_normal_(self.r1_to_f2.weight, mode='fan_out', nonlinearity='relu')

        # Base decoder
        self.decoder = BaselineDecoder(num_classes=num_classes, return_boundary=use_boundary)
        self.decoder_channels = [512, 320, 128, 64]

        # KAN modules (if enabled)
        if use_kan:
            if kan_cfg is None:
                kan_cfg = {'apply_at': [1, 2], 'reduction': 16, 'use_gate': True}
            kan_modules = make_kan_modules(self.decoder_channels, kan_cfg)
            self.decoder = DecoderWithKAN(self.decoder, kan_modules, apply_before=True)

            # Log KAN parameters
            n_params = sum(p.numel() for p in kan_modules.parameters() if p.requires_grad)
            logging.info(f'KAN enabled. Added params: {n_params:,}. Applied at {kan_cfg.get("apply_at", [])}.')

        # Expose boundary head for training (only if using boundary)
        self.has_boundary_head = use_boundary
        if use_boundary:
            # Get boundary head from decoder (handles both regular and KAN-wrapped decoder)
            if hasattr(self.decoder, 'base'):
                self.boundary_head = self.decoder.base.boundary_head
            else:
                self.boundary_head = self.decoder.boundary_head

    def forward(self, x):
        # Extract features from backbone
        f1, f2, f3, f4 = self.backbone(x)

        # Apply SDI fusion if enabled
        if self.use_sdi:
            R2 = self.sdi2(f4, f3)  # (B,512,22,22)
            R1 = self.sdi1(f3, f2)  # (B,320,44,44)

            # Adapt SDI outputs to match original feature dimensions
            f3 = self.r2_to_f3(R2)  # (B,320,22,22)
            f2 = self.r1_to_f2(R1)  # (B,128,44,44)

        # Pass through decoder (with KAN if enabled)
        if self.use_boundary:
            seg_out, bnd_out = self.decoder(f4, f3, f2, f1, original_size=x.shape[2:])
            return seg_out, bnd_out
        else:
            seg_out = self.decoder(f4, f3, f2, f1, original_size=x.shape[2:])
            return seg_out


# ============================================================================
# PART 6: Training Utilities
# ============================================================================

def add_kan_args(parser):
    """Add KAN-related arguments to argparse parser."""
    parser.add_argument('--enable-kan', action='store_true', help='Enable KAN modules in decoder')
    parser.add_argument('--kan-apply-at', type=str, default='1,2',
                        help='Comma list of decoder stage indices to apply KAN (0=f4,1=f3,2=f2,3=f1)')
    parser.add_argument('--kan-reduction', type=int, default=16,
                        help='KAN SE reduction factor')
    parser.add_argument('--kan-use-gate', action='store_true',
                        help='Enable gating in KAN modules')
    return parser


# ============================================================================
# TEST CODE
# ============================================================================

if __name__ == "__main__":
    # Test all configurations
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing HIBINet on {device}")

    B, C, H, W = 2, 3, 352, 352
    x = torch.randn(B, C, H, W, device=device)

    print("\n=== Testing Configurations ===")

    # Test 1: Baseline (no SDI, no KAN, no boundary)
    print("\n1. Baseline Configuration:")
    model = HIBINet(use_sdi=False, use_kan=False, use_boundary=False).to(device)
    out = model(x)
    print(f"   Output shape: {out.shape}")

    # Test 2: SDI only
    print("\n2. SDI Only Configuration:")
    model = HIBINet(use_sdi=True, use_kan=False, use_boundary=False).to(device)
    out = model(x)
    print(f"   Output shape: {out.shape}")

    # Test 3: KAN only
    print("\n3. KAN Only Configuration:")
    model = HIBINet(use_sdi=False, use_kan=True, use_boundary=False).to(device)
    out = model(x)
    print(f"   Output shape: {out.shape}")

    # Test 4: Boundary only
    print("\n4. Boundary Only Configuration:")
    model = HIBINet(use_sdi=False, use_kan=False, use_boundary=True).to(device)
    seg, bnd = model(x)
    print(f"   Segmentation shape: {seg.shape}")
    print(f"   Boundary shape: {bnd.shape}")

    # Test 5: Full model (SDI + KAN + Boundary)
    print("\n5. Full model Configuration:")
    model = HIBINet(use_sdi=True, use_kan=True, use_boundary=True).to(device)
    seg, bnd = model(x)
    print(f"   Segmentation shape: {seg.shape}")
    print(f"   Boundary shape: {bnd.shape}")

    print("\n✅ All configurations tested successfully!")
