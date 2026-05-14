# hibinet.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from pvtv2 import pvt_v2_b2  # Links directly to your pvtv2.py backbone file


class ConvBlock(nn.Module):
    """Standard structural decoder block: Conv(3x3) -> BN -> ReLU"""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class SplitChannelMKIBlock(nn.Module):
    """
    Optimized Multi-scale Kernel Interaction Block.
    Splits input channels into parallel 1/4 subsets to maintain parameter efficiency.
    Replaces the heavy 7x7 spatial matrix with a 3x3 dilated kernel (Dilation=2)
    to expand the Effective Receptive Field smoothly on highly downsampled maps.
    """

    def __init__(self, in_ch, out_ch, reduction=16):
        super().__init__()
        self.split_ch = in_ch // 4

        # 1x1 Input projection bottleneck
        self.pre_conv = nn.Conv2d(in_ch, self.split_ch * 3, kernel_size=1, bias=False)

        # Parallel scale pathways utilizing lightweight channel-grouped convolutions
        self.branch_3x3 = nn.Conv2d(self.split_ch, self.split_ch, kernel_size=3, padding=1, groups=self.split_ch,
                                    bias=False)
        self.branch_5x5 = nn.Conv2d(self.split_ch, self.split_ch, kernel_size=5, padding=2, groups=self.split_ch,
                                    bias=False)
        self.branch_dilated = nn.Conv2d(self.split_ch, self.split_ch, kernel_size=3, padding=2, dilation=2,
                                        groups=self.split_ch, bias=False)

        # Re-fusion mapping
        self.post_conv = nn.Sequential(
            nn.Conv2d(self.split_ch * 3, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

        # Dual-Gating Recalibration Mechanics
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_ch, out_ch // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch // reduction, out_ch, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        self.spatial_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_ch, out_ch, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        residual = x
        bottleneck = self.pre_conv(x)

        # Explicitly chunk features along the channel dimension
        ch1, ch2, ch3 = torch.chunk(bottleneck, chunks=3, dim=1)

        # Run parallel multi-scale extractions
        feat_3x3 = self.branch_3x3(ch1)
        feat_5x5 = self.branch_5x5(ch2)
        feat_dilated = self.branch_dilated(ch3)

        # Re-concatenate clean structural sub-streams
        merged = torch.cat([feat_3x3, feat_5x5, feat_dilated], dim=1)
        out = self.post_conv(merged)

        # Apply Channel-Spatial Dual Modulation
        out = out * self.channel_gate(out)
        out = out * self.spatial_gate(out)
        return out + residual


class GatedSpatialFusionModule(nn.Module):
    """
    Enhanced Spatial-Domain Interaction (eSDI) Module.
    Replaces element-wise multiplication with feature Concatenation to ensure
    geometric boundaries are never zeroed out or erased due to sub-pixel misalignment.
    """

    def __init__(self, high_ch, low_ch, out_ch, reduction=16):
        super().__init__()
        self.proj_high = nn.Sequential(
            nn.Conv2d(high_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.proj_low = nn.Sequential(
            nn.Conv2d(low_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

        # Concat fusion layer: (out_ch + out_ch) -> out_ch
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(out_ch * 2, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_ch, out_ch // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch // reduction, out_ch, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, f_high, f_low):
        # Align spatial dimensions dynamically via bilinear upsampling
        if f_high.shape[2:] != f_low.shape[2:]:
            f_high = F.interpolate(f_high, size=f_low.shape[2:], mode='bilinear', align_corners=True)

        feat_high = self.proj_high(f_high)
        feat_low = self.proj_low(f_low)

        # Concatenate features safely to guarantee edge texture preservation
        concat_feat = torch.cat([feat_high, feat_low], dim=1)
        fused = self.fuse_conv(concat_feat)

        # Channel recalibration gate
        gated_fused = fused * self.se(fused)
        return gated_fused + feat_high


class HIBINet(nn.Module):
    """
    HIBI-Net Architecture Blueprint
    Backbone: PVTv2-B2 (Hierarchical Feature Extractor)
    Bottleneck Module: Split-Channel MKI Block
    Decoder Modules: Concat Gated Spatial Fusion (eSDI Overhaul)
    Dual Outputs: Region Mask and Boundary Reference Edge Map
    """

    def __init__(self, num_classes=1, pretrained=True):
        super().__init__()
        self.backbone = pvt_v2_b2()
        if pretrained:
            logger = logging.getLogger(__name__)
            try:
                # Attempts to load your standard local backbone checkpoint safely
                # Replace path string matching your local setup
                checkpoint = torch.load('./pretrained/pvt_v2_b2.pth', map_location='cpu')
                self.backbone.load_state_dict(checkpoint, strict=False)
                print("--- PVTv2-B2 pre-trained weights mounted successfully ---")
            except Exception as e:
                print(f"--- Warning: Pre-trained backbone checkpoint bypassed ({e}) ---")

        # PVTv2-B2 Native Output Channels: f1=64, f2=128, f3=320, f4=512
        self.mki = SplitChannelMKIBlock(in_ch=512, out_ch=512)

        # Context connectors bridging the semantic-spatial resolution gaps
        self.sdi_stage3 = GatedSpatialFusionModule(high_ch=512, low_ch=320, out_ch=256)
        self.sdi_stage2 = GatedSpatialFusionModule(high_ch=256, low_ch=128, out_ch=128)

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.decoder_stage1 = ConvBlock(128 + 64, 64)

        # Region Prediction Decoder Branch
        self.segmentation_head = nn.Conv2d(64, num_classes, kernel_size=1)

        # Structural Boundary Prediction Decoder Branch
        self.boundary_head = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, kernel_size=1)
        )

    def forward(self, x):
        original_size = x.shape[2:]
        # Extract hierarchical features from PVTv2
        f1, f2, f3, f4 = self.backbone.forward_features(x)

        # 1. Process deepest features through the multi-scale bottleneck
        mki_feat = self.mki(f4)

        # 2. Re-route structural features through gated progressive decoder steps
        d3 = self.sdi_stage3(mki_feat, f3)
        d2 = self.sdi_stage2(d3, f2)

        # 3. Final aggregation stage
        d2_up = self.up(d2)
        if d2_up.shape[2:] != f1.shape[2:]:
            d2_up = F.interpolate(d2_up, size=f1.shape[2:], mode='bilinear', align_corners=True)

        d1 = self.decoder_stage1(torch.cat([d2_up, f1], dim=1))

        # 4. Extract region maps and boundary targets
        seg_logits = self.segmentation_head(d1)
        bnd_logits = self.boundary_head(d1)

        # Up-sample both maps cleanly to restore input geometry
        seg_out = F.interpolate(seg_logits, size=original_size, mode='bilinear', align_corners=True)
        bnd_out = F.interpolate(bnd_logits, size=original_size, mode='bilinear', align_corners=True)

        return seg_out, bnd_out


if __name__ == '__main__':
    # Standalone sanity test execution loop
    model = HIBINet(pretrained=False).cuda()
    mock_batch = torch.randn(2, 3, 352, 352).cuda()
    seg, bnd = model(mock_batch)
    print(f"\nVerification Passed!")
    print(f"Region Map Tensor Shape: {seg.shape}")
    print(f"Boundary Map Tensor Shape: {bnd.shape}")
