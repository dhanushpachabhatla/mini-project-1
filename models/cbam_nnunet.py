import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# CBAM (3D)
# -------------------------
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)

        self.fc = nn.Sequential(
            nn.Conv3d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels // reduction, in_channels, 1, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = self.fc(self.avg_pool(x))
        max_ = self.fc(self.max_pool(x))
        return self.sigmoid(avg + max_)


class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv3d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        max_, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg, max_], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


class CBAM3D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.ca = ChannelAttention(channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x


# -------------------------
# Residual Block
# -------------------------
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, padding=1)
        self.norm1 = nn.InstanceNorm3d(out_channels)
        self.act1  = nn.LeakyReLU(0.01, inplace=True)

        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.InstanceNorm3d(out_channels)

        self.skip = nn.Conv3d(in_channels, out_channels, 1) \
            if in_channels != out_channels else nn.Identity()

        self.act2 = nn.LeakyReLU(0.01, inplace=True)

    def forward(self, x):
        identity = self.skip(x)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.norm2(x)

        x = x + identity
        x = self.act2(x)
        return x


# -------------------------
# Upsample Block + CBAM
# -------------------------
class UpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, use_cbam=True):
        super().__init__()

        self.up = nn.ConvTranspose3d(in_channels, out_channels, 2, stride=2)
        self.conv = ResidualBlock(out_channels + skip_channels, out_channels)

        self.use_cbam = use_cbam
        if use_cbam:
            self.cbam = CBAM3D(out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)

        if self.use_cbam:
            x = self.cbam(x)

        return x


# -------------------------
# FINAL nnUNet Model
# -------------------------
class nnUNet3D_CBAM(nn.Module):
    def __init__(self, in_channels=1, out_channels=7, base_filters=24):
        super().__init__()

        f = base_filters

        # Encoder
        self.enc1 = ResidualBlock(in_channels, f)
        self.enc2 = ResidualBlock(f, f*2)
        self.enc3 = ResidualBlock(f*2, f*4)
        self.enc4 = ResidualBlock(f*4, f*8)

        self.pool = nn.MaxPool3d(2)

        # Bottleneck
        self.bottleneck = ResidualBlock(f*8, f*16)

        # Decoder (CBAM only here)
        self.up4 = UpBlock(f*16, f*8, f*8, use_cbam=True)
        self.up3 = UpBlock(f*8, f*4, f*4, use_cbam=True)
        self.up2 = UpBlock(f*4, f*2, f*2, use_cbam=True)
        self.up1 = UpBlock(f*2, f, f, use_cbam=True)

        # Deep supervision heads
        self.ds4 = nn.Conv3d(f*8, out_channels, 1)
        self.ds3 = nn.Conv3d(f*4, out_channels, 1)
        self.ds2 = nn.Conv3d(f*2, out_channels, 1)

        self.final = nn.Conv3d(f, out_channels, 1)

        self.apply(self._init_weights)

    def forward(self, x):

        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        # Decoder
        d4 = self.up4(b, e4)
        d3 = self.up3(d4, e3)
        d2 = self.up2(d3, e2)
        d1 = self.up1(d2, e1)

        out = self.final(d1)

        # Deep supervision outputs (UPSAMPLED to same size)
        ds2 = F.interpolate(self.ds2(d2), size=out.shape[2:], mode="trilinear", align_corners=False)
        ds3 = F.interpolate(self.ds3(d3), size=out.shape[2:], mode="trilinear", align_corners=False)
        ds4 = F.interpolate(self.ds4(d4), size=out.shape[2:], mode="trilinear", align_corners=False)

        return out, ds2, ds3, ds4

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)