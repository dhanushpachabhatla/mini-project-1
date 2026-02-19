import torch
import torch.nn as nn
import torch.nn.functional as F



# Residual Block (nnU-Net style)
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, padding=1)
        self.norm1 = nn.InstanceNorm3d(out_channels)
        self.act1  = nn.LeakyReLU(negative_slope=0.01,inplace=True)

        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.InstanceNorm3d(out_channels)

        # skip conv if channels differ
        self.skip = nn.Conv3d(in_channels, out_channels, 1) \
            if in_channels != out_channels else nn.Identity()

        self.act2 = nn.LeakyReLU(negative_slope=0.01,inplace=True)

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


# Upsample block
class UpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, out_channels, 2, stride=2)
        self.conv = ResidualBlock(out_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# Lightweight nnU-Net 3D Full-Res
class nnUNet3D(nn.Module):
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

        # Decoder
        self.up4 = UpBlock(f*16, f*8, f*8)
        self.up3 = UpBlock(f*8, f*4, f*4)
        self.up2 = UpBlock(f*4, f*2, f*2)
        self.up1 = UpBlock(f*2, f, f)

        # Deep supervision heads (nnU-Net key feature)
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

        # Return deep supervision outputs (only used later)
        return out
    
    def _init_weights(self, m):
        if isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
