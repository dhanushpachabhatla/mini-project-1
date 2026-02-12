import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------- Double Conv ----------
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, 3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


# ---------- Attention Gate ----------
class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()

        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1),
            nn.InstanceNorm3d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1),
            nn.InstanceNorm3d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1),
            nn.InstanceNorm3d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)

        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi


# ---------- Attention UNet 3D ----------
class AttentionUNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=7, base_filters=16):
        super().__init__()

        # Encoder
        self.enc1 = DoubleConv(in_channels, base_filters)
        self.enc2 = DoubleConv(base_filters, base_filters * 2)
        self.enc3 = DoubleConv(base_filters * 2, base_filters * 4)
        self.enc4 = DoubleConv(base_filters * 4, base_filters * 8)

        self.pool = nn.MaxPool3d(2)

        self.bottleneck = DoubleConv(base_filters * 8, base_filters * 16)

        # Attention Gates
        self.att4 = AttentionGate(base_filters * 8, base_filters * 8, base_filters * 4)
        self.att3 = AttentionGate(base_filters * 4, base_filters * 4, base_filters * 2)
        self.att2 = AttentionGate(base_filters * 2, base_filters * 2, base_filters)
        self.att1 = AttentionGate(base_filters, base_filters, base_filters // 2)

        # Decoder
        self.up4 = nn.ConvTranspose3d(base_filters * 16, base_filters * 8, 2, stride=2)
        self.dec4 = DoubleConv(base_filters * 16, base_filters * 8)

        self.up3 = nn.ConvTranspose3d(base_filters * 8, base_filters * 4, 2, stride=2)
        self.dec3 = DoubleConv(base_filters * 8, base_filters * 4)

        self.up2 = nn.ConvTranspose3d(base_filters * 4, base_filters * 2, 2, stride=2)
        self.dec2 = DoubleConv(base_filters * 4, base_filters * 2)

        self.up1 = nn.ConvTranspose3d(base_filters * 2, base_filters, 2, stride=2)
        self.dec1 = DoubleConv(base_filters * 2, base_filters)

        self.out_conv = nn.Conv3d(base_filters, out_channels, 1)

    def forward(self, x):

        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        # Decoder + Attention

        d4 = self.up4(b)
        e4_att = self.att4(d4, e4)
        d4 = torch.cat([d4, e4_att], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        e3_att = self.att3(d3, e3)
        d3 = torch.cat([d3, e3_att], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        e2_att = self.att2(d2, e2)
        d2 = torch.cat([d2, e2_att], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        e1_att = self.att1(d1, e1)
        d1 = torch.cat([d1, e1_att], dim=1)
        d1 = self.dec1(d1)

        return self.out_conv(d1)
