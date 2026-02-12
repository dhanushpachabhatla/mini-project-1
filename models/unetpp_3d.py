import torch
import torch.nn as nn


# -------------------------
# Double Convolution Block
# -------------------------
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


# -------------------------
# UNet++ 3D (Lightweight)
# -------------------------
class UNetPP3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=7, base_filters=16):
        super().__init__()

        f = base_filters

        self.pool = nn.MaxPool3d(2)
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

        # Encoder
        self.conv00 = DoubleConv(in_channels, f)
        self.conv10 = DoubleConv(f, f*2)
        self.conv20 = DoubleConv(f*2, f*4)
        self.conv30 = DoubleConv(f*4, f*8)

        # Nested Decoder
        self.conv01 = DoubleConv(f + f*2, f)
        self.conv11 = DoubleConv(f*2 + f*4, f*2)
        self.conv21 = DoubleConv(f*4 + f*8, f*4)

        self.conv02 = DoubleConv(f*2 + f, f)
        self.conv12 = DoubleConv(f*4 + f*2, f*2)

        self.conv03 = DoubleConv(f*3 + f, f)

        self.final = nn.Conv3d(f, out_channels, kernel_size=1)

    def forward(self, x):

        # Encoder
        x00 = self.conv00(x)
        x10 = self.conv10(self.pool(x00))
        x20 = self.conv20(self.pool(x10))
        x30 = self.conv30(self.pool(x20))

        # Decoder level 1
        x01 = self.conv01(torch.cat([x00, self.up(x10)], dim=1))
        x11 = self.conv11(torch.cat([x10, self.up(x20)], dim=1))
        x21 = self.conv21(torch.cat([x20, self.up(x30)], dim=1))

        # Decoder level 2
        x02 = self.conv02(torch.cat([x00, x01, self.up(x11)], dim=1))
        x12 = self.conv12(torch.cat([x10, x11, self.up(x21)], dim=1))

        # Decoder level 3
        x03 = self.conv03(torch.cat([x00, x01, x02, self.up(x12)], dim=1))

        output = self.final(x03)

        return output
