import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, k1=3, s1=1, p1=1, k2=3, s2=1, p2=1):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=k1, stride=s1, padding=p1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=k2, stride=s2, padding=p2),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool1d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, k=3):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=k, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64, 25, 5, 0, 5, 1, 0)
        self.down1 = Down(64, 128)
        self.down2 = Down(128,256)
        self.up1 = Up(256, 128, bilinear)
        self.up2 = Up(128, 64, bilinear, k=2)
        self.outc = OutConv(64, n_classes)

        self.mlp = nn.Sequential(
            nn.Linear(in_features=242, out_features=128),
            nn.ReLU(True),
            nn.Dropout(0.1),
            nn.Linear(in_features=128, out_features=48),
            nn.ReLU(True),
        )
        self.linear = nn.Linear(48, 1)

    def forward(self, x):
        x1 = self.inc(x)  # 242
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x = self.up1(x3, x2)  # 121
        x = self.up2(x, x1)
        x = self.outc(x).reshape(-1, 242)
        out1 = self.mlp(x)
        out = self.linear(out1)
        return out


if __name__ == '__main__':
    inp = torch.randn(32, 2, 1250)
    net = UNet(n_channels=2, n_classes=1)
    out = net(inp)
    print(out)
