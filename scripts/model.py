import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_KERNEL_SIZE = 3
LEAKY_SLOPE = 0.1


class _DoubleConv(nn.Module):
    def __init__(self, in_chan, out_chan):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(
                in_chan, out_chan, kernel_size=DEFAULT_KERNEL_SIZE, padding=1
            ),
            nn.BatchNorm2d(out_chan),
            nn.LeakyReLU(LEAKY_SLOPE, inplace=True),
            nn.Conv2d(
                out_chan, out_chan, kernel_size=DEFAULT_KERNEL_SIZE, padding=1
            ),
            nn.BatchNorm2d(out_chan),
            nn.LeakyReLU(LEAKY_SLOPE, inplace=True),
        )

    def forward(self, x):
        return self.model(x)


class _DownSample(nn.Module):
    def __init__(self, size=2):
        super().__init__()
        self.model = nn.MaxPool2d(size)

    def forward(self, x):
        return self.model(x)


class _Down(nn.Module):
    def __init__(self, in_chan, out_chan):
        super().__init__()
        self.model = nn.Sequential(
            _DownSample(), _DoubleConv(in_chan, out_chan)
        )

    def forward(self, x):
        return self.model(x)


class _UpSample(nn.Module):
    def __init__(self, in_chan):
        super().__init__()
        self.model = nn.ConvTranspose2d(
            in_chan, in_chan // 2, kernel_size=2, stride=2
        )

    def forward(self, x):
        return self.model(x)


class _Up(nn.Module):
    def __init__(self, in_chan, out_chan):
        super().__init__()
        self.upsample = _UpSample(in_chan)
        self.conv = _DoubleConv(in_chan, out_chan)

    def forward(self, lhs, bot):
        upbot = self.upsample(bot)
        dh = lhs.size(2) - upbot.size(2)
        dw = lhs.size(3) - upbot.size(3)
        upbot = F.pad(
            upbot, (dw // 2, dw - (dw // 2), dh // 2, dh - (dh // 2))
        )
        x = torch.cat((lhs, upbot), dim=1)
        return self.conv(x)


class UNet(nn.Module):
    # Note: in the original paper, base_filter_bank_size is 64
    def __init__(self, in_chan, n_classes, base_filter_bank_size=16):
        super().__init__()

        n = base_filter_bank_size
        self.input = _DoubleConv(in_chan, n)
        self.down1 = _Down(n, 2 * n)
        self.down2 = _Down(2 * n, 4 * n)
        self.down3 = _Down(4 * n, 8 * n)
        self.down4 = _Down(8 * n, 16 * n)
        self.up1 = _Up(16 * n, 8 * n)
        self.up2 = _Up(8 * n, 4 * n)
        self.up3 = _Up(4 * n, 2 * n)
        self.up4 = _Up(2 * n, n)
        self.out = nn.Conv2d(n, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.input(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.down4(x4)
        x = self.up1(x4, x)
        x = self.up2(x3, x)
        x = self.up3(x2, x)
        x = self.up4(x1, x)
        x = self.out(x)
        return x
