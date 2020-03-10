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
    """A depth-generalized UNet implementation

    ref: https://arxiv.org/abs/1505.04597
    """

    def __init__(self, in_chan, n_classes, depth=4, base_filter_bank_size=16):
        # Note: in the original paper, base_filter_bank_size is 64
        super().__init__()

        if depth < 0:
            raise ValueError(f"UNet depth must be at least 0: {depth}")
        nb = base_filter_bank_size
        if nb <= 0:
            raise ValueError(f"Filter bank size must be greater than 0: {nb}")
        self.input = _DoubleConv(in_chan, nb)
        self.downs = nn.ModuleList()
        for i in range(0, depth):
            self.downs.append(_Down((2 ** i) * nb, (2 ** (i + 1)) * nb))
        self.ups = nn.ModuleList()
        for i in range(depth, 0, -1):
            self.ups.append(_Up((2 ** i) * nb, (2 ** (i - 1)) * nb))
        self.out = nn.Conv2d(nb, n_classes, kernel_size=1)

    def forward(self, x):
        xdowns = [self.input(x)]
        for down in self.downs:
            xdowns.append(down(xdowns[-1]))
        x = xdowns[-1]
        for xleft, up in zip(xdowns[:-1][::-1], self.ups):
            x = up(xleft, x)
        return self.out(x)
