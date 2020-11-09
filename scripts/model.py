import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_KERNEL_SIZE = 3
LEAKY_SLOPE = 0.1


class _ConvLayer(nn.Module):
    def __init__(self, in_chan, out_chan):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(
                in_chan, out_chan, kernel_size=DEFAULT_KERNEL_SIZE, padding=1
            ),
            nn.BatchNorm2d(out_chan),
            nn.LeakyReLU(LEAKY_SLOPE, inplace=True),
        )

    def forward(self, x):
        return self.model(x)


class _MultiConv(nn.Module):
    def __init__(self, in_chan, out_chan, n=2):
        super().__init__()
        if n < 1:
            raise ValueError(
                "Expected number of convolution layers to be greater than 1 "
                f" (got {n})"
            )
        self.layers = nn.ModuleList([_ConvLayer(in_chan, out_chan)])
        for _ in range(n - 1):
            self.layers.append(_ConvLayer(out_chan, out_chan))

    def forward(self, x):
        for m in self.layers:
            x = m(x)
        return x


class _MultiConvSkip(nn.Module):
    """A block of multiple conv blocks with a skip connection around it."""

    def __init__(self, in_chan, out_chan, n=2):
        super().__init__()
        if n < 1:
            raise ValueError(
                "Expected number of convolution layers to be greater than 1 "
                f" (got {n})"
            )
        self.layers = nn.ModuleList([_ConvLayer(in_chan, out_chan)])
        for _ in range(n - 1):
            self.layers.append(_ConvLayer(out_chan, out_chan))
        self.skip = nn.Conv2d(in_chan, out_chan, kernel_size=1)
        self.activation = nn.LeakyReLU(LEAKY_SLOPE, inplace=True)

    def forward(self, x):
        xskip = self.skip(x)
        for m in self.layers:
            x = m(x)
        x = x + xskip
        return self.activation(x)


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
            _DownSample(), _MultiConv(in_chan, out_chan)
        )

    def forward(self, x):
        return self.model(x)


class _DownSkip(nn.Module):
    def __init__(self, in_chan, out_chan):
        super().__init__()
        self.model = nn.Sequential(
            _DownSample(), _MultiConvSkip(in_chan, out_chan)
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
        self.conv = _MultiConv(in_chan, out_chan)

    def forward(self, lhs, bot):
        upbot = self.upsample(bot)
        dh = lhs.size(2) - upbot.size(2)
        dw = lhs.size(3) - upbot.size(3)
        upbot = F.pad(
            upbot, (dw // 2, dw - (dw // 2), dh // 2, dh - (dh // 2))
        )
        x = torch.cat((lhs, upbot), dim=1)
        return self.conv(x)


class _UpSkip(nn.Module):
    def __init__(self, in_chan, out_chan):
        super().__init__()
        self.upsample = _UpSample(in_chan)
        self.conv = _MultiConvSkip(in_chan, out_chan)

    def forward(self, lhs, bot):
        upbot = self.upsample(bot)
        dh = lhs.size(2) - upbot.size(2)
        dw = lhs.size(3) - upbot.size(3)
        upbot = F.pad(
            upbot, (dw // 2, dw - (dw // 2), dh // 2, dh - (dh // 2))
        )
        x = torch.cat((lhs, upbot), dim=1)
        return self.conv(x)


LABEL_FROZEN = 0
LABEL_THAWED = 1
LABEL_OTHER = 2


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

        self.depth = depth
        self.input = _MultiConv(in_chan, nb)
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


class UNetSkip(nn.Module):
    """A depth-generalized UNet implementation with skip connections around the
    convolution blocks.

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

        self.depth = depth
        self.input = _MultiConv(in_chan, nb)
        self.downs = nn.ModuleList()
        for i in range(0, depth):
            self.downs.append(_DownSkip((2 ** i) * nb, (2 ** (i + 1)) * nb))
        self.ups = nn.ModuleList()
        for i in range(depth, 0, -1):
            self.ups.append(_UpSkip((2 ** i) * nb, (2 ** (i - 1)) * nb))
        self.out = nn.Conv2d(nb, n_classes, kernel_size=1)

    def forward(self, x):
        xdowns = [self.input(x)]
        for down in self.downs:
            xdowns.append(down(xdowns[-1]))
        x = xdowns[-1]
        for xleft, up in zip(xdowns[:-1][::-1], self.ups):
            x = up(xleft, x)
        return self.out(x)


class UNetDepth4(nn.Module):
    def __init__(self, in_chan, n_classes, base_filter_bank_size=16):
        super().__init__()

        nb = base_filter_bank_size
        if nb <= 0:
            raise ValueError(f"Filter bank size must be greater than 0: {nb}")

        self.input = _MultiConv(in_chan, nb)
        self.downs = nn.ModuleList()
        self.downs.append(_Down(1 * nb, 2 * nb))
        self.downs.append(_Down(2 * nb, 4 * nb))
        self.downs.append(_Down(4 * nb, 8 * nb))
        self.downs.append(_Down(8 * nb, 16 * nb))
        self.ups = nn.ModuleList()
        self.ups.append(_Up(16 * nb, 8 * nb))
        self.ups.append(_Up(8 * nb, 4 * nb))
        self.ups.append(_Up(4 * nb, 2 * nb))
        self.ups.append(_Up(2 * nb, 1 * nb))
        self.out = nn.Conv2d(nb, n_classes, kernel_size=1)

    def forward(self, x):
        xdowns = [self.input(x)]
        xdowns.append(self.downs[0](xdowns[-1]))
        xdowns.append(self.downs[1](xdowns[-1]))
        xdowns.append(self.downs[2](xdowns[-1]))
        xdowns.append(self.downs[3](xdowns[-1]))
        x = xdowns[4]
        x = self.ups[0](xdowns[3], x)
        x = self.ups[1](xdowns[2], x)
        x = self.ups[2](xdowns[1], x)
        x = self.ups[3](xdowns[0], x)
        return self.out(x)


class UNetSkipDepth4(nn.Module):
    def __init__(self, in_chan, n_classes, base_filter_bank_size=16):
        super().__init__()

        nb = base_filter_bank_size
        if nb <= 0:
            raise ValueError(f"Filter bank size must be greater than 0: {nb}")

        self.input = _MultiConv(in_chan, nb)
        self.downs = nn.ModuleList()
        self.downs.append(_DownSkip(1 * nb, 2 * nb))
        self.downs.append(_DownSkip(2 * nb, 4 * nb))
        self.downs.append(_DownSkip(4 * nb, 8 * nb))
        self.downs.append(_DownSkip(8 * nb, 16 * nb))
        self.ups = nn.ModuleList()
        self.ups.append(_UpSkip(16 * nb, 8 * nb))
        self.ups.append(_UpSkip(8 * nb, 4 * nb))
        self.ups.append(_UpSkip(4 * nb, 2 * nb))
        self.ups.append(_UpSkip(2 * nb, 1 * nb))
        self.out = nn.Conv2d(nb, n_classes, kernel_size=1)

    def forward(self, x):
        xdowns = [self.input(x)]
        xdowns.append(self.downs[0](xdowns[-1]))
        xdowns.append(self.downs[1](xdowns[-1]))
        xdowns.append(self.downs[2](xdowns[-1]))
        xdowns.append(self.downs[3](xdowns[-1]))
        x = xdowns[4]
        x = self.ups[0](xdowns[3], x)
        x = self.ups[1](xdowns[2], x)
        x = self.ups[2](xdowns[1], x)
        x = self.ups[3](xdowns[0], x)
        return self.out(x)


def local_variation_loss(data, loss_func=nn.L1Loss()):
    """Compute the local variation around each pixel.

    This loss discourages high frequency noise on borders.
    """
    # Compute vertical variation
    loss = loss_func(data[..., 1:, :], data[..., :-1, :])
    # Compute horizontal variation
    loss += loss_func(data[..., :, 1:], data[..., :, :-1])
    return loss


def ft_loss(prediction, label, distance):
    loss = (prediction - label) ** 2 / (torch.log(distance) + 1)
    return torch.mean(loss)
