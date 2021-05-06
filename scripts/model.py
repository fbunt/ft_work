import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast


DEFAULT_KERNEL_SIZE = 3
LEAKY_SLOPE = 0.1
P_BOUNDARY_DROP = 0.2


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
    """A block of multiple convolutions with a skip connection around it."""

    def __init__(self, in_chan, out_chan, n=2):
        super().__init__()
        if n < 1:
            raise ValueError(
                "Expected number of convolution layers to be greater than 1 "
                f" (got {n})"
            )
        self.conv_block = _MultiConv(in_chan, out_chan, n=n)
        self.skip = nn.Conv2d(in_chan, out_chan, kernel_size=1)
        self.activation = nn.LeakyReLU(LEAKY_SLOPE, inplace=True)

    def forward(self, x):
        xskip = self.skip(x)
        x = self.conv_block(x)
        x = x + xskip
        return self.activation(x)


def _passthrough(x):
    return x


class _MultiConvBlock(nn.Module):
    """A block of multiple conv blocks. Optional skip connection and 2D
    dropout.
    """

    def __init__(
        self,
        in_chan,
        out_chan,
        n=2,
        skip=False,
        dropout=False,
        dropout_p=P_BOUNDARY_DROP,
    ):
        """
        Parameters
        ----------
        in_chan : int
            number of input channels
        out_chan : int
            number of output channels
        n : int
            number of convolutions in convolution block. Default is 2.
        skip : bool
            Whether to add skip connection around conv block. Default is False.
        dropout : bool
            Whether to use 2D dropout at output. Default is False.
        dropout_p : float
            Dropout probability. Default is 0.2.
        """
        super().__init__()
        if n < 1:
            raise ValueError(
                "Expected number of convolution layers to be greater than 1 "
                f" (got {n})"
            )
        conv_class = _MultiConvSkip if skip else _MultiConv
        self.conv_block = conv_class(in_chan, out_chan, n=n)
        self.out = nn.Dropout2d(p=dropout_p) if dropout else _passthrough

    def forward(self, x):
        x = self.conv_block(x)
        return self.out(x)


class _DownSample(nn.Module):
    """Down sample operation using max pooling"""

    def __init__(self, size=2):
        super().__init__()
        self.model = nn.MaxPool2d(size)

    def forward(self, x):
        return self.model(x)


class _Down(nn.Module):
    """
    A downsample followed by a convolution block with optional skip connection
    and 2D dropout
    """

    def __init__(
        self,
        in_chan,
        out_chan,
        n=2,
        skip=False,
        dropout=False,
        dropout_p=P_BOUNDARY_DROP,
    ):
        """
        Parameters
        ----------
        in_chan : int
            number of input channels
        out_chan : int
            number of output channels
        n : int
            number of convolutions in convolution block. Default is 2.
        skip : bool
            Whether to add skip connection around conv block. Default is False.
        dropout : bool
            Whether to use 2D dropout at output. Default is False.
        dropout_p : float
            Dropout probability. Default is 0.2.
        """
        super().__init__()
        blocks = [
            _DownSample(),
            _MultiConvBlock(in_chan, out_chan, n=n, skip=skip),
        ]
        if dropout:
            blocks.append(nn.Dropout2d(p=dropout_p))
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)


class _UpSample(nn.Module):
    """Upsample using transposed convolution"""

    def __init__(self, in_chan):
        super().__init__()
        self.model = nn.ConvTranspose2d(
            in_chan, in_chan // 2, kernel_size=2, stride=2
        )

    def forward(self, x):
        return self.model(x)


class _Up(nn.Module):
    """
    Upscale followed by a convolution block with optional skip connection and
    2D dropout.
    """

    def __init__(
        self,
        in_chan,
        out_chan,
        n=2,
        skip=False,
        dropout=False,
        dropout_p=P_BOUNDARY_DROP,
    ):
        """
        Parameters
        ----------
        in_chan : int
            number of input channels
        out_chan : int
            number of output channels
        n : int
            number of convolutions in convolution block. Default is 2.
        skip : bool
            Whether to add skip connection around conv block. Default is False.
        dropout : bool
            Whether to use 2D dropout at output. Default is False.
        dropout_p : float
            Dropout probability. Default is 0.2.
        """
        super().__init__()
        self.upsample = _UpSample(in_chan)
        self.conv = _MultiConvBlock(in_chan, out_chan, n=n, skip=skip)
        self.out = nn.Dropout2d(p=dropout_p) if dropout else _passthrough

    def forward(self, lhs, bot):
        upbot = self.upsample(bot)
        dh = lhs.size(2) - upbot.size(2)
        dw = lhs.size(3) - upbot.size(3)
        upbot = F.pad(
            upbot, (dw // 2, dw - (dw // 2), dh // 2, dh - (dh // 2))
        )
        x = torch.cat((lhs, upbot), dim=1)
        x = self.conv(x)
        return self.out(x)


LABEL_FROZEN = 0
LABEL_THAWED = 1
LABEL_OTHER = 2


class UNet(nn.Module):
    """A depth-generalized UNet implementation

    ref: https://arxiv.org/abs/1505.04597
    """

    def __init__(
        self,
        in_chan,
        n_classes,
        depth=4,
        base_filter_bank_size=16,
        skip=False,
        bndry_dropout=False,
        bndry_dropout_p=P_BOUNDARY_DROP,
    ):
        """
        Parameters
        ----------
        in_chan : int
            number of input channels
        n_classes : int
            number of output channels. One per class.
        depth : int
            Number of downscales and corresponding upscales. Default is 4.
        base_filter_bank_size : int
            Number of filters at input level. This is multiplied by a factor of
            2 at every downscale. Default is 16
        skip : bool
            Whether to add skip connection around conv block. Default is False.
        bndry_dropout : bool
            Whether to use 2D dropout at every block output. Default is False.
        bndry_dropout_p : float
            Dropout probability. Default is 0.2.
        """
        # Note: in the original paper, base_filter_bank_size is 64
        super().__init__()

        if depth < 0:
            raise ValueError(f"UNet depth must be at least 0: {depth}")
        nb = base_filter_bank_size
        if nb <= 0:
            raise ValueError(f"Filter bank size must be greater than 0: {nb}")

        self.depth = depth
        self.input = _MultiConvBlock(
            in_chan,
            nb,
            skip=skip,
            dropout=bndry_dropout,
            dropout_p=bndry_dropout_p,
        )
        self.downs = nn.ModuleList()
        for i in range(0, depth):
            self.downs.append(
                _Down(
                    (2 ** i) * nb,
                    (2 ** (i + 1)) * nb,
                    skip=skip,
                    dropout=bndry_dropout,
                    dropout_p=bndry_dropout_p,
                )
            )
        self.ups = nn.ModuleList()
        for i in range(depth, 0, -1):
            self.ups.append(
                _Up(
                    (2 ** i) * nb,
                    (2 ** (i - 1)) * nb,
                    skip=skip,
                    dropout=bndry_dropout,
                )
            )
        self.out = nn.Conv2d(nb, n_classes, kernel_size=1)

    @autocast()
    def forward(self, x):
        xdowns = [self.input(x)]
        for down in self.downs:
            xdowns.append(down(xdowns[-1]))
        x = xdowns[-1]
        for xleft, up in zip(xdowns[:-1][::-1], self.ups):
            x = up(xleft, x)
        return self.out(x)


class UNetDepth4(nn.Module):
    def __init__(
        self, in_chan, n_classes, base_filter_bank_size=16, skip=False
    ):
        super().__init__()

        nb = base_filter_bank_size
        if nb <= 0:
            raise ValueError(f"Filter bank size must be greater than 0: {nb}")

        self.input = _MultiConvBlock(in_chan, nb, skip=skip)
        self.downs = nn.ModuleList()
        self.downs.append(_Down(1 * nb, 2 * nb, skip=skip))
        self.downs.append(_Down(2 * nb, 4 * nb, skip=skip))
        self.downs.append(_Down(4 * nb, 8 * nb, skip=skip))
        self.downs.append(_Down(8 * nb, 16 * nb, skip=skip))
        self.ups = nn.ModuleList()
        self.ups.append(_Up(16 * nb, 8 * nb, skip=skip))
        self.ups.append(_Up(8 * nb, 4 * nb, skip=skip))
        self.ups.append(_Up(4 * nb, 2 * nb, skip=skip))
        self.ups.append(_Up(2 * nb, 1 * nb, skip=skip))
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


DEFAULT_KERNEL_SIZE_LEGACY = 3
LEAKY_SLOPE_LEGACY = 0.1


class _ConvLayerLegacy(nn.Module):
    def __init__(self, in_chan, out_chan):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(
                in_chan,
                out_chan,
                kernel_size=DEFAULT_KERNEL_SIZE_LEGACY,
                padding=1,
            ),
            nn.BatchNorm2d(out_chan),
            nn.LeakyReLU(LEAKY_SLOPE_LEGACY, inplace=True),
        )

    def forward(self, x):
        return self.model(x)


class _MultiConvLegacy(nn.Module):
    def __init__(self, in_chan, out_chan, n=2):
        super().__init__()
        if n < 1:
            raise ValueError(
                "Expected number of convolution layers to be greater than 1 "
                f" (got {n})"
            )
        self.layers = nn.ModuleList([_ConvLayerLegacy(in_chan, out_chan)])
        for _ in range(n - 1):
            self.layers.append(_ConvLayerLegacy(out_chan, out_chan))

    def forward(self, x):
        for m in self.layers:
            x = m(x)
        return x


class _DownSampleLegacy(nn.Module):
    def __init__(self, size=2):
        super().__init__()
        self.model = nn.MaxPool2d(size)

    def forward(self, x):
        return self.model(x)


class _DownLegacy(nn.Module):
    def __init__(self, in_chan, out_chan):
        super().__init__()
        self.model = nn.Sequential(
            _DownSampleLegacy(), _MultiConvLegacy(in_chan, out_chan)
        )

    def forward(self, x):
        return self.model(x)


class _UpSampleLegacy(nn.Module):
    def __init__(self, in_chan):
        super().__init__()
        self.model = nn.ConvTranspose2d(
            in_chan, in_chan // 2, kernel_size=2, stride=2
        )

    def forward(self, x):
        return self.model(x)


class _UpLegacy(nn.Module):
    def __init__(self, in_chan, out_chan):
        super().__init__()
        self.upsample = _UpSampleLegacy(in_chan)
        self.conv = _MultiConvLegacy(in_chan, out_chan)

    def forward(self, lhs, bot):
        upbot = self.upsample(bot)
        dh = lhs.size(2) - upbot.size(2)
        dw = lhs.size(3) - upbot.size(3)
        upbot = F.pad(
            upbot, (dw // 2, dw - (dw // 2), dh // 2, dh - (dh // 2))
        )
        x = torch.cat((lhs, upbot), dim=1)
        return self.conv(x)


class UNetLegacy(nn.Module):
    """A depth-generalized UNet implementation for use with legacy runs.

    ref: https://arxiv.org/abs/1505.04597
    """

    def __init__(
        self,
        in_chan,
        n_classes,
        depth=4,
        base_filter_bank_size=16,
        *args,
        **kwargs,
    ):
        # Note: in the original paper, base_filter_bank_size is 64
        super().__init__()

        if depth < 0:
            raise ValueError(f"UNet depth must be at least 0: {depth}")
        nb = base_filter_bank_size
        if nb <= 0:
            raise ValueError(f"Filter bank size must be greater than 0: {nb}")

        self.depth = depth
        self.input = _MultiConvLegacy(in_chan, nb)
        self.downs = nn.ModuleList()
        for i in range(0, depth):
            self.downs.append(_DownLegacy((2 ** i) * nb, (2 ** (i + 1)) * nb))
        self.ups = nn.ModuleList()
        for i in range(depth, 0, -1):
            self.ups.append(_UpLegacy((2 ** i) * nb, (2 ** (i - 1)) * nb))
        self.out = nn.Conv2d(nb, n_classes, kernel_size=1)

    def forward(self, x):
        xdowns = [self.input(x)]
        for down in self.downs:
            xdowns.append(down(xdowns[-1]))
        x = xdowns[-1]
        for xleft, up in zip(xdowns[:-1][::-1], self.ups):
            x = up(xleft, x)
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
