import torch
import torch.nn as nn
from .activations import GELU


class Conv2D(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, activation=GELU):
        super(Conv2D, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            activation(),
        )


class MBConv(nn.Module):
    def __init__(self, in_channels, out_channels, expansion_factor=4, activation=GELU):
        expanded_channels = in_channels * expansion_factor
        super(MBConv, self).__init__()
        self.steps = nn.Sequential(
            # Narrow to wide
            nn.Conv2d(in_channels, expanded_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(expanded_channels),
            activation(),
            # Wide to wide (depthwise convolution)
            nn.Conv2d(expanded_channels, expanded_channels, kernel_size=3, stride=1, padding=1, groups=expanded_channels),
            nn.BatchNorm2d(expanded_channels),
            activation(),
            # Wide to narrow
            nn.Conv2d(expanded_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            activation(),
        )

    def forward(self, x):
        res = x
        x = self.steps(x)
        x = torch.add(x, res)
        return x


# Note: Currently implemented is actually a CSP-ized MBConv block.
# TODO: Make a parent CSPBlock for various CSP-ized blocks. Change the name of the classes to more appropriate ones.
class CSPBlock(nn.Module):
    def __init__(self, in_channels, num_blocks=1, expansion_factor=4, activation=GELU):
        super().__init__()

        # Use the same value for hidden_channels to avoid the issue with odd in_channels
        self.main_channels = in_channels // 2
        self.shortcut_channels = in_channels-self.main_channels

        # Main path (processed part)
        self.main_path = nn.Sequential(
            *[MBConv(
                in_channels=self.main_channels,
                out_channels=self.main_channels,
                expansion_factor=expansion_factor,
                activation=activation,
            ) for _ in range(num_blocks)],
        )

        # Shortcut path is just a passthrough
        self.shortcut_path = nn.Identity()

        # Final 1x1 convolution after merging
        self.final_transition = nn.Sequential(
            nn.Conv2d(
                in_channels=self.main_channels+self.shortcut_channels,
                out_channels=in_channels,
                kernel_size=1,
            ),
            nn.BatchNorm2d(in_channels),
            activation(),
        )

    def forward(self, x):
        # Apply first transition which is just a passthrough here
        #shortcut = nn.Identity(x)

        # Splitting the input tensor into two paths
        main_data = x[:, :self.main_channels, :, :]
        shortcut_data = x[:, self.main_channels:, :, :]

        # Process the main path
        main_data = self.main_path(main_data)

        # Shortcut path
        shortcut_data = self.shortcut_path(shortcut_data)

        # Concatenating the main and shortcut paths
        combined = torch.cat(tensors=(main_data, shortcut_data), dim=1)

        # Apply final transition
        out = self.final_transition(combined)
        return out


class DenseNetBlock(nn.Module):
    """
    Basic DenseNet block composed by one 3x3 convs with residual connection.
    The residual connection is perfomed by concatenate the input and the output.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, activation=GELU):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            activation(),
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
            )

    def forward(self, x):
        res = x
        x = self.block(x)
        return torch.cat([res, x], dim=1)
