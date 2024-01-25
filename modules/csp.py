import torch
import torch.nn as nn
import modules.gelu
import modules.mbconv


# Note: Currently implemented is actually a CSP-ized MBConv block.
# TODO: Make a parent CSPBlock for various CSP-ized blocks. Change the name of the classes to more appropriate ones.
class CSPBlock(nn.Module):
    def __init__(self, in_channels, num_blocks=1, expansion_factor=4, activation=modules.gelu.GELU):
        super(CSPBlock, self).__init__()

        # Use the same value for hidden_channels to avoid the issue with odd in_channels
        self.main_channels = in_channels // 2
        self.shortcut_channels = in_channels-self.main_channels

        # Main path (processed part)
        self.main_path = nn.Sequential(
            *[modules.mbconv.MBConv(
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
        combined = torch.cat((main_data, shortcut_data), dim=1)

        # Apply final transition
        out = self.final_transition(combined)
        return out
