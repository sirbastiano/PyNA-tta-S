import torch
import torch.nn as nn
from .activations import ReLU
from .utils import SEBlock


# Classic Conv
class ConvAct(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, activation=ReLU):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            activation(),
        )


class ConvBnAct_old(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, activation=ReLU):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(num_features=out_channels),
            activation(),
        )


class ConvBnAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None, activation=ReLU):
        super(ConvBnAct, self).__init__()
        if padding == None:
            padding=kernel_size//2 # To maintain same convolution as a default 
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = activation()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class ConvSE(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, activation=ReLU):
        super().__init__(
            ConvBnAct(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, activation=activation),
            SEBlock(reduction=16, in_channels=out_channels),
        )

# class DepthwiseConvAct
# class DepthwiseConvBnAct


# MBConv Inverted
class MBConv(nn.Module):
    def __init__(self, in_channels, out_channels, dw_kernel_size=3, expansion_factor=4, activation=ReLU):
        expanded_channels = in_channels * expansion_factor
        super().__init__()
        self.steps = nn.Sequential(
            # Narrow to wide
            ConvBnAct(in_channels, expanded_channels, kernel_size=1, stride=1, padding=0, activation=activation),
            # Wide to wide (depthwise convolution)
            nn.Conv2d(expanded_channels, expanded_channels, kernel_size=dw_kernel_size, stride=1, padding=1, groups=expanded_channels),
            nn.BatchNorm2d(expanded_channels),
            activation(),
            # Wide to narrow
            ConvBnAct(expanded_channels, out_channels, kernel_size=1, stride=1, padding=0, activation=activation)
        )

    def forward(self, x):
        res = x
        x = self.steps(x)
        x = torch.add(x, res)
        return x
    

class MBConvNoRes(nn.Module):
    def __init__(self, in_channels, out_channels, dw_kernel_size=3, expansion_factor=4, activation=ReLU):
        expanded_channels = in_channels * expansion_factor
        super().__init__()
        self.steps = nn.Sequential(
            # Narrow to wide
            ConvBnAct(in_channels, expanded_channels, kernel_size=1, stride=1, padding=0, activation=activation),
            # Wide to wide (depthwise convolution)
            nn.Conv2d(expanded_channels, expanded_channels, kernel_size=dw_kernel_size, stride=1, padding=1, groups=expanded_channels),
            nn.BatchNorm2d(expanded_channels),
            activation(),
            # Wide to narrow
            ConvBnAct(expanded_channels, out_channels, kernel_size=1, stride=1, padding=0, activation=activation)
        )

    def forward(self, x):
        x = self.steps(x)
        return x


# CSP Conv
class CSPConvBlock(nn.Module):
    def __init__(self, in_channels, num_blocks=1, activation=ReLU):
        super().__init__()

        # Use the same value for hidden_channels to avoid the issue with odd in_channels
        self.main_channels = in_channels // 2
        self.shortcut_channels = in_channels-self.main_channels

        # Main path (processed part)
        self.main_path = nn.Sequential(
            *[ConvBnAct(
                in_channels=self.main_channels,
                out_channels=self.main_channels,
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

        main_data = self.main_path(main_data)
        shortcut_data = self.shortcut_path(shortcut_data)

        # Concatenating the main and shortcut paths
        combined = torch.cat(tensors=(main_data, shortcut_data), dim=1)
        out = self.final_transition(combined)
        return out


class CSPMBConvBlock(nn.Module):
    def __init__(self, in_channels, num_blocks=1, dw_kernel_size=3, expansion_factor=4, activation=ReLU):
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
                dw_kernel_size=dw_kernel_size,
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

        main_data = self.main_path(main_data)
        shortcut_data = self.shortcut_path(shortcut_data)

        # Concatenating the main and shortcut paths
        combined = torch.cat(tensors=(main_data, shortcut_data), dim=1)
        out = self.final_transition(combined)
        return out


# DenseNetBlock
class DenseNetBlock(nn.Module):
    """
    Basic DenseNet block composed by one 3x3 convs with residual connection.
    The residual connection is perfomed by concatenate the input and the output.
    """
    def __init__(self, in_channels, out_channels, activation=ReLU):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            activation(),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
            )

    def forward(self, x):
        res = x
        x = self.block(x)
        return torch.cat([res, x], dim=1)


# ResNetBlock and derivatives (https://paperswithcode.com/method/resnext)
class ResNetBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, reduction_factor=4, activation=ReLU):
        super().__init__()
        assert out_channels == in_channels
        reduced_channels = in_channels // reduction_factor

        if reduced_channels == 0:
            reduced_channels = in_channels
            
        self.steps = nn.Sequential(
            # Narrow to wide
            ConvBnAct(in_channels, reduced_channels, kernel_size=1, stride=1, padding=0, activation=activation),
            # Wide to wide (depthwise convolution)
            ConvBnAct(reduced_channels, reduced_channels, kernel_size=3, stride=1, padding=1, activation=activation),
            # Wide to narrow
            ConvBnAct(reduced_channels, out_channels, kernel_size=1, stride=1, padding=0, activation=activation)
        )

    def forward(self, x):
        x = self.steps(x)
        return x
    

class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, reduction_factor=4, activation=ReLU):
        super().__init__()
        assert out_channels == in_channels
        self.main_path = ResNetBasicBlock(
            in_channels=in_channels, 
            out_channels=out_channels,
            reduction_factor=reduction_factor,
            activation=activation,
            )

    def forward(self, x):
        res = x
        x = self.main_path(x)
        x = torch.add(x, res)
        return x
    

# Recupera
class ResNetBlockNB(nn.Module):
    def __init__(self, in_channels, out_channels, reduction_factor=4, activation=ReLU, num_block=1):
        super(ResNetBlock, self).__init__()
        assert out_channels == in_channels
        self.main_path = nn.Sequential(
            *[ResNetBasicBlock(in_channels=in_channels, 
                               out_channels=out_channels, 
                               reduction_factor=reduction_factor, 
                               activation=activation) for _ in range(num_block)]
        )

    def forward(self, x):
        res = x
        x = self.main_path(x)
        x = torch.add(x, res)
        return x