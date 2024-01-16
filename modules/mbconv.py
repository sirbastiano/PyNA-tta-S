import torch
from torch import nn
from torch import Tensor
import modules.relu
from typing import Optional


# Mine:
class MBConv(nn.Module):
    def __init__(self, in_channels, out_channels, expansion_factor=4, activation=modules.relu.ReLU):
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


class MBConv_no_res(nn.Sequential):
    def __init__(self, in_channels, out_channels, expansion_factor, kernel_size=3, stride=1, activation=modules.relu.ReLU):
        expanded_channels = in_channels * expansion_factor
        super(MBConv_no_res, self).__init__(
            # Expansion layer
            nn.Conv2d(in_channels, expanded_channels, kernel_size=1, stride=1, padding=1),
            nn.BatchNorm2d(expanded_channels),
            activation(),
            # Depthwise convolution
            nn.Conv2d(expanded_channels, expanded_channels, kernel_size, stride, kernel_size//2, groups=expanded_channels),
            nn.BatchNorm2d(expanded_channels),
            activation(),
            # Projection layer
            nn.Conv2d(expanded_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            activation(),
        )


class ResidualAdd(nn.Module):
    def __init__(self, block: nn.Module, shortcut: Optional[nn.Module] = None):
        super().__init__()
        self.block = block
        self.shortcut = shortcut

    def forward(self, x: Tensor) -> Tensor:
        res = x
        x = self.block(x)
        if self.shortcut:
            res = self.shortcut(res)
        x += res
        return x


# MobileNetLikeBlock
# from https://github.com/FrancescoSaverioZuppichini/BottleNeck-InvertedResidual-FusedMBConv-in-PyTorch
class MBConv_MobileNetLike(nn.Sequential):
    def __init__(self, in_channels, out_channels, expansion_factor=4, kernel_size=3, stride=1, activation=modules.relu.ReLU):
        # use ResidualAdd if features match, otherwise a normal Sequential
        residual = ResidualAdd if in_channels == out_channels else nn.Sequential
        expanded_channels = in_channels * expansion_factor
        super().__init__(
            nn.Sequential(
                residual(
                    nn.Sequential(
                        # narrow -> wide
                        nn.Conv2d(in_channels, expanded_channels, kernel_size=1, padding=0),
                        nn.BatchNorm2d(expanded_channels),
                        activation(),
                        # wide -> wide
                        nn.Conv2d(expanded_channels, expanded_channels, kernel_size=kernel_size, padding=kernel_size//2, groups=expanded_channels),
                        nn.BatchNorm2d(expanded_channels),
                        activation(),
                        # wide -> narrow
                        nn.Conv2d(expanded_channels, out_channels, kernel_size=1, padding=0),
                        nn.BatchNorm2d(out_channels),
                        activation(),
                    ),
                ),
                #nn.ReLU(),
            )
        )