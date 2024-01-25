import torch.nn as nn
import modules.relu


class Conv2D(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, activation=modules.relu.ReLU):
        super(Conv2D, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            activation(),
        )
