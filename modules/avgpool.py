import torch.nn as nn


class AvgPool(nn.Sequential):
    def __init__(self, kernel_size=2, stride=2):
        super(AvgPool, self).__init__(
            nn.AvgPool2d(kernel_size, stride)
        )
