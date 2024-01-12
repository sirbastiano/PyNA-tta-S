import torch.nn as nn


class MaxPool(nn.Sequential):
    def __init__(self, kernel_size=2, stride=None, padding=0):
        if stride is None:
            stride = kernel_size
        super(MaxPool, self).__init__(
            nn.MaxPool2d(kernel_size, stride, padding)
        )
