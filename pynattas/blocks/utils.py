import torch.nn as nn


class Dropout(nn.Sequential):
    """
    Dropout Module.
    This module implements a dropout layer, which is a widely used regularization
    technique in neural networks. Dropout helps to prevent overfitting by randomly
    setting a fraction of input units to zero at each update during training time,
    which is controlled by the probability `p`.

    Args:
        p (float, optional): Probability of an element to be zeroed. Defaults to 0.4.

    The module wraps the `nn.Dropout` function in PyTorch.
    """
    def __init__(self, p=0.4):
        super(Dropout, self).__init__(
            nn.Dropout(p)
        )


class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        reduced_channels = max(in_channels // reduction, 1)  # Ensure at least 1 output feature
        self.fc = nn.Sequential(
            nn.Linear(in_channels, reduced_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

