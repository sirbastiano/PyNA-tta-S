import torch.nn as nn


class Dropout(nn.Sequential):
    """Dropout Module.

    This module implements a dropout layer, which is a widely used regularization technique in neural networks. Dropout helps to prevent overfitting by randomly setting a fraction of input units to zero at each update during training time, which is controlled by the probability `p`.

    Args:
        p (float, optional): Probability of an element to be zeroed. Defaults to 0.4.

    The module wraps the `nn.Dropout` function in PyTorch.
    """

    def __init__(self, p=0.4):
        super(Dropout, self).__init__(
            nn.Dropout(p)
        )
