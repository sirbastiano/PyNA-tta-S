import math
from typing import Callable
from torch import nn, Tensor
from torch.nn import functional as F
from torchvision.ops import StochasticDepth

__all__ = ["activations", "convolutions", "pooling", "utils", "heads"]

class Lambda(nn.Module):
    """A utility module that allows a custom function to be passed and applied to a tensor.

    Args:
        lambd (Callable[[Tensor], Tensor]): A function that processes a tensor.

    Examples:
        >>> add_two = Lambda(lambda x: x + 2)
        >>> add_two(Tensor([0]))  # Output: tensor([2.])
    """

    def __init__(self, lambd: Callable[[Tensor], Tensor]):
        super().__init__()
        self.lambd = lambd

    def forward(self, x: Tensor) -> Tensor:
        return self.lambd(x)