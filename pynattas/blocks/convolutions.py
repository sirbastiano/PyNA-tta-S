import math
from collections import OrderedDict
from typing import Callable, Tuple, Optional, List
from torch import nn
from torch import Tensor
import torch
import torch.nn.functional as F
from typing import Optional

from torchvision.ops import StochasticDepth

from .activations import *


class Conv2dPad(nn.Conv2d):
    """2D Convolutions with different padding modes.

    'auto' will use the kernel_size to calculate the padding
    'same' same padding as TensorFLow. It will dynamically pad the image based on its size

    Args:
        mode (str, optional): [description]. Defaults to 'auto'.
    """

    def __init__(self, *args, mode: str = "auto", padding: int = 0, **kwargs):

        super().__init__(*args, **kwargs)
        self.mode = mode
        # dynamic add padding based on the kernel_size
        if self.mode == "auto":
            self.padding = (
                self._get_padding(padding)
                if padding != 0
                else (self.kernel_size[0] // 2, self.kernel_size[1] // 2)
            )

    def _get_padding(self, padding: int) -> Tuple[int, int]:
        return (padding, padding)

    def forward(self, x: Tensor) -> Tensor:
        if self.mode == "same":
            ih, iw = x.size()[-2:]
            kh, kw = self.weight.size()[-2:]
            sh, sw = self.stride
            # change the output size according to stride
            oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
            pad_h = max(
                (oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0
            )
            pad_w = max(
                (ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0
            )
            if pad_h > 0 or pad_w > 0:
                x = F.pad(
                    x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2]
                )
            return F.conv2d(
                x,
                self.weight,
                self.bias,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )

        else:
            return super().forward(x)


class ConvNormAct(nn.Sequential):
    """Utility module that stacks one convolution layer, a normalization layer and an activation function.

    Example:
        >>> ConvNormAct(32, 64, kernel_size=3)
            ConvNormAct(
                (conv): Conv2dPad(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (norm): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (act): ReLU()
            )

        >>> ConvNormAct(32, 64, kernel_size=3, normalization = None )
            ConvNormAct(
                (conv): Conv2dPad(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (act): ReLU()
            )

        >>> ConvNormAct(32, 64, kernel_size=3, activation = None )
            ConvNormAct(
                (conv): Conv2dPad(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (norm): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )

    We also provide additional modules built on top of this one: `ConvBn`, `ConvAct`, `Conv3x3BnAct`
    Args:
            out_features (int): Number of input features
            out_features (int): Number of output features
            conv (nn.Module, optional): Convolution layer. Defaults to Conv2dPad.
            normalization (nn.Module, optional): Normalization layer. Defaults to nn.BatchNorm2d.
            activation (nn.Module, optional): Activation function. Defaults to nn.ReLU.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        conv: nn.Module = Conv2dPad,
        activation: Optional[nn.Module] = nn.ReLU,
        normalization: Optional[nn.Module] = nn.BatchNorm2d,
        bias: bool = False,
        **kwargs
    ):
        super().__init__()
        self.add_module("conv", conv(in_features, out_features, **kwargs, bias=bias))
        if normalization:
            self.add_module("norm", normalization(out_features))
        if activation:
            self.add_module("act", activation())


class ConvNormRegAct(nn.Sequential):
    """Utility module that stacks one convolution layer, a normalization layer, a regularization layer and an activation function.

    Example:
        >>> ConvNormDropAct(32, 64, kernel_size=3)
            ConvNormDropAct(
                (conv): Conv2dPad(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (norm): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (reg): StochasticDepth(p=0.2)
                (act): ReLU()
            )
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        conv: nn.Module = Conv2dPad,
        activation: Optional[nn.Module] = nn.ReLU,
        normalization: Optional[nn.Module] = nn.BatchNorm2d,
        regularization: Optional[nn.Module] = partial(StochasticDepth, mode="batch"),
        p: float = 0.2,
        bias: bool = False,
        **kwargs
    ):
        super().__init__()
        self.add_module("conv", conv(in_features, out_features, **kwargs, bias=bias))
        if normalization:
            self.add_module("norm", normalization(out_features))
        if regularization:
            self.add_module("reg", regularization(p=p))
        if activation:
            self.add_module("act", activation())


class NormActConv(nn.Sequential):
    """A Sequential layer composed by a normalization, an activation and a convolution layer. This is usually known as a 'Preactivation Block'

    Args:
        in_features (int): Number of input features
        out_features (int): Number of output features
        conv (nn.Module, optional): [description]. Defaults to Conv2dPad.
        normalization (nn.Module, optional): [description]. Defaults to nn.BatchNorm2d.
        activation (nn.Module, optional): [description]. Defaults to nn.ReLU.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        conv: nn.Module = Conv2dPad,
        normalization: Optional[nn.Module] = nn.BatchNorm2d,
        activation: Optional[nn.Module] = ReLUInPlace,
        *args,
        **kwargs
    ):
        super().__init__()
        if normalization:
            self.add_module("norm", normalization(in_features))
        if activation:
            self.add_module("act", activation())
        self.add_module("conv", conv(in_features, out_features, *args, **kwargs))


ConvBnAct = partial(ConvNormAct, normalization=nn.BatchNorm2d)
ConvBn = partial(ConvBnAct, activation=None)
ConvAct = partial(ConvBnAct, normalization=None, bias=True)
Conv3x3BnAct = partial(ConvBnAct, kernel_size=3)
BnActConv = partial(NormActConv, normalization=nn.BatchNorm2d)
ConvBnDropAct = partial(
    ConvNormRegAct, normalization=nn.BatchNorm2d, regularization=nn.Dropout2d
)


class DenseNetBlock(nn.Module):
    """Basic DenseNet block composed by one 3x3 convs with residual connection.
    The residual connection is perfomed by concatenate the input and the output.

    .. borrowed from:: https://github.com/FrancescoSaverioZuppichini/glasses/

    Args:
        out_features (int): Number of input features
        out_features (int): Number of output features
        conv (nn.Module, optional): [description]. Defaults to nn.Conv2d.
        activation (nn.Module, optional): [description]. Defaults to ReLUInPlace.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation: nn.Module = ReLUInPlace,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.block = nn.Sequential(
            OrderedDict(
                {
                    "bn": nn.BatchNorm2d(in_features),
                    "act": activation(),
                    "conv": Conv2dPad(
                        in_features, out_features, kernel_size=3, *args, **kwargs
                    ),
                }
            )
        )

    def forward(self, x: Tensor) -> Tensor:
        res = x
        x = self.block(x)
        return torch.cat([res, x], dim=1)
    
    
# Mine:
class MBConv(nn.Module):
    """Mobile Inverted Bottleneck Convolution (MBConv) Block.

    This module implements the MBConv block, a key component of MobileNetV2 and EfficientNet architectures. It uses an inverted residual structure where the input and output are typically thin layers (narrow), and it expands to a thicker layer (wide) in between. The block consists of three convolutional steps: 1x1 convolution for expansion, 3x3 depthwise convolution, and 1x1 convolution for projection. Each convolution is followed by batch normalization and an activation function. The block includes a residual connection adding the input to the output.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        expansion_factor (int, optional): Factor by which the channels are expanded. Defaults to 4.
        activation (nn.Module, optional): Activation function to be used. Defaults to modules.relu.ReLU.

    The sequence of operations is: 1x1 Conv -> BatchNorm -> Activation -> 3x3 Depthwise Conv -> BatchNorm -> Activation -> 1x1 Conv -> BatchNorm -> Activation -> Residual Connection.
    """

    def __init__(self, in_channels, out_channels, expansion_factor=4, activation=ReLU):
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
    """Mobile Inverted Bottleneck Convolution (MBConv) Block without Residual Connection.

    This class implements a variant of the MBConv block, commonly used in MobileNetV2 and EfficientNet architectures, but without a residual connection. It consists of three main steps: an expansion layer (1x1 convolution), a depthwise convolution, and a projection layer (1x1 convolution). Each convolutional layer is followed by batch normalization and an activation function. This block is suitable for scenarios where residual connections are not desirable or necessary.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        expansion_factor (int): Factor by which the channels are expanded.
        kernel_size (int, optional): Size of the kernel for the depthwise convolution. Defaults to 3.
        stride (int, optional): Stride of the depthwise convolution. Defaults to 1.
        activation (nn.Module, optional): Activation function to be used. Defaults to ReLU.

    The sequence of operations is: Expansion (1x1 Conv) -> BatchNorm -> Activation -> Depthwise Conv -> BatchNorm -> Activation -> Projection (1x1 Conv) -> BatchNorm -> Activation.
    """

    def __init__(self, in_channels, out_channels, expansion_factor, kernel_size=3, stride=1, activation=ReLU):
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
    """Residual Addition Module with Optional Shortcut.

    This module implements a residual connection in a neural network. It applies a specified block to the input tensor and adds the result back to the original input, forming a residual connection. Additionally, it allows for an optional shortcut block to modify the input before adding it back. This design is prevalent in architectures like ResNets, enhancing training deep networks by addressing the vanishing gradient problem.

    Args:
        block (nn.Module): The primary block to be applied to the input tensor.
        shortcut (Optional[nn.Module]): An optional block to be applied to the input tensor before adding it to the block output. If None, the original input tensor is used.

    Returns:
        Tensor: The output tensor after processing through the block and combining with the shortcut connection.
    """

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
    def __init__(self, in_channels, out_channels, expansion_factor=4, kernel_size=3, stride=1, activation=ReLU):
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