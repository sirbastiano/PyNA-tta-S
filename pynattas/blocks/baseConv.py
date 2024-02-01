import torch
import torch.nn as nn

class InvertedResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expansion_factor, stride=1):
        super(InvertedResidualBlock, self).__init__()

        self.bottleneck = nn.Conv2d(in_channels, in_channels * expansion_factor, kernel_size=1, stride=stride)
        self.residual_path = nn.Sequential(
            nn.Conv2d(in_channels * expansion_factor, in_channels * expansion_factor, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(in_channels * expansion_factor),
            nn.ReLU6(inplace=True),
            nn.Conv2d(in_channels * expansion_factor, out_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        bottleneck_output = self.bottleneck(x)
        return x + self.residual_path(bottleneck_output)
    
    
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, act_layer=None, norm_layer=None, drop_path_rate=None):
        super().__init__()

        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, groups=in_channels, padding=kernel_size // 2)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)

        if norm_layer is not None:
            self.norm = norm_layer(out_channels)
        else:
            self.norm = None

        self.act_layer = act_layer
        
        self.drop_path_rate = drop_path_rate if drop_path_rate is not None else 0.0

    def forward(self, x):
        
        x = self.depthwise(x)
        x = self.pointwise(x)
        if self.act_layer is not None:
            x = self.act_layer(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.drop_path_rate > 0.0:
            x = drop_path(x, self.drop_path_rate)
        return x


class AtrousConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=2):
        super(AtrousConv, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation)

    def forward(self, x):
        x = self.conv(x)
        return x

class DilatedConv(nn.Module):
    # Dilated convolution: This type of convolution is similar to an atrous convolution, 
    # but it uses a different method to increase the receptive field of the convolution. 
    # Dilated convolutions use a stride of 1, but they insert zeros between the input pixels. 
    # This allows the convolution to capture more context from the input image, without increasing 
    # the kernel size.
    # https://www.geeksforgeeks.org/dilated-convolution/
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=2):
        super(DilatedConv, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=dilation * (kernel_size - 1))
    def forward(self, x):
        x = self.conv(x)
        return x

class SqueezeAndExcitationBlock(nn.Module):
    def __init__(self, in_channels, num_outputs=None, rd_ratio=0.2):
        super().__init__()
        
        if num_outputs is None:
            num_outputs = int(in_channels * (1 - rd_ratio))
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(in_channels, num_outputs, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(num_outputs, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        features = self.squeeze(x).view(b, c)
        excitation = self.excitation(features).view(b, c, 1, 1)
        return x * excitation
    
    
class ResidualDeconvolutionUpsample2d(nn.Module):
    # The residual connection helps to preserve the features of the input tensor, 
    # while the upsampling layer increases the resolution of the output tensor.
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, output_padding=1):
        super(ResidualDeconvolutionUpsample2d, self).__init__()

        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding)

    def forward(self, x):
        residual = self.residual(x)
        upsample = self.upsample(x)
        return residual + upsample
    

def drop_path(x, drop_prob=0.0):
    """
    Drop path with probability (0.0-1.0).

    Args:
        x (Tensor): input tensor.
        drop_prob (float): probability of an element to be dropped.

    Returns:
        Tensor: output tensor.
    """
    if drop_prob == 0.0:
        return x
    keep_prob = 1.0 - drop_prob
    dim = x.size(-1)
    mask = (torch.rand(dim, device=x.device) < keep_prob).float()
    x = x.masked_fill(mask, 0.0)
    return x


def trunc_normal_(tensor, mean=0.0, std=1., a=-2., b=2.):
    """
    Truncated normal initialization.
    Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq text{mean} \leq b`.

    Args:
        tensor (Tensor): input tensor.
        mean (float): mean of the truncated normal distribution.
        std (float): standard deviation of the truncated normal distribution.

    Returns:
        Tensor: output tensor.
    """
    if not std > 0.0:
        raise ValueError("Argument `std` should be greater than zero.")
    with torch.no_grad():
        return tensor.normal_(mean, std).clamp_(a, b)
    

# def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
#     r"""Fills the input Tensor with values drawn from a truncated
#     normal distribution. The values are effectively drawn from the
#     normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
#     with values outside :math:`[a, b]` redrawn until they are within
#     the bounds. The method used for generating the random values works
#     best when :math:`a \leq \text{mean} \leq b`.

#     # type: (Tensor, float, float, float, float) -> Tensor
#     NOTE: this impl is similar to the PyTorch trunc_normal_, the bounds [a, b] are
#     applied while sampling the normal with mean/std applied, therefore a, b args
#     should be adjusted to match the range of mean, std args.

#     Args:
#         tensor: an n-dimensional `torch.Tensor`
#         mean: the mean of the normal distribution
#         std: the standard deviation of the normal distribution
#         a: the minimum cutoff value
#         b: the maximum cutoff value
#     Examples:
#         >>> w = torch.empty(3, 5)
#         >>> nn.init.trunc_normal_(w)
#     """
#     with torch.no_grad():
#         return _trunc_normal_(tensor, mean, std, a, b)


class Mlp(nn.Module):
    """
    MLP block.

    Args:
        in_features (int): number of input features.
        hidden_features (int, optional): number of hidden features. Defaults to None.
        out_features (int, optional): number of output features. Defaults to None.
        act_func (nn.Module, optional): activation function. Defaults to nn.ReLU.
        drop_path_rate (float, optional): dropout path rate. Defaults to 0.0.

    Returns:
        Tensor: output tensor.
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_func=nn.ReLU, drop_path_rate=0.0):
        super(Mlp, self).__init__()
        if hidden_features is None:
            hidden_features = in_features
        if out_features is None:
            out_features = in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act_func = act_func
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop_path = drop_path_rate

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_func(x)
        x = self.drop_path(x) if self.drop_path > 0.0 else x
        x = self.fc2(x)
        return x


class DropPath(nn.Module):
    """
    DropPath module.

    Args:
        drop_prob (float): drop path rate.

    Returns:
        Tensor: output tensor.
    """

    def __init__(self, drop_prob=0.0):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob)
