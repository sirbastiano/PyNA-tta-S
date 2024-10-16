import torch 
import configparser
from typing import Dict, Tuple
from torch import nn

from ..blocks import *

def get_activation_fn(activation=activations.GELU):
    if activation == 'ReLU':
        return activations.ReLU
    elif activation == 'GELU':
        return activations.GELU
    elif activation == 'LeakyReLU':
        return activations.LeakyReLU
    # Add more activation functions as needed
    else:
        raise ValueError(f"Unknown activation function: {activation}")


def layer_torcher(layer: Dict[str, str], dims: Tuple[int, int, int]) -> nn.Module:
    # Necessary for basic information
    config = configparser.ConfigParser()
    config.read('config.ini')
    # For Sanithy checks
    current_channels, current_height, current_width = dims
    
    # For switch case
    layer_type = layer['layer_type']
    
    if layer_type == 'ConvAct':
        kernel_size = int(layer.get(
            'kernel_size',
            config['ConvAct']['default_kernel_size']
        ))
        stride = int(layer.get(
            'stride',
            config['ConvAct']['default_stride']
        ))
        padding = int(layer.get(
            'padding',
            config['ConvAct']['default_padding']
        ))
        out_channels_coeff = float(layer.get(
            'out_channels_coefficient',
            config['ConvAct']['default_out_channels_coefficient']
        ))

        assert kernel_size <= current_height and kernel_size <= current_width, f"Kernel size is too large for the current dimensions: Heigt=({current_height}), Width=({current_width})" 

        out_channels = int(current_channels * out_channels_coeff)

        layer = convolutions.ConvAct(
            in_channels=current_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            activation=get_activation_fn(layer['activation']),
        )
        current_channels = out_channels
        current_height = ((current_height - kernel_size + 2 * padding) // stride) + 1
        current_width = ((current_width - kernel_size + 2 * padding) // stride) + 1

    elif layer_type == 'ConvBnAct':
        kernel_size = int(layer.get(
            'kernel_size',
            config['ConvBnAct']['default_kernel_size']
        ))
        stride = int(layer.get(
            'stride',
            config['ConvBnAct']['default_stride']
        ))
        padding = int(layer.get(
            'padding',
            config['ConvBnAct']['default_padding']
        ))
        out_channels_coeff = float(layer.get(
            'out_channels_coefficient',
            config['ConvBnAct']['default_out_channels_coefficient']
        ))

        assert kernel_size <= current_height and kernel_size <= current_width, f"Kernel size is too large for the current dimensions: Heigt=({current_height}), Width=({current_width})" 

        out_channels = int(current_channels * out_channels_coeff)

        layer = convolutions.ConvBnAct(
            in_channels=current_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            activation=get_activation_fn(layer['activation']),
        )
        current_channels = out_channels
        current_height = ((current_height - kernel_size + 2 * padding) // stride) + 1
        current_width = ((current_width - kernel_size + 2 * padding) // stride) + 1

    elif layer_type == 'ConvSE':
        kernel_size = int(layer.get(
            'kernel_size',
            config['ConvSE']['default_kernel_size']
        ))
        stride = int(layer.get(
            'stride',
            config['ConvSE']['default_stride']
        ))
        padding = int(layer.get(
            'padding',
            config['ConvSE']['default_padding']
        ))
        out_channels_coeff = float(layer.get(
            'out_channels_coefficient',
            config['ConvSE']['default_out_channels_coefficient']
        ))

        assert kernel_size <= current_height and kernel_size <= current_width, f"Kernel size is too large for the current dimensions: Heigt=({current_height}), Width=({current_width})" 

        out_channels = int(current_channels * out_channels_coeff)

        layer = convolutions.ConvSE(
            in_channels=current_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            activation=get_activation_fn(layer['activation']),
        )
        current_channels = out_channels
        current_height = ((current_height - kernel_size + 2 * padding) // stride) + 1
        current_width = ((current_width - kernel_size + 2 * padding) // stride) + 1

    elif layer_type == 'MBConv':
        # Extracting MBConv parameters
        expansion_factor = int(layer.get(
            'expansion_factor',
            config['MBConv']['default_expansion_factor']
        ))
        dw_kernel_size = int(layer.get(
            'dw_kernel_size',
            config['MBConv']['default_dw_kernel_size']
        ))

        # Creating MBConv layer
        layer = convolutions.MBConv(
            in_channels=current_channels,
            out_channels=current_channels,
            expansion_factor=expansion_factor,
            dw_kernel_size=dw_kernel_size,
            activation=get_activation_fn(layer['activation']),
        )

        current_channels = current_channels
        current_height = current_height
        current_width = current_width

    elif layer_type == 'MBConvNoRes':
        # Extracting MBConv parameters
        expansion_factor = int(layer.get(
            'expansion_factor',
            config['MBConvNoRes']['default_expansion_factor']
        ))
        dw_kernel_size = int(layer.get(
            'dw_kernel_size',
            config['MBConvNoRes']['default_dw_kernel_size']
        ))

        # Creating MBConv layer
        layer = convolutions.MBConvNoRes(
            in_channels=current_channels,
            out_channels=current_channels,
            dw_kernel_size=dw_kernel_size,
            expansion_factor=expansion_factor,
            activation=get_activation_fn(layer['activation']),
        )

        current_channels = current_channels
        current_height = current_height
        current_width = current_width
    
    elif layer_type == 'CSPConvBlock':
        # Extracting CSPBlock parameters
        out_channels_coeff = float(layer.get(
            'out_channels_coefficient',
            config['CSPConvBlock']['default_out_channels_coefficient']
        ))
        num_blocks = int(layer.get(
            'num_blocks',
            config['CSPConvBlock']['default_num_blocks']
        ))
        out_channels = int(current_channels * out_channels_coeff)

        layer = convolutions.CSPConvBlock(
            in_channels=current_channels,
            #out_channels=current_channels,
            num_blocks=num_blocks,
            activation=get_activation_fn(layer['activation']),
        )
        current_channels = out_channels
        current_height = current_height
        current_width = current_width

    elif layer_type == 'CSPMBConvBlock':
        # Extracting CSPBlock parameters
        out_channels_coeff = float(layer.get(
            'out_channels_coefficient',
            config['CSPMBConvBlock']['default_out_channels_coefficient']
        ))
        num_blocks = int(layer.get(
            'num_blocks',
            config['CSPMBConvBlock']['default_num_blocks']
        ))
        expansion_factor = int(layer.get(
            'expansion_factor',
            config['CSPMBConvBlock']['default_expansion_factor']
        ))
        dw_kernel_size = int(layer.get(
            'dw_kernel_size',
            config['CSPMBConvBlock']['default_dw_kernel_size']
        ))
        out_channels = int(current_channels * out_channels_coeff)

        layer = convolutions.CSPMBConvBlock(
            in_channels=current_channels,
            #out_channels=current_channels,
            expansion_factor=expansion_factor,
            dw_kernel_size=dw_kernel_size,
            num_blocks=num_blocks,
            activation=get_activation_fn(layer['activation']),
        )
        current_channels = out_channels

    elif layer_type == 'DenseNetBlock':
        out_channels_coeff = float(layer.get(
            'out_channels_coefficient',
            config['DenseNetBlock']['default_out_channels_coefficient']
        ))

        out_channels = int(current_channels * out_channels_coeff)

        layer = convolutions.DenseNetBlock(
            in_channels=current_channels,
            out_channels=out_channels,
            activation=get_activation_fn(layer['activation']),
        )
        current_channels = current_channels + out_channels
        current_height = current_height
        current_width = current_width

    elif layer_type == 'ResNetBlock':
        out_channels_coeff = float(layer.get(
            'out_channels_coefficient',
            config['ResNetBlock']['default_out_channels_coefficient']
        ))
        reduction_factor = int(layer.get(
            'reduction_factor',
            config['ResNetBlock']['default_reduction_factor']
        ))

        out_channels = int(current_channels * out_channels_coeff)

        layer = convolutions.ResNetBlock(
            in_channels=current_channels,
            out_channels=current_channels,
            activation=get_activation_fn(layer['activation']),
        )
        current_channels = current_channels
        current_height = current_height
        current_width = current_width

    elif layer_type == 'AvgPool':
        kernel_size = int(layer.get(
            'kernel_size',
            config['AvgPool']['default_kernel_size']
        ))
        stride = int(layer.get(
            'stride',
            config['AvgPool']['default_stride']
        ))

        assert kernel_size <= current_height and kernel_size <= current_width, f"Kernel size is too large for the current dimensions: Heigt=({current_height}), Width=({current_width})" 

        layer = pooling.AvgPool(kernel_size=kernel_size, stride=stride)

        current_channels = current_channels
        current_height = ((current_height - kernel_size) // stride) + 1
        current_width = ((current_width - kernel_size) // stride) + 1

    elif layer_type == 'MaxPool':
        kernel_size = int(layer.get(
            'kernel_size',
            config['MaxPool']['default_kernel_size']
        ))
        stride = int(layer.get(
            'stride',
            config['MaxPool']['default_stride']
        ))

        assert kernel_size <= current_height and kernel_size <= current_width, f"Kernel size is too large for the current dimensions: Heigt=({current_height}), Width=({current_width})" 

        layer = pooling.MaxPool(kernel_size=kernel_size, stride=stride)

        current_channels = current_channels
        current_height = ((current_height - kernel_size) // stride) + 1
        current_width = ((current_width - kernel_size) // stride) + 1
    
    elif layer_type == 'Identity':
        layer = nn.Identity()

        current_channels = current_channels
        current_height = current_height
        current_width = current_width

    elif layer_type == 'ClassificationHead':
        # Calculate the input size for ClassificationHead
        num_classes = int(config['ClassificationHead']['num_classes']) # Useless, probably?
        # input_size_for_head = current_height * current_width * current_channels # TODO: Substituted with the following line
        bottle_neck_size = int(config['GeneralNetwork']['bottle_neck_size'])
        input_size_for_head = current_channels * bottle_neck_size * bottle_neck_size
        layer = heads.ClassificationHead(input_size=input_size_for_head, num_classes=num_classes)

    elif layer_type == 'DetectionHeadYOLOv3':
        # Calculate the input size for DetectionHeadYOLOv3
        yolo_conv_l = convolutions.ConvBnAct(outchannels_size[2], 1024, 1)
        yolo_conv_m = convolutions.ConvBnAct(outchannels_size[1], 512, 1)
        yolo_conv_s = convolutions.ConvBnAct(outchannels_size[0], 256, 1)
        layer = heads.DetectionHeadYOLOv3(num_classes=num_classes)

    elif layer_type == 'DetectionHeadYOLOv3_SmallObjects':
        # Calculate the input size for DetectionHeadYOLOv3_SmallObjects
        yolo_conv_m = convolutions.ConvBnAct(outchannels_size[1], 512, 1)
        yolo_conv_s = convolutions.ConvBnAct(outchannels_size[0], 256, 1)
        layer = heads.DetectionHeadYOLOv3_SmallObjects(num_classes=num_classes)

    else:
        raise ValueError(f"Unknown layer type: {layer_type}")
    
    return layer, (current_channels, current_height, current_width)