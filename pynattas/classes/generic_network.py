import torch.nn as nn
from ..blocks import *
import configparser


class GenericNetwork(nn.Module):
    """
        A customizable neural network that dynamically constructs a model based on a specified architecture.

        This class allows the construction of a neural network from a list of layer specifications,
        making it highly flexible for experimenting with different architectures. The class supports
        various types of layers including convolutional layers, MobileNet-like blocks (MBConv),
        pooling layers, and a classification head. It dynamically adjusts the network architecture
        based on the input configuration and supports custom parameters for each layer.

        Parameters:
        -parsed_layers (list of dicts): A list where each element is a dictionary specifying a layer type and its
            parameters.
        -layer (dict): A dictionary containing model-specific parameters like kernel size, stride, etc., for
            each layer.
        -input_channels (int, optional): The number of input channels. Default is 4.
        -input_height (int, optional): The height of the input tensor. Default is 256.
        -input_width (int, optional): The width of the input tensor. Default is 256.
        -num_classes (int, optional): The number of classes for the classification head. Default is 2.

        The architecture of the network is defined by the 'parsed_layers', which is a list of dictionaries where each
        dictionary contains the type of layer ('Conv2D', 'MBConv', etc.) and specific parameters for that layer.
        The 'layer' dictionary complements this by providing detailed configuration for each layer,
        which allows for fine-grained control over the network's structure.

        The network supports dynamic input sizes and can adjust internal layer dimensions accordingly.
        The final layer is typically a classification head that aggregates features for the classification task.

        Example Usage:
            parsed_layers = [
                {'layer_type': 'Conv2D', 'activation': 'ReLU'},
                {'layer_type': 'MBConv', 'activation': 'GELU'},
                ...
            ]
            model = GenericNetwork(parsed_layers)

        Methods:
        forward(x): Defines the forward pass of the model.
        get_activation_fn(activation): Returns the activation function based on the specified string.
    """

    def __init__(self, parsed_layers, input_channels=3, input_height=256, input_width=256, num_classes=2):
        super(GenericNetwork, self).__init__()
        self.layers = nn.ModuleList()

        config = configparser.ConfigParser()
        config.read('config.ini')

        current_channels = input_channels
        current_height, current_width = input_height, input_width
        for layer in parsed_layers:
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

                out_channels = int(current_channels * out_channels_coeff)

                layer = convolutions.ConvAct(
                    in_channels=current_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    activation=self.get_activation_fn(layer['activation']),
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

                out_channels = int(current_channels * out_channels_coeff)

                layer = convolutions.ConvBnAct(
                    in_channels=current_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    activation=self.get_activation_fn(layer['activation']),
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

                out_channels = int(current_channels * out_channels_coeff)

                layer = convolutions.ConvSE(
                    in_channels=current_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    activation=self.get_activation_fn(layer['activation']),
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
                    activation=self.get_activation_fn(layer['activation']),
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
                    activation=self.get_activation_fn(layer['activation']),
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
                    activation=self.get_activation_fn(layer['activation']),
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
                    activation=self.get_activation_fn(layer['activation']),
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
                    activation=self.get_activation_fn(layer['activation']),
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
                    activation=self.get_activation_fn(layer['activation']),
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

                layer = pooling.MaxPool(kernel_size=kernel_size, stride=stride)

                current_channels = current_channels
                current_height = ((current_height - kernel_size) // stride) + 1
                current_width = ((current_width - kernel_size) // stride) + 1

            elif layer_type == 'ClassificationHead':
                # Calculate the input size for ClassificationHead
                num_classes = int(config['ClassificationHead']['num_classes'])
                input_size_for_head = current_height * current_width * current_channels
                layer = heads.ClassificationHead(input_size=input_size_for_head, num_classes=num_classes)

            else:
                raise ValueError(f"Unknown layer type: {layer_type}")

            self.layers.append(layer)

    @staticmethod
    def get_activation_fn(activation=activations.GELU):
        if activation == 'ReLU':
            return activations.ReLU
        elif activation == 'GELU':
            return activations.GELU
        # Add more activation functions as needed
        else:
            raise ValueError(f"Unknown activation function: {activation}")

    def forward(self, x):
        for layer in self.layers:
            if isinstance(layer, heads.ClassificationHead):
                # Flatten the output before feeding it into the ClassificationHead
                x = x.view(x.size(0), -1)
            x = layer(x)
        return x
