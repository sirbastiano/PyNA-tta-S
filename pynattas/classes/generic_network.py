import torch
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

    def __init__(
            self, 
            parsed_layers, 
            input_channels: int, 
            input_height: int, 
            input_width: int, 
            num_classes: int, 
            ):
        super(GenericNetwork, self).__init__()
        self.layers = nn.ModuleList()
        if parsed_layers[-1]['layer_type'] == 'DetectionHeadYOLOv3':
            self.outchannels = [
                parsed_layers[-1]['outchannel1_index'],
                parsed_layers[-1]['outchannel2_index'],
                parsed_layers[-1]['outchannel3_index'],
            ]
        elif parsed_layers[-1]['layer_type'] == 'DetectionHeadYOLOv3_SmallObjects':
            self.outchannels = [
                parsed_layers[-1]['outchannel1_index'],
                parsed_layers[-1]['outchannel2_index'],
            ]
        self.outchannels_size = []
        self.is_too_deep = False

        config = configparser.ConfigParser()
        config.read('config.ini')

        current_channels = input_channels
        current_height, current_width = input_height, input_width


        # Jumpstart layer
        self.jumpstart = convolutions.ConvAct(
            in_channels=current_channels,
            out_channels=32,
            kernel_size=1,
            stride=1,
            padding=0,
            activation=activations.LeakyReLU,
        )
        current_channels = 32

        # Layers proper
        for layer_idx,layer in enumerate(parsed_layers):
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

                if kernel_size > current_height or kernel_size > current_width:
                    self.is_too_deep = True
                    break

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

                if kernel_size > current_height or kernel_size > current_width:
                    self.is_too_deep = True
                    break

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

                if kernel_size > current_height or kernel_size > current_width:
                    self.is_too_deep = True
                    break

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

                if kernel_size > current_height or kernel_size > current_width:
                    self.is_too_deep = True
                    break

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

                if kernel_size > current_height or kernel_size > current_width:
                    self.is_too_deep = True
                    break

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
                input_size_for_head = current_height * current_width * current_channels
                layer = heads.ClassificationHead(input_size=input_size_for_head, num_classes=num_classes)

            elif layer_type == 'DetectionHeadYOLOv3':
                # Calculate the input size for DetectionHeadYOLOv3
                self.yolo_conv_l = convolutions.ConvBnAct(self.outchannels_size[2], 1024, 1)
                self.yolo_conv_m = convolutions.ConvBnAct(self.outchannels_size[1], 512, 1)
                self.yolo_conv_s = convolutions.ConvBnAct(self.outchannels_size[0], 256, 1)
                layer = heads.DetectionHeadYOLOv3(num_classes=num_classes)

            elif layer_type == 'DetectionHeadYOLOv3_SmallObjects':
                # Calculate the input size for DetectionHeadYOLOv3_SmallObjects
                self.yolo_conv_m = convolutions.ConvBnAct(self.outchannels_size[1], 512, 1)
                self.yolo_conv_s = convolutions.ConvBnAct(self.outchannels_size[0], 256, 1)
                layer = heads.DetectionHeadYOLOv3_SmallObjects(num_classes=num_classes)

            else:
                raise ValueError(f"Unknown layer type: {layer_type}")
            
            if layer_idx in self.outchannels:
                self.outchannels_size.append(current_channels)

            self.layers.append(layer)

            if current_height == 1 or current_width == 1:
                self.is_too_deep = True
                break

        
    @staticmethod
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

    def forward(self, x):
        if torch.isnan(x).any():
            print(f"NaNs found in the input!")
        x = self.jumpstart(x)
        if torch.isnan(x).any():
            print(f"NaNs found after the jumpstart layer")
        outchannel_tensors = []
        for layer_idx, layer in enumerate(self.layers):

            if torch.isnan(x).any():
                print(f"NaNs found in raw output in training before layer {layer_idx}")
            #print(f"Shape of input before layer {layer_idx} is {x.shape}")

            if isinstance(layer, heads.ClassificationHead):
                # Flatten the output before feeding it into the ClassificationHead
                x = x.view(x.size(0), -1)

            if isinstance(layer, heads.DetectionHeadYOLOv3):
                # Change the shapes of the outchannels before feeding them into the DetectionHeadYOLOv3
                outchannel_tensors[2] = self.yolo_conv_l(outchannel_tensors[2])
                outchannel_tensors[1] = self.yolo_conv_m(outchannel_tensors[1])
                outchannel_tensors[0] = self.yolo_conv_s(outchannel_tensors[0])
                return layer(outchannel_tensors[::-1]) # Reverse the output layers to have the deeper ones be first
            
            if isinstance(layer, heads.DetectionHeadYOLOv3_SmallObjects):
                # Change the shapes of the outchannels before feeding them into the DetectionHeadYOLOv3
                outchannel_tensors[1] = self.yolo_conv_m(outchannel_tensors[1])
                outchannel_tensors[0] = self.yolo_conv_s(outchannel_tensors[0])
                return layer(outchannel_tensors[::-1]) # Reverse the output layers to have the deeper ones be first

            x = layer(x)

            if layer_idx in self.outchannels:
                outchannel_tensors.append(x)

        return x
