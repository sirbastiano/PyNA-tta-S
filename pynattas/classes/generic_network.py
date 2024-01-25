import torch.nn as nn
import modules.conv2d
import modules.mbconv
import modules.relu
import modules.gelu
import modules.classification_head
import modules.avgpool
import modules.maxpool
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
        -model_parameters (dict): A dictionary containing model-specific parameters like kernel size, stride, etc., for
            each layer.
        -input_channels (int, optional): The number of input channels. Default is 4.
        -input_height (int, optional): The height of the input tensor. Default is 256.
        -input_width (int, optional): The width of the input tensor. Default is 256.
        -num_classes (int, optional): The number of classes for the classification head. Default is 2.

        The architecture of the network is defined by the 'parsed_layers', which is a list of dictionaries where each
        dictionary contains the type of layer ('Conv2D', 'MBConv', etc.) and specific parameters for that layer.
        The 'model_parameters' dictionary complements this by providing detailed configuration for each layer,
        which allows for fine-grained control over the network's structure.

        The network supports dynamic input sizes and can adjust internal layer dimensions accordingly.
        The final layer is typically a classification head that aggregates features for the classification task.

        Example Usage:
            parsed_layers = [
                {'type': 'Conv2D', 'activation': 'ReLU'},
                {'type': 'MBConv', 'activation': 'GELU'},
                ...
            ]
            model_parameters = {
                'Conv2D_0_kernel_size': 3,
                'MBConv_1_expansion_factor': 4,
                ...
            }
            model = GenericNetwork(parsed_layers, model_parameters)

        Methods:
        forward(x): Defines the forward pass of the model.
        get_activation_fn(activation): Returns the activation function based on the specified string.
        """
    def __init__(self, parsed_layers, model_parameters, input_channels=4, input_height=256, input_width=256, num_classes=2):
        super(GenericNetwork, self).__init__()
        self.layers = nn.ModuleList()

        config = configparser.ConfigParser()
        config.read('config.ini')

        current_channels = input_channels
        current_height, current_width = input_height, input_width
        for index, layer_info in enumerate(parsed_layers):
            layer_type = layer_info['type']

            if layer_type == 'Conv2D':
                kernel_size = int(model_parameters.get(
                    f'Conv2D_{index}_kernel_size',
                    config['Conv2D']['default_kernel_size']
                ))
                stride = int(model_parameters.get(
                    f'Conv2D_{index}_stride',
                    config['Conv2D']['default_stride']
                ))
                padding = int(model_parameters.get(
                    f'Conv2D_{index}_padding',
                    config['Conv2D']['default_padding']
                ))
                out_channels_coeff = float(model_parameters.get(
                    f'Conv2D_{index}_out_channels_coefficient',
                    config['Conv2D']['default_out_channels_coefficient']
                ))

                out_channels = int(current_channels * out_channels_coeff)

                layer = modules.conv2d.Conv2D(
                    in_channels=current_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    activation=self.get_activation_fn(layer_info['activation']),
                )
                current_channels = out_channels
                current_height = ((current_height - kernel_size + 2 * padding) // stride) + 1
                current_width = ((current_width - kernel_size + 2 * padding) // stride) + 1

            elif layer_type == 'MBConv':
                # Extracting MBConv parameters
                expansion_factor = int(model_parameters.get(
                    f'MBConv_{index}_expansion_factor',
                    config['MBConv']['default_expansion_factor']
                ))

                # Creating MBConv layer
                layer = modules.mbconv.MBConv(
                    in_channels=current_channels,
                    out_channels=current_channels,
                    expansion_factor=expansion_factor,
                    activation=self.get_activation_fn(layer_info['activation']),
                )

                current_channels = current_channels
                current_height = current_height
                current_width = current_width

            elif layer_type == 'AvgPool':
                kernel_size = int(model_parameters.get(
                    f'AvgPool_{index}_kernel_size',
                    config['AvgPool']['default_kernel_size']
                ))
                stride = int(model_parameters.get(
                    f'AvgPool_{index}_stride',
                    config['AvgPool']['default_stride']
                ))

                layer = modules.avgpool.AvgPool(kernel_size=kernel_size, stride=stride)

                current_channels = current_channels
                current_height = ((current_height - kernel_size) // stride) + 1
                current_width = ((current_width - kernel_size) // stride) + 1

            elif layer_type == 'MaxPool':
                kernel_size = int(model_parameters.get(
                    f'MaxPool_{index}_kernel_size',
                    config['MaxPool']['default_kernel_size']
                ))
                stride = int(model_parameters.get(
                    f'MaxPool_{index}_stride',
                    config['MaxPool']['default_stride']
                ))

                layer = modules.maxpool.MaxPool(kernel_size=kernel_size, stride=stride)

                current_channels = current_channels
                current_height = ((current_height - kernel_size) // stride) + 1
                current_width = ((current_width - kernel_size) // stride) + 1

            elif layer_type == 'ClassificationHead':
                # Calculate the input size for ClassificationHead
                num_classes = int(config['ClassificationHead']['num_classes'])
                input_size_for_head = current_height * current_width * current_channels
                layer = modules.classification_head.ClassificationHead(input_size=input_size_for_head, num_classes=num_classes)

            self.layers.append(layer)

    @staticmethod
    def get_activation_fn(activation):
        if activation == 'ReLU':
            return modules.relu.ReLU
        elif activation == 'GELU':
            return modules.gelu.GELU
        # Add more activation functions as needed
        else:
            raise ValueError(f"Unknown activation function: {activation}")

    def forward(self, x):
        for layer in self.layers:
            if isinstance(layer, modules.classification_head.ClassificationHead):
                # Flatten the output before feeding it into the ClassificationHead
                x = x.view(x.size(0), -1)
            x = layer(x)
        return x
