import torch
import torch.nn as nn
import configparser
import torchsummary

from ..blocks import *
from .toTorch import layer_torcher
from pynattas.utils import layerCoder
from tqdm import tqdm


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
            architecture_code: str, 
            input_channels: int, 
            input_height: int, 
            input_width: int, 
            num_classes: int,
            features_only: bool = False, 
            ):
        super(GenericNetwork, self).__init__()
        self.config = configparser.ConfigParser()
        self.config.read('config.ini')
        
        self.parsed_layers = layerCoder.parse_architecture_code(architecture_code)
        self.num_classes = num_classes
        self.input_channels = input_channels
        self.input_height = input_height
        self.input_width = input_width
        self.features_only = features_only
        
        # TODO: Parsed layers must be obtained inside this GenericNetwork class
        self._jumpstart_channels = int(self.config['GeneralNetwork']['jumpstart_channels'])
        self.layers = nn.ModuleList()


    # TODO: Change builder to separate sections for heads
    def build(self):
        """
        Builds a neural network from the parsed layer configurations.

        Parameters:
        -----------
        parsed_layers : list
            List of layer configurations.
        input_channels : int
            The number of input channels (e.g., 3 for RGB images).
        input_height : int
            The height of the input images.
        input_width : int
            The width of the input images.
        num_classes : int
            The number of output classes for the classification task.
        
        Returns:
        --------
        None
        """
        # Initial input dimensions
        parsed_layers = self.parsed_layers
        num_classes = self.num_classes
        dims = (self.input_channels, self.input_height, self.input_width)
        # Jumpstart layer --> Expanding the number of channels to the desired number
        jumpstart = convolutions.ConvAct(
            in_channels=self.input_channels,
            out_channels=self._jumpstart_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            activation=activations.LeakyReLU,)
        dims = (self._jumpstart_channels, self.input_height, self.input_width)
        self.layers.append(jumpstart)
        
        # **- Initialize progress bar outside the loop -**
        progress_bar = tqdm(total=len(parsed_layers), desc="Building Network", unit="layer")
        
        for layer_idx, layer in enumerate(parsed_layers): # Layer processing
            
            # Convert the layer configuration into a PyTorch layer and update dimensions
            torchlayer, dims = layer_torcher(layer, dims)
            
            # Update progress bar with current layer status
            progress_bar.set_description(f"Building layer {layer_idx + 1}/{len(parsed_layers)}: {layer['layer_type']}")
            progress_bar.update(1)
            
            # Extract the current height and width after layer application
            current_channels, current_height, current_width = dims
            
            # Extract kernel size for assertion
            kernel_size = layer.get('kernel_size', 1)  # Default to 1 if not found
            if 'kernel_size' in layer.keys():
                kernel_size = int(layer['kernel_size'])
            
            # Assert that the kernel size is appropriate for the current dimensions
            assert kernel_size <= current_height and kernel_size <= current_width, \
                f"Kernel size {kernel_size} is too large for the current dimensions: Heigt=({current_height}), Width=({current_width})"
            
            # Add the constructed layer to the model
            self.layers.append(torchlayer)

        # Close the progress bar
        progress_bar.close()


    def print_summary(self, input_size=(3, 224, 224)):
        return torchsummary.summary(self, input_size)


    def get_param_size(self):
        """
        Calculate the total size of the model parameters in megabytes (MB).

        This method sums up the number of elements in all the parameters of the model,
        assuming each parameter is a 32-bit float (4 bytes), and converts the total size
        to megabytes.

        Returns:
            float: The total size of the model parameters in MB, rounded to three decimal places.
        """
        # Extract the total parameters and calculate the size in MB
        total_params = sum(p.numel() for p in self.parameters())
        params_size_mb = total_params * 4 / (1024 ** 2)  # Assuming 4 bytes per parameter (float32)
        return round(params_size_mb, 3)


    def forward(self, x):
        """
        Defines the forward pass of the network with optimized performance.
        
        Parameters:
        -----------
        x : torch.Tensor
            Input tensor to the model.
        
        Returns:
        --------
        torch.Tensor
            Output tensor after passing through all layers.
        """
        # Check for NaNs in the input tensor once
        if torch.isnan(x).any():
            raise ValueError("NaNs detected in the input tensor.")

        # Process layers in the network
        for layer_idx, layer in enumerate(self.layers):

            if isinstance(layer, heads.ClassificationHead):
                # This reduces the feature map to a fixed size (e.g., 1x1)
                x = nn.AdaptiveAvgPool2d((6, 6))(x)
                # Efficiently flatten the output before the classification head
                x = x.view(x.size(0), -1)

            x = layer(x)
        return x
