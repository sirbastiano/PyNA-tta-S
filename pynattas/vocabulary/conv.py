# Vocabulary for different convolutional layers
convolution_layer_vocabulary = {
    'b': 'ConvAct',           # Basic Conv layer with activation
    'c': 'ConvBnAct',         # Conv layer with batch normalization and activation
    'e': 'ConvSE',            # Conv layer with Squeeze-and-Excitation
    'd': 'DenseNetBlock',     # DenseNet block
    'm': 'MBConv',            # MobileNet inverted bottleneck convolution
    'n': 'MBConvNoRes',       # MobileNet bottleneck convolution without residuals
    'O': 'CSPConvBlock',      # Cross Stage Partial convolutional block
    'P': 'CSPMBConvBlock',    # Cross Stage Partial MBConv block
    'R': 'ResNetBlock',       # Residual Network block
}

# Common parameters for convolutional layers with activation
common_conv_params = ['out_channels_coefficient', 'kernel_size', 'stride', 'padding', 'activation']
mbconv_params = ['expansion_factor', 'activation']

# Layer-specific parameters for each layer type
layer_parameters = {
    'ConvAct': common_conv_params,
    'ConvBnAct': common_conv_params,
    'ConvSE': common_conv_params,
    'DenseNetBlock': ['out_channels_coefficient', 'activation'],
    'MBConv': mbconv_params,
    'MBConvNoRes': mbconv_params,
    'CSPConvBlock': ['num_blocks', 'activation'],
    'CSPMBConvBlock': ['num_blocks', 'expansion_factor', 'activation'],
    'ResNetBlock': ['reduction_factor', 'activation'],
    'AvgPool': [],
    'MaxPool': [],
    'ClassificationHead': [],
    'DetectionHeadYOLOv3': ['outchannel1_index', 'outchannel2_index', 'outchannel3_index'],
    'DetectionHeadYOLOv3_SmallObjects': ['outchannel1_index', 'outchannel2_index'],
}

# Vocabulary for the parameter names used in architecture codes
parameter_vocabulary = {
    'kernel_size': 'k',
    'stride': 's',
    'padding': 'p',
    'out_channels_coefficient': 'o',
    'expansion_factor': 'e',
    'num_blocks': 'n',
    'reduction_factor': 'r',
    'activation': 'a',
    'outchannel1_index': 'u',
    'outchannel2_index': 'v',
    'outchannel3_index': 'w',
}