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