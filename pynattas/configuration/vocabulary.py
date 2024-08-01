convolution_layer_vocabulary = {
    'b': 'ConvAct',
    'c': 'ConvBnAct',
    'e': 'ConvSE',
    'd': 'DenseNetBlock',
    'm': 'MBConv',
    'n': 'MBConvNoRes',
    'o': 'CSPConvBlock',
    'p': 'CSPMBConvBlock',
    'R': 'ResNetBlock',
}

activation_functions_vocabulary = {
    'r': 'ReLU',
    'g': 'GELU',
    'l': 'LeakyReLU'
}

pooling_layer_vocabulary = {
    'a': 'AvgPool',
    'M': 'MaxPool',
    'I': 'Identity',
}

head_vocabulary_C = {
    'C': 'ClassificationHead',
}

head_vocabulary_D = {
    #'Y': 'DetectionHeadYOLOv3',
    'S': 'DetectionHeadYOLOv3_SmallObjects'
}

head_vocabulary_S = {
}

layer_parameters = {
    'ConvAct': ['out_channels_coefficient', 'kernel_size', 'stride', 'padding', 'activation'],
    'ConvBnAct': ['out_channels_coefficient', 'kernel_size', 'stride', 'padding', 'activation'],
    'ConvSE': ['out_channels_coefficient', 'kernel_size', 'stride', 'padding', 'activation'],
    'DenseNetBlock': ['out_channels_coefficient', 'activation'],
    'MBConv': ['expansion_factor', 'activation'],
    'MBConvNoRes': ['expansion_factor', 'activation'],
    'CSPConvBlock': ['num_blocks', 'activation'],
    'CSPMBConvBlock': ['num_blocks', 'expansion_factor', 'activation'],
    'ResNetBlock': ['reduction_factor', 'activation'], #, 'num_blocks'
    'AvgPool': [],
    'MaxPool': [],
    'ClassificationHead': [],
    'DetectionHeadYOLOv3': ['outchannel1_index', 'outchannel2_index', 'outchannel3_index'],
    'DetectionHeadYOLOv3_SmallObjects': ['outchannel1_index', 'outchannel2_index'],
}

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
