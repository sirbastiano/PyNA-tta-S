# HEAD LAYERS
from .heads.classification_heads import head_vocabulary_C
from .heads.detection_heads import head_vocabulary_D
from .heads.segmentation_heads import head_vocabulary_S

# BASE LAYERS
from .conv import convolution_layer_vocabulary
from .act import activation_functions_vocabulary
from .pooling import pooling_layer_vocabulary


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