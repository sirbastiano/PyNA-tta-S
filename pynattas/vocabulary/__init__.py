from pynattas.vocabulary.layers import (
    convolution_layer_vocabulary,
    activation_functions_vocabulary,
    pooling_layer_vocabulary,
    layer_parameters
)


from pynattas.vocabulary.conv import parameter_vocabulary, parameter_vocabulary_rev

from pynattas.vocabulary.layers import (
    head_vocabulary_C,
    head_vocabulary_D,
    head_vocabulary_S,
)


if __name__ == '__main__':
    print(convolution_layer_vocabulary)
    print(activation_functions_vocabulary)
    print(pooling_layer_vocabulary)
    print(parameter_vocabulary)
    print(head_vocabulary_C)
    print(head_vocabulary_D)
    print(head_vocabulary_S)