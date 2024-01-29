import torch.nn as nn


class ClassificationHead(nn.Sequential):
    """Classification Head for Neural Networks.

    This module represents a classification head typically used at the end of a neural network. It consists of a linear layer, a ReLU activation, dropout for regularization, and a final linear layer that maps to the number of classes. This head is designed to be attached to the feature-extracting layers of a network to perform classification tasks.

    Args:
        input_size (int): The size of the input features.
        num_classes (int, optional): The number of classes for classification. Defaults to 2.

    The sequence of operations is as follows: Linear -> ReLU -> Dropout -> Linear.
    """

    def __init__(self, input_size, num_classes=2):
        super(ClassificationHead, self).__init__(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(512, num_classes)
        )
