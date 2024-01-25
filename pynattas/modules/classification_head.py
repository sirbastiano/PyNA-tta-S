import torch.nn as nn


class ClassificationHead(nn.Sequential):
    def __init__(self, input_size, num_classes=2):
        super(ClassificationHead, self).__init__(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(512, num_classes)
        )
