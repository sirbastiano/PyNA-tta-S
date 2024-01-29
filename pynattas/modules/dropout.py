import torch.nn as nn


class Dropout(nn.Sequential):
    def __init__(self, p=0.4):
        super(Dropout, self).__init__(
            nn.Dropout(p)
        )
