import torch.nn as nn


class ReLU(nn.Module):
    def forward(self, x):
        return nn.ReLU()(x)