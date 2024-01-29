import torch.nn as nn


class ReLU(nn.Module):
    def forward(self, x):
        return nn.ReLU()(x)


class GELU(nn.Module):
    def forward(self, x):
        return nn.GELU()(x)
