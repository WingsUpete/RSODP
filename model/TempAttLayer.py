import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F

from .ScaledDotProductAttention import ScaledDotProductAttention


class TempAttLayer(nn.Module):
    def __init__(self):
        super(TempAttLayer, self).__init__()

    def forward(self):
        pass


if __name__ == '__main__':
    print('Hello World')
