import torch
import torch.nn as nn


class LSTNet(nn.Module):
    def __init__(self, refAR):
        super(LSTNet, self).__init__()
        self.refAR = refAR

    def forward(self, recordGD):
        return -1
