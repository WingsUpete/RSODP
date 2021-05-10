import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F

from .SpatAttLayer import SpatAttLayer


class Gallat(nn.Module):
    def __init__(self, feat_dim=7, query_dim=5, hidden_dim=16):
        super(Gallat, self).__init__()
        self.feat_dim = feat_dim
        self.query_dim = query_dim
        self.hidden_dim = hidden_dim

        # Spatial Attention Layer
        self.spatAttLayer = SpatAttLayer(feat_dim=self.feat_dim, hidden_dim=self.hidden_dim, num_heads=1, gate=False)

    def forward(self, record, query):
        return 0


if __name__ == '__main__':
    print('Hello World')
