import torch
import torch.nn as nn
import torch.nn.functional as F

from .SpatAttLayer import SpatAttLayer

import Config

class GEML(nn.Module):
    def __init__(self, feat_dim=43, query_dim=41, hidden_dim=16):
        super(GEML, self).__init__()

        self.feat_dim = feat_dim
        self.query_dim = query_dim
        self.hidden_dim = hidden_dim

        self.num_dim = 2

        self.spat_embed_dim = int(self.num_dim * self.hidden_dim)     # Embedding dimension after spatial feature extraction
        self.temp_embed_dim = self.spat_embed_dim   # Embedding dimension after temporal feature extraction

        # Spatial Attention Layer
        self.spatAttLayer = SpatAttLayer(feat_dim=self.feat_dim, hidden_dim=self.hidden_dim, num_heads=1, att=False, gate=False, merge='mean', num_dim=self.num_dim, cat_orig=False)

        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('linear')
        # nn.init.xavier_normal_(self.l_stConv_last_D.weight, gain=gain)

    def forward(self, record_p: list):
        # Extract spatial features
        spat_embed_p = [self.spatAttLayer(list(gs)) for gs in record_p]

        if spat_embed_p[-1].device.type == 'cuda':
            torch.cuda.empty_cache()

        # Extract temporal features

