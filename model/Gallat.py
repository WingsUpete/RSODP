import time
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F

from .SpatAttLayer import SpatAttLayer
from .TempAttLayer import TempAttLayer
from .TranAttLayer import TranAttLayer

TEMP_FEAT_NAMES = ['St', 'Sp', 'Stpm', 'Stpp']


class Gallat(nn.Module):
    def __init__(self, feat_dim=7, query_dim=5, hidden_dim=16):
        super(Gallat, self).__init__()
        self.feat_dim = feat_dim
        self.query_dim = query_dim
        self.hidden_dim = hidden_dim

        self.spat_embed_dim = 4 * hidden_dim    # Embedding dimension after spatial feature extraction
        self.temp_embed_dim = 4 * hidden_dim    # Embedding dimension after temporal feature extraction

        # Spatial Attention Layer
        self.spatAttLayer = SpatAttLayer(feat_dim=self.feat_dim, hidden_dim=self.hidden_dim, num_heads=1, gate=False)
        self.spatActivation = nn.ReLU()

        # Temporal Attention Layer
        self.tempAttLayer = TempAttLayer(query_dim=self.query_dim, embed_dim=self.spat_embed_dim, rec_merge='sum', comb_merge='sum')
        self.tempActivation = nn.ReLU()

        # Transferring Attention Layer
        self.tranAttLayer = TranAttLayer(embed_dim=self.temp_embed_dim, activate_function_method=None)

    def forward(self, record, query, predict_G=False):
        # Extract spatial features
        spat_embed_dict = {}
        for temp_feat in TEMP_FEAT_NAMES:
            spat_embed_dict[temp_feat] = [self.spatActivation(self.spatAttLayer(fg, bg, gg)) for (fg, bg, gg) in record[temp_feat]]

        # Extract temporal features
        temp_embed = self.tempActivation(self.tempAttLayer(query, spat_embed_dict))

        # Transferring features to perform predictions
        res = self.tranAttLayer(temp_embed, predict_G)

        return res
