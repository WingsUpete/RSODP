import torch
import torch.nn as nn

from .SpatAttLayer import SpatAttLayer
from .TempAttLayer import TempAttLayer
from .TranAttLayer import TranAttLayer

from Config import TEMP_FEAT_NAMES


class GallatExt(nn.Module):
    def __init__(self, feat_dim=43, query_dim=41, hidden_dim=16, num_heads=3, num_dim=3):
        super(GallatExt, self).__init__()
        self.num_heads = num_heads

        self.feat_dim = feat_dim
        self.query_dim = query_dim
        self.hidden_dim = hidden_dim

        self.num_dim = num_dim

        self.spat_embed_dim = int((self.num_dim + 1) * self.hidden_dim)     # Embedding dimension after spatial feature extraction
        self.temp_embed_dim = self.spat_embed_dim   # Embedding dimension after temporal feature extraction

        # Spatial Attention Layer
        self.spatAttLayer = SpatAttLayer(feat_dim=self.feat_dim, hidden_dim=self.hidden_dim, num_heads=self.num_heads, gate=True, merge='mean', num_dim=self.num_dim, cat_orig=True, use_pre_w=True)

        # Temporal Attention Layer
        self.tempAttLayer = TempAttLayer(query_dim=self.query_dim, embed_dim=self.spat_embed_dim, rec_merge='mean', comb_merge='mean')

        # Transferring Attention Layer
        self.tranAttLayer = TranAttLayer(embed_dim=self.temp_embed_dim, activate_function_method='relu')

    def forward(self, record, query, ref_D=None, ref_G=None, predict_G=False, ref_extent=0.2):
        # Extract spatial features
        spat_embed_dict = {}
        for temp_feat in TEMP_FEAT_NAMES:
            spat_embed_dict[temp_feat] = [self.spatAttLayer(list(gs)) for gs in record[temp_feat]]

        if query.device.type == 'cuda':
            torch.cuda.empty_cache()

        # Extract temporal features
        temp_embed = self.tempAttLayer(query, spat_embed_dict)

        if query.device.type == 'cuda':
            torch.cuda.empty_cache()

        # Transferring features to perform predictions
        res = self.tranAttLayer(temp_embed, predict_G, ref_D, ref_G, ref_extent)

        return res


class GallatExtFull(nn.Module):
    def __init__(self, feat_dim=43, query_dim=41, hidden_dim=16, num_heads=3, num_dim=3):
        super(GallatExtFull, self).__init__()
        self.num_heads = num_heads

        self.feat_dim = feat_dim
        self.query_dim = query_dim
        self.hidden_dim = hidden_dim

        self.num_dim = num_dim

        self.spat_embed_dim = int((self.num_dim * self.num_heads + 1) * self.hidden_dim)    # Embedding dimension after spatial feature extraction
        self.temp_embed_dim = int(4 * self.spat_embed_dim)    # Embedding dimension after temporal feature extraction

        # Spatial Attention Layer
        self.spatAttLayer = SpatAttLayer(feat_dim=self.feat_dim, hidden_dim=self.hidden_dim, num_heads=self.num_heads, gate=True, merge='cat', num_dim=self.num_dim, cat_orig=True, use_pre_w=True)

        # Temporal Attention Layer
        self.tempAttLayer = TempAttLayer(query_dim=self.query_dim, embed_dim=self.spat_embed_dim, rec_merge='mean', comb_merge='cat')

        # Transferring Attention Layer
        self.tranAttLayer = TranAttLayer(embed_dim=self.temp_embed_dim, activate_function_method=None)

    def forward(self, record, query, ref_D=None, ref_G=None, predict_G=False, ref_extent=0.2):
        # Extract spatial features
        spat_embed_dict = {}
        for temp_feat in TEMP_FEAT_NAMES:
            spat_embed_dict[temp_feat] = [self.spatAttLayer(list(gs)) for gs in record[temp_feat]]

        if query.device.type == 'cuda':
            torch.cuda.empty_cache()

        # Extract temporal features
        temp_embed = self.tempAttLayer(query, spat_embed_dict)

        if query.device.type == 'cuda':
            torch.cuda.empty_cache()

        # Transferring features to perform predictions
        res = self.tranAttLayer(temp_embed, predict_G, ref_D, ref_G, ref_extent)

        return res
