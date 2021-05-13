import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F

from .ScaledDotProductAttention import ScaledDotProductAttention

TEMP_FEAT_NAMES = ['St', 'Sp', 'Stpm', 'Stpp']


class TempAttLayer(nn.Module):
    def __init__(self, query_dim, embed_dim, rec_merge='sum', comb_merge='sum'):
        super(TempAttLayer, self).__init__()
        self.query_dim = query_dim
        self.embed_dim = embed_dim
        self.rec_merge = rec_merge      # merging method for historical record attention
        self.comb_merge = comb_merge    # merging method for combination attention

        # Scaled Dot Product Attention
        self.recScaledDotProductAttention = ScaledDotProductAttention(self.query_dim, self.embed_dim, merge=self.rec_merge)
        self.combScaledDotProductAttention = ScaledDotProductAttention(self.query_dim, self.embed_dim, merge=self.comb_merge)

        # Dropout
        self.recDropout = nn.Dropout(0.2)
        self.combDropout = nn.Dropout(0.2)

        # Normalization
        self.recNorm = nn.LayerNorm(self.embed_dim)
        self.combNorm = nn.LayerNorm(self.embed_dim)

    def forward(self, query_feat, embed_feat_dict):
        rec_embed_list = [self.recNorm(self.recDropout(self.recScaledDotProductAttention(query_feat, embed_feat_dict[temp_feat]))) for temp_feat in TEMP_FEAT_NAMES]
        comb_embed = self.combNorm(self.combDropout(self.combScaledDotProductAttention(query_feat, rec_embed_list)))
        return comb_embed
