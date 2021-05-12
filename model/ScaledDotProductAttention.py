import dgl
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    def __init__(self, query_dim, embed_dim, merge='sum'):
        super(ScaledDotProductAttention, self).__init__()
        self.query_dim = query_dim
        self.embed_dim = embed_dim
        self.merge = merge

        self.rooted_embed_dim = math.sqrt(self.embed_dim)

        # Query, Key, Value weights
        self.Wq = nn.Linear(self.query_dim, self.embed_dim, bias=False)
        self.Wk = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.Wv = nn.Linear(self.embed_dim, self.embed_dim, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.Wq.weight, gain=gain)
        nn.init.xavier_normal_(self.Wk.weight, gain=gain)
        nn.init.xavier_normal_(self.Wv.weight, gain=gain)

    def apply_scaled_dot_product_attention(self, query_feat, embedding_feat):
        proj_Q = self.Wq(query_feat)
        proj_K = self.Wk(embedding_feat)
        proj_V = self.Wv(embedding_feat)

        # Note that batch size is the first dimension, while the last two dimensions are the ones we care about.
        scores = torch.matmul(proj_Q, torch.transpose(proj_K, -2, -1)) / self.rooted_embed_dim
        norm_scores = F.softmax(scores, dim=-1)

        output = torch.matmul(norm_scores, proj_V)
        return output

    def forward(self, query_feat, embed_feat_list):
        embed_outputs = [self.apply_scaled_dot_product_attention(query_feat, embed_feat) for embed_feat in embed_feat_list]
        if self.merge == 'sum':
            return sum(embed_outputs)
        elif self.merge == 'mean':
            return torch.mean(sum(embed_outputs) / len(embed_outputs))
        elif self.merge == 'cat':
            return torch.cat(embed_outputs, dim=-1)
        else:   # Default: sum, as Gallat
            return sum(embed_outputs)
