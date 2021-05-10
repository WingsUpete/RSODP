import dgl
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    def __init__(self, query_dimension, embedding_dimension, merge='sum'):
        super(ScaledDotProductAttention, self).__init__()
        self.query_dimension = query_dimension
        self.embedding_dimension = embedding_dimension
        self.merge = merge

        # Query, Key, Value weights
        self.Wq = nn.Linear(self.query_dimension, self.embedding_dimension, bias=False)
        self.Wk = nn.Linear(self.embedding_dimension, self.embedding_dimension, bias=False)
        self.Wv = nn.Linear(self.embedding_dimension, self.embedding_dimension, bias=False)

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

        scores = torch.matmul(proj_Q, torch.transpose(proj_K, -2, -1)) / math.sqrt(self.embedding_dimension)
        norm_scores = F.softmax(scores, dim=1)

        output = torch.matmul(norm_scores, proj_V)
        return output

    def forward(self, query_feat, embedding_feat_list):
        embed_outputs = [self.apply_scaled_dot_product_attention(query_feat, embedding_feat) for embedding_feat in embedding_feat_list]
        if self.merge == 'sum':
            return torch.sum(torch.stack(embed_outputs), dim=0)
        elif self.merge == 'mean':
            return torch.mean(torch.stack(embed_outputs), dim=0)
        elif self.merge == 'cat':
            return torch.cat(embed_outputs, dim=1)
        else:   # Default: sum, as Gallat
            return torch.sum(torch.stack(embed_outputs), dim=0)


if __name__ == '__main__':
    print('hello world')
