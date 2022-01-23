import torch
import torch.nn as nn
import torch.nn.functional as F

from .SpatAttLayer import SpatAttLayer


class GCRN(nn.Module):
    def __init__(self, num_nodes=361, hidden_dim=16):
        super(GCRN, self).__init__()

        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim

        self.num_dim = 1

        self.spat_embed_dim = int(self.num_dim * self.hidden_dim)     # Embedding dimension after spatial feature extraction
        self.temp_embed_dim = self.spat_embed_dim   # Embedding dimension after temporal feature extraction
        self.tran_embed_dim = self.temp_embed_dim   # Embedding dimension after transition projection

        # Spatial Attention Layer
        self.spatLayer_D = SpatAttLayer(feat_dim=1, hidden_dim=self.hidden_dim, num_heads=1, att=False, gate=False, merge='mean', num_dim=self.num_dim, cat_orig=False, use_pre_w=False)
        self.spatLayer_G = SpatAttLayer(feat_dim=self.num_nodes, hidden_dim=self.hidden_dim, num_heads=1, att=False, gate=False, merge='mean', num_dim=self.num_dim, cat_orig=False, use_pre_w=False)

        # Temporal Layer (GRU)
        self.tempLayer_D = nn.LSTM(input_size=self.spat_embed_dim, hidden_size=self.temp_embed_dim)
        self.tempLayer_G = nn.LSTM(input_size=self.spat_embed_dim, hidden_size=self.temp_embed_dim)

        # Transfer Function
        self.tran_d_l = nn.Linear(in_features=self.temp_embed_dim, out_features=1, bias=True)
        self.tran_g_l = nn.Linear(in_features=self.temp_embed_dim, out_features=self.tran_embed_dim, bias=True)

        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('linear')
        nn.init.xavier_normal_(self.tran_d_l.weight, gain=gain)
        nn.init.xavier_normal_(self.tran_g_l.weight, gain=gain)

    def forward(self, record_t: list):
        # Extract spatial features
        spat_embed_p = torch.stack([self.spatLayer(list(gs)) for gs in record_t])
        spat_embed_p = F.sigmoid(spat_embed_p)

        if spat_embed_p.device.type == 'cuda':
            torch.cuda.empty_cache()

        num_records, num_nodes = spat_embed_p.shape[0], spat_embed_p.shape[-2]

        # Extract temporal features
        o, (h, c) = self.tempLayer(spat_embed_p.reshape(num_records, -1, self.spat_embed_dim))
        temp_embed_p = h.reshape(-1, num_nodes, self.temp_embed_dim)
        del spat_embed_p
        del o
        del h
        del c

        if temp_embed_p.device.type == 'cuda':
            torch.cuda.empty_cache()

        # Transfer D
        res_D = self.tran_d_l(temp_embed_p).reshape(-1, num_nodes)

        # Transfer G
        reshaped_temp_embed_p = temp_embed_p.reshape(-1, num_nodes, 1, self.temp_embed_dim)
        gl = reshaped_temp_embed_p.repeat(1, 1, num_nodes, 1)
        gr = torch.transpose(reshaped_temp_embed_p, -3, -2).repeat(1, num_nodes, 1, 1)
        res_G = torch.sum(self.tran_g_l(gl) * gr, dim=-1)
        del reshaped_temp_embed_p
        del gl
        del gr

        del temp_embed_p

        return res_D, res_G

