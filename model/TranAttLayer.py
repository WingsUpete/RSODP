import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F


class TranAttLayer(nn.Module):
    def __init__(self, embed_dim, activate_function_method='sigmoid'):
        super(TranAttLayer, self).__init__()
        self.embed_dim = embed_dim

        # Demand Prediction Linear Layer + Activation
        self.demand_fc = nn.Linear(self.embed_dim, 1, bias=True)

        self.activate_function_method = activate_function_method
        self.activate_function = nn.Sigmoid()
        gain_val = nn.init.calculate_gain('sigmoid')
        if self.activate_function_method == 'sigmoid':
            self.activate_function = nn.Sigmoid()
            gain_val = nn.init.calculate_gain('sigmoid')
        elif self.activate_function_method == 'relu':
            self.activate_function = nn.ReLU()
            gain_val = nn.init.calculate_gain('relu')
        elif self.activate_function_method == 'leaky_relu':
            self.activate_function = nn.LeakyReLU()
            gain_val = nn.init.calculate_gain('leaky_relu')
        elif self.activate_function_method == 'selu':
            self.activate_function = nn.SELU()
            gain_val = 0.75
        else:   # Do not use activation
            self.activate_function = None
            gain_val = nn.init.calculate_gain('relu')

        # Shared Weight W_a for AttentionNet
        self.Wa = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        # AttentionNet outer linear layer
        # split two fc to avoid concat
        self.att_out_fc_l = nn.Linear(self.embed_dim, 1, bias=False)
        self.att_out_fc_r = nn.Linear(self.embed_dim, 1, bias=False)

        self.reset_parameters(gain=gain_val)

    def reset_parameters(self, gain):
        # Demand
        nn.init.xavier_normal_(self.demand_fc.weight, gain=gain)

        # Attention
        gain = nn.init.calculate_gain('leaky_relu')
        nn.init.xavier_normal_(self.Wa.weight, gain=gain)
        nn.init.xavier_normal_(self.att_out_fc_l.weight, gain=gain)
        nn.init.xavier_normal_(self.att_out_fc_r.weight, gain=gain)

    def predict_request_graphs(self, embed_feat, demands, ref_G=None):
        num_nodes = embed_feat.shape[-2]
        proj_embed_feat = self.Wa(embed_feat)

        # Apply Attention: Use repeat to expand the features
        el = self.att_out_fc_l(proj_embed_feat)
        er = self.att_out_fc_r(proj_embed_feat)
        el_exp = el.repeat(1, 1, num_nodes)
        er_exp = torch.transpose(er, -2, -1).repeat(1, num_nodes, 1)
        A = el_exp + er_exp
        del proj_embed_feat
        del el
        del er
        del el_exp
        del er_exp

        A = F.leaky_relu(A)
        Q = F.softmax(A, dim=-1)
        Q = F.dropout(Q, 0.1)
        del A

        if ref_G is not None:
            norm_ref_G = F.normalize(ref_G, p=1.0, dim=-1)
            Q = (Q + norm_ref_G) / 2

        # Expand D as well
        rel_D = demands.repeat(1, 1, num_nodes)

        # Get graph
        G = Q * rel_D
        del Q
        del rel_D
        del num_nodes

        return G

    def forward(self, embed_feat, predict_G, ref_D=None, ref_G=None):
        num_nodes = embed_feat.shape[-2]

        # Predict demands
        demands = self.demand_fc(embed_feat)
        if self.activate_function:
            demands = self.activate_function(demands).clone()
        demands_out = demands.reshape(-1, num_nodes)
        if ref_D is not None:   # scale
            demands_out *= ref_D
        demands = demands_out.reshape(-1, num_nodes, 1)
        del num_nodes

        if predict_G:
            # Predict Request Graph
            req_gs = self.predict_request_graphs(embed_feat, demands, ref_G=ref_G)
            del demands
            return demands_out, req_gs
        else:
            return demands_out, None
