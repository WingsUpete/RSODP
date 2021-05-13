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
        gain_str = 'sigmoid'
        if self.activate_function_method == 'sigmoid':
            self.activate_function = nn.Sigmoid()
            gain_str = 'sigmoid'

        # Shared Weight W_a for AttentionNet
        self.Wa = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        # AttentionNet outer linear layer
        # split two fc to avoid concat
        self.att_out_fc_l = nn.Linear(self.embed_dim, 1, bias=False)
        self.att_out_fc_r = nn.Linear(self.embed_dim, 1, bias=False)

        self.reset_parameters(nonlinearity='sigmoid')

    def reset_parameters(self, nonlinearity):
        gain = nn.init.calculate_gain(nonlinearity)
        nn.init.xavier_normal_(self.demand_fc.weight, gain=gain)

        # Attention
        gain = nn.init.calculate_gain('leaky_relu', 0.2)
        nn.init.xavier_normal_(self.Wa.weight, gain=gain)
        nn.init.xavier_normal_(self.att_out_fc_l.weight, gain=gain)
        nn.init.xavier_normal_(self.att_out_fc_r.weight, gain=gain)

    def predict_request_graphs(self, embed_feat, demands):
        num_nodes = embed_feat.shape[-2]
        proj_embed_feat = self.Wa(embed_feat)

        # Apply Attention: Use repeat to expand the features
        el = self.att_out_fc_l(proj_embed_feat)
        er = self.att_out_fc_r(proj_embed_feat)
        el_exp = el.repeat(1, 1, num_nodes)
        er_exp = torch.transpose(er, -2, -1).repeat(1, num_nodes, 1)
        A = el_exp + er_exp

        A = F.leaky_relu(A)
        Q = F.softmax(A, dim=-1)

        # Expand D as well
        rel_D = demands.repeat(1, 1, num_nodes)

        # Get graph
        G = Q * rel_D

        return G

    def forward(self, embed_feat, predict_G):
        num_nodes = embed_feat.shape[-2]

        # Predict demands
        demands = self.demand_fc(embed_feat)
        demands = self.activate_function(demands)
        demands_out = demands.reshape(-1, num_nodes)

        if predict_G:
            # Predict Request Graph
            req_gs = self.predict_request_graphs(embed_feat, demands)
            return demands_out, req_gs
        else:
            return demands_out, None
