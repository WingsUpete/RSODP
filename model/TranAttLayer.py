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

        self.reset_parameters(nonlinearity='sigmoid')

    def reset_parameters(self, nonlinearity):
        gain = nn.init.calculate_gain(nonlinearity)
        nn.init.xavier_normal_(self.demand_fc.weight, gain=gain)

    def forward(self, embed_feat):
        # Predict demands
        demands = self.demand_fc(embed_feat)
        demands = self.activate_function(demands)

        # TODO: Predict Request Graph

        return demands
