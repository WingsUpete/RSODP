import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F


class PwGaANLayer(nn.Module):
    def __init__(self, in_dim, out_dim, gate=False):
        super(PwGaANLayer, self).__init__()
        # Shared Weight W_a for AttentionNet
        self.Wa = nn.Linear(in_dim, out_dim, bias=False)
        # AttentionNet outer linear layer
        self.att_out_fc = nn.Linear(2 * out_dim, 1, bias=False)
        # Head gate layer
        self.gate = gate
        if self.gate:
            self.gate_fc = nn.Linear(2 * in_dim + out_dim, 1, bias=False)
            self.Wg = nn.Linear(in_dim, out_dim, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        """ Reinitialize learnable parameters. """
        gain = nn.init.calculate_gain('relu')
        # gain = nn.init.calculate_gain('leaky_relu', 0.2)  # TODO: gain - leaky_relu with negative_slope=0.2
        nn.init.xavier_normal_(self.Wa.weight, gain=gain)
        nn.init.xavier_normal_(self.att_out_fc.weight, gain=gain)
        if self.gate:
            # TODO: gain - sigmoid
            nn.init.xavier_normal_(self.Wg.weight, gain=gain)
            nn.init.xavier_normal_(self.gate_fc.weight, gain=gain)

    def edge_attention(self, edges):
        z_comb = torch.cat([edges.data['pre_w'] * edges.src['z'], edges.dst['z']], dim=1)
        a = self.att_out_fc(z_comb)
        return {'e': F.leaky_relu(a)}

    def message_func(self, edges):
        """ Specify messages to be propagated along edges """
        # The messages will be sent to the mailbox
        # mailbox['proj_z']: z->x, so we need z's projected features
        # mailbox['e']: z->x has a e for attention calculation
        if self.gate:
            pwFeat = edges.data['pre_w'] * edges.src['v']
            return {'proj_z': edges.src['proj_z'], 'e': edges.data['e'], 'pre_v_g': pwFeat}
        else:
            return {'proj_z': edges.src['proj_z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        """ Specify how messages are processed and propagated to nodes """
        # Aggregate features to nodes
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['proj_z'], dim=1)

        # head gates
        if self.gate:
            pwFeat = nodes.mailbox['pre_v_g']
            gateProj = self.Wg(pwFeat)
            maxFeat = torch.max(gateProj, dim=1)[0]
            meanFeat = torch.mean(pwFeat, dim=1)
            gComb = torch.cat([nodes.data['v'], maxFeat, meanFeat], dim=1)
            gFCVal = self.gate_fc(gComb)
            gVal = torch.sigmoid(gFCVal)
            h = gVal * h
            test1 = gFCVal.detach().numpy()
            test2 = gVal.detach().numpy()

        return {'h': h}

    def forward(self, g: dgl.DGLGraph):
        with g.local_scope():
            feat = g.ndata['v']

            # Wa: shared attention to features v (or h for multiple GAT layers)
            z = self.Wa(feat)
            g.ndata['z'] = z

            # AttentionNet
            g.apply_edges(self.edge_attention)
            # Message Passing
            g.update_all(self.message_func, self.reduce_func)
            return g.ndata['h']


class MultiHeadPwGaANLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, merge='cat', gate=False):
        super(MultiHeadPwGaANLayer, self).__init__()
        self.gate = gate
        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(PwGaANLayer(in_dim, out_dim, self.gate))
        self.merge = merge

    def forward(self, g: dgl.DGLGraph):
        head_outs = [attn_head(g) for attn_head in self.heads]
        if self.merge == 'cat':
            return torch.cat(head_outs, dim=1)
        else:
            return torch.mean(torch.stack(head_outs))


if __name__ == '__main__':
    """ Test """
    pass
