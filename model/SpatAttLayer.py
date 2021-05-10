import numpy as np
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F

from .PwGaANLayer import MultiHeadPwGaANLayer


class SpatAttLayer(nn.Module):
    def __init__(self, feat_dim, hidden_dim, num_heads, gate=False):
        super(SpatAttLayer, self).__init__()
        self.gate = gate
        self.fwdSpatAttLayer = MultiHeadPwGaANLayer(feat_dim, hidden_dim, num_heads, gate=self.gate)
        self.bwdSpatAttLayer = MultiHeadPwGaANLayer(feat_dim, hidden_dim, num_heads, gate=self.gate)
        self.geoSpatAttLayer = MultiHeadPwGaANLayer(feat_dim, hidden_dim, num_heads, gate=self.gate)
        self.proj_fc = nn.Linear(feat_dim, hidden_dim, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        """ Reinitialize learnable parameters. """
        gain = nn.init.calculate_gain('relu')
        # gain = nn.init.calculate_gain('leaky_relu', 0.2)  # TODO: gain - leaky_relu with negative_slope=0.2
        nn.init.xavier_normal_(self.proj_fc.weight, gain=gain)

    def forward(self, fg: dgl.DGLGraph, bg: dgl.DGLGraph, gg: dgl.DGLGraph):
        feat = fg.ndata['v']
        proj_feat = self.proj_fc(feat)
        fg.ndata['proj_z'] = proj_feat
        bg.ndata['proj_z'] = proj_feat
        gg.ndata['proj_z'] = proj_feat
        h_fwd = self.fwdSpatAttLayer(fg)
        h_bwd = self.bwdSpatAttLayer(bg)
        h_geo = self.geoSpatAttLayer(gg)
        h = torch.cat([proj_feat, h_fwd, h_bwd, h_geo], dim=1)
        return h


if __name__ == '__main__':
    """ Test """
    GVQ = np.load('test/GVQ.npy', allow_pickle=True).item()
    G, V, Q = GVQ['G'], GVQ['V'], GVQ['Q']
    (dfg, dbg,), _ = dgl.load_graphs('test/FBGraphs.dgl')
    (dgg,), _ = dgl.load_graphs('test/GeoGraph.dgl')
    V = torch.from_numpy(V)

    spatAttLayer = SpatAttLayer(feat_dim=7, hidden_dim=16, num_heads=3, gate=True)
    print(V, V.shape)
    dfg.ndata['v'] = V
    dbg.ndata['v'] = V
    dgg.ndata['v'] = V
    out = spatAttLayer(dfg, dbg, dgg)
    print(out, out.shape)
    test = out.detach().numpy()

