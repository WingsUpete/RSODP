import numpy as np
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F

from .PwGaANLayer import MultiHeadPwGaANLayer


class SpatAttLayer(nn.Module):
    def __init__(self, feat_dim, hidden_dim, num_heads, gate=False, merge='cat'):
        super(SpatAttLayer, self).__init__()
        self.feat_dim = feat_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.gate = gate
        self.merge = merge

        self.fwdSpatAttLayer = MultiHeadPwGaANLayer(self.feat_dim, self.hidden_dim, self.num_heads, gate=self.gate, merge=self.merge)
        self.bwdSpatAttLayer = MultiHeadPwGaANLayer(self.feat_dim, self.hidden_dim, self.num_heads, gate=self.gate, merge=self.merge)
        self.geoSpatAttLayer = MultiHeadPwGaANLayer(self.feat_dim, self.hidden_dim, self.num_heads, gate=self.gate, merge=self.merge)
        self.proj_fc = nn.Linear(self.feat_dim, self.hidden_dim, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        """ Reinitialize learnable parameters. """
        gain = nn.init.calculate_gain('leaky_relu')
        nn.init.xavier_normal_(self.proj_fc.weight, gain=gain)

    def forward(self, fg: dgl.DGLGraph, bg: dgl.DGLGraph, gg: dgl.DGLGraph):
        feat = fg.ndata['v']
        feat = F.dropout(feat, 0.1)
        fg.ndata['v'] = feat
        bg.ndata['v'] = feat
        fg.ndata['v'] = feat

        proj_feat = self.proj_fc(feat)
        del feat

        fg.ndata['proj_z'] = proj_feat
        bg.ndata['proj_z'] = proj_feat
        gg.ndata['proj_z'] = proj_feat

        out_proj_feat = proj_feat.reshape(fg.batch_size, -1, self.hidden_dim)
        del proj_feat

        h_fwd = self.fwdSpatAttLayer(fg)
        h_bwd = self.bwdSpatAttLayer(bg)
        h_geo = self.geoSpatAttLayer(gg)

        h = torch.cat([out_proj_feat, h_fwd, h_bwd, h_geo], dim=-1)
        del out_proj_feat
        del h_fwd
        del h_bwd
        del h_geo

        return h


if __name__ == '__main__':
    """ Test: Remove dot in the package importing to avoid errors """
    GDVQ = np.load('test/GDVQ.npy', allow_pickle=True).item()
    V = GDVQ['V']
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
