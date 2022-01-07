import torch
import torch.nn as nn

import Config


class LSTNet(nn.Module):
    def __init__(self, p, refAR):
        super(LSTNet, self).__init__()
        self.p = p
        self.refAR = refAR

        self.L = 2 * p

        # stConv
        self.l_stConv_last_D = nn.Linear(in_features=1, out_features=1, bias=True)
        self.l_stConv_current_D = nn.Linear(in_features=1, out_features=1, bias=True)
        self.l_stConv_last_G = nn.Linear(in_features=1, out_features=1, bias=True)
        self.l_stConv_current_G = nn.Linear(in_features=1, out_features=1, bias=True)

        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('linear')
        nn.init.xavier_normal_(self.l_stConv_last_D.weight, gain=gain)
        nn.init.xavier_normal_(self.l_stConv_current_D.weight, gain=gain)
        nn.init.xavier_normal_(self.l_stConv_last_G.weight, gain=gain)
        nn.init.xavier_normal_(self.l_stConv_current_G.weight, gain=gain)

    def forward(self, recordGD):
        padDs = torch.stack(
            [torch.zeros(recordGD[Config.LSTNET_TEMP_FEAT][0][0].shape, device=recordGD[Config.LSTNET_TEMP_FEAT][0][0].device)] +
            [recordGD[Config.LSTNET_TEMP_FEAT][i][0] for i in range(len(recordGD[Config.LSTNET_TEMP_FEAT]))]
        )
        padDs = padDs.reshape(padDs.shape + (1,))
        padGs = torch.stack(
            [torch.zeros(recordGD[Config.LSTNET_TEMP_FEAT][0][1].shape, device=recordGD[Config.LSTNET_TEMP_FEAT][0][1].device)] +
            [recordGD[Config.LSTNET_TEMP_FEAT][i][1] for i in range(len(recordGD[Config.LSTNET_TEMP_FEAT]))]
        )
        padGs = padGs.reshape(padGs.shape + (1,))

        # stConv
        stConvD = self.l_stConv_last_D(padDs[:self.L]) + self.l_stConv_current_D(padDs[1:])
        stConvG = self.l_stConv_last_G(padGs[:self.L]) + self.l_stConv_current_G(padGs[1:])
        del padDs
        del padGs

        # Evaluate refAR
        self.refAR.eval()
        resAR = self.refAR(recordGD)

        res = resAR

        return res
