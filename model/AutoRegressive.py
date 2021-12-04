import torch
import torch.nn as nn


class AR(nn.Module):
    def __init__(self, p):
        super(AR, self).__init__()
        self.p = p
        self.feed_forward = nn.Linear(in_features=p, out_features=1, bias=True)

    def forward(self, x):
        return self.feed_forward(x)


if __name__ == '__main__':
    ar = AR(p=7)
    inputs = torch.stack([torch.ones(5, 361) for i in range(7)], dim=-1)
    print(inputs.shape)
    output = ar(inputs)
    print(output.shape)
    print(output)
