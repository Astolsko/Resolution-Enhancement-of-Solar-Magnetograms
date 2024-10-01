import torch.nn as nn
from conv_layer import conv_layer
from activation import activation
from ESA import ESA

class RLFB(nn.Module):
    def __init__(self, in_channels, mid_channels=None, out_channels=None, esa_channels=16):
        super(RLFB, self).__init__()

        if mid_channels is None:
            mid_channels = in_channels
        if out_channels is None:
            out_channels = in_channels

        self.c1_r = conv_layer(in_channels, mid_channels, 3)
        self.c2_r = conv_layer(mid_channels, mid_channels, 3)
        self.c3_r = conv_layer(mid_channels, in_channels, 3)

        self.c5 = conv_layer(in_channels, out_channels, 1)
        self.esa = ESA(esa_channels, out_channels, nn.Conv2d)

        self.act = activation('silu')

    def forward(self, x):
        out = self.c1_r(x)
        out = self.act(out)

        out = self.c2_r(out)
        out = self.act(out)

        out = self.c3_r(out)
        out = self.act(out)

        out = out + x
        out = self.esa(self.c5(out))

        return out
