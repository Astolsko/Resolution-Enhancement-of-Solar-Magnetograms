import torch.nn as nn
from conv_layer import conv_layer
from RFLB import RLFB

class MESR(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, num_blocks=12, esa_channels=16):
        super(MESR, self).__init__()

        self.conv_in = conv_layer(in_channels, mid_channels, 3)

        # 12 RLFB blocks
        self.rlfb_blocks = nn.Sequential(*[RLFB(mid_channels, esa_channels=esa_channels) for _ in range(num_blocks)])

        self.conv_out = conv_layer(mid_channels, out_channels, 3)


    def forward(self, x):
        
        out_conv_in = self.conv_in(x)

        out_rlfb = self.rlfb_blocks(out_conv_in)

        # Skip connection
        out_skip = out_rlfb + out_conv_in

        out = self.conv_out(out_skip)

        return out

# example usage
model = MESR(in_channels=3, mid_channels=64, out_channels=3, num_blocks=12)
