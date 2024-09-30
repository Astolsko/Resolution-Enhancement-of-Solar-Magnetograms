import torch
import torch.nn as nn
from conv_layer import conv_layer
from RFLB import RLFB
from torchsummary import summary

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

# Function to print summary
def print_model_summary(device):
    # Instantiate the MESR model
    model = MESR(in_channels=3, mid_channels=64, out_channels=3, num_blocks=12)
    model.to(device)
    # Assuming input image size is (3, 256, 256) for RGB images with height and width 256
    summary(model, input_size=(3, 256, 256))


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_model_summary(device)
