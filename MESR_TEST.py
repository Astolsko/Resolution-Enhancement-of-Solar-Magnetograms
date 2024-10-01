import torch.nn as nn
from conv_layer import conv_layer
from RLFB import RLFB
from SUBP import SubPixelConvBlock  
from torchsummary import summary
import torch

class MESR(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, num_blocks=12, esa_channels=16, upscale_factor=2):
        super(MESR, self).__init__()

        self.conv_in = conv_layer(in_channels, mid_channels, 3)

        # 12 RLFB blocks
        self.RLFB_blocks = nn.Sequential(*[RLFB(mid_channels, esa_channels=esa_channels) for _ in range(num_blocks)])

        self.conv_out = conv_layer(mid_channels, out_channels, 3)

        # Sub-pixel convolution block
        self.sub_pixel_conv = SubPixelConvBlock(out_channels, out_channels, upscale_factor=upscale_factor)

    def forward(self, x):
        out_conv_in = self.conv_in(x)  # First convolution layer
        out_RLFB = self.RLFB_blocks(out_conv_in)  # RLFB blocks
        
        # Skip connection
        out_skip = out_RLFB + out_conv_in

        out = self.conv_out(out_skip) 

        out = self.sub_pixel_conv(out) # assumption : upscaling actor is 2

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