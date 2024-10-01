import torch
import torch.nn as nn
from RFLB import RFLB

class SubPixelConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, upscale_factor=2):
        super(SubPixelConvBlock, self).__init__()
        
        # Sub-Pixel Convolutional Layer
        self.conv = nn.Conv2d(in_channels, out_channels * (upscale_factor ** 2), kernel_size=3, stride=1, padding=1)
        
        # Sub-pixel reorganization
        self.subpixel = nn.PixelShuffle(upscale_factor)
        
        # Optional activation (ReLU)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.subpixel(x)
        x = self.relu(x)
        return x

class ImageEnhancementModel(nn.Module):
    def __init__(self, in_channels=3, num_rlfb_blocks=4, upscale_factor=2):
        super(ImageEnhancementModel, self).__init__()
        
        # Initial Conv layer
        self.initial_conv = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1)
        
        # RLFB blocks (assuming Residual Learning Feature Blocks or similar)
        self.rl_blocks = nn.Sequential(*[RFLB(64) for _ in range(num_rlfb_blocks)]) # Assuming RLFB is a pre-defined block
        
        # Final convolution layer before sub-pixel block
        self.final_conv = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        
        # Sub-pixel convolution block (replaces PixelShuffle)
        self.sub_pixel_conv_block = SubPixelConvBlock(64, in_channels, upscale_factor)
    
    def forward(self, x):
        x_initial = self.initial_conv(x)  # Initial Conv layer
        x = self.rl_blocks(x_initial)     # RLFB blocks
        x = self.final_conv(x)            # Final conv before sub-pixel block
        x = x + x_initial                 # Skip connection
        x = self.sub_pixel_conv_block(x)  # Sub-pixel convolution
        return x