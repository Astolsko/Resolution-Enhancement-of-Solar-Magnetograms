import torch
import torch.nn as nn
from torch import nn
import torch.nn.init as init

class convolutions(nn.Module): # 3 by 3 convolutions
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_op = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.weights_init()

    def weights_init(self): # he initialization aka kaiming initializer
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                # bias to zeros
                if m.bias is not None:
                    init.zeros_(m.bias)

    def forward(self, x):
        return self.conv_op(x)
    
class encode(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = convolutions(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        down = self.conv(x)
        p = self.pool(down)

        return down, p
    
class decode(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride=2)
        self.conv = convolutions(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x1, x2], 1)
        return self.conv(x)
    
class UNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.down_convolution_1 = encode(in_channels, 32)
        self.down_convolution_2 = encode(32, 64)
        self.down_convolution_3 = encode(64, 128)
        self.down_convolution_4 = encode(128, 256)

        self.bottle_neck = convolutions(256, 512)

        self.up_convolution_1 = decode(512, 256)
        self.up_convolution_2 = decode(256, 128)
        self.up_convolution_3 = decode(128, 64)
        self.up_convolution_4 = decode(64, 32)

        self.out = nn.Conv2d(in_channels=32, out_channels=num_classes, kernel_size=1)

    def forward(self, x):
        down_1, p1 = self.down_convolution_1(x)
        down_2, p2 = self.down_convolution_2(p1)
        down_3, p3 = self.down_convolution_3(p2)
        down_4, p4 = self.down_convolution_4(p3)

        b = self.bottle_neck(p4)

        up_1 = self.up_convolution_1(b, down_4)
        up_2 = self.up_convolution_2(up_1, down_3)
        up_3 = self.up_convolution_3(up_2, down_2)
        up_4 = self.up_convolution_4(up_3, down_1)

        out = self.out(up_4)
        return out
