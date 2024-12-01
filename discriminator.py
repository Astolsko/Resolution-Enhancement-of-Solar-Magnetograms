import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torchinfo import summary


input_channels = 3
feature_map_size = 64
ngpu = 1

device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

def weights_init(m):
    classname = m.__class__.__name__

    # print("The class name is : ",classname)
    # print("Finding if it contains 'Conv' : ", classname.find('Conv'))
    # print("Finding if it contains 'BatchNorm' : ",classname.find('BatchNorm'))

    if classname.find('Conv') != -1:
        # 0.0 is the mean of the normal distribution and 0.2 is the standard deviation
        nn.init.normal_(m.weight.data, 0.0, 0.02) 
    elif classname.find('BatchNorm') != -1:
        # Batch normalization weights are often initialized close to 1.0 to start with almost no scaling effect.
        nn.init.normal_(m.weight.data, 1.0, 0.02) 
        nn.init.constant_(m.bias.data, 0)


class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input size. ``(input_channels) x 32 x 32``
            nn.Conv2d(input_channels, feature_map_size, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(feature_map_size) x 16 x 16``

            nn.Conv2d(feature_map_size, feature_map_size * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_map_size * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(feature_map_size*2) x 8 x 8``

            nn.Conv2d(feature_map_size * 2, feature_map_size * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_map_size * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(feature_map_size*4) x 4 x 4``

            nn.Conv2d(feature_map_size * 4, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

block = Discriminator(ngpu).to(device)

# multi gpu handling
if (device.type == 'cuda') and (ngpu > 1):
    block = nn.DataParallel(block, list(range(ngpu)))

block.apply(weights_init)
summary(block, input_size=(1, 3, 32, 32))