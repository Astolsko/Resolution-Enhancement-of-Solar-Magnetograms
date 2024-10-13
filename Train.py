import os

import torch.nn as nn
import torch
from torchsummary import summary
import torch.optim as optim
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, random_split
import random

# from our codebase
from conv_layer import conv_layer
from RLFB import RLFB
from SUBP import SubPixelConvBlock  
from Trainning_Loop import train_model, CharbonnierLoss


class MESR(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, num_blocks=12, esa_channels=16, upscale_factor=2):
        super(MESR, self).__init__()

        self.conv_in = conv_layer(in_channels, mid_channels, 3)
        self.RLFB_blocks = nn.Sequential(*[RLFB(mid_channels, esa_channels=esa_channels) for _ in range(num_blocks)])
        self.conv_out = conv_layer(mid_channels, out_channels, 3)
        self.sub_pixel_conv = SubPixelConvBlock(out_channels, out_channels, upscale_factor=upscale_factor)

    def forward(self, x):
        out_conv_in = self.conv_in(x)  
        out_RLFB = self.RLFB_blocks(out_conv_in)  
        out_skip = out_RLFB + out_conv_in  
        out = self.conv_out(out_skip)  
        out = self.sub_pixel_conv(out)  
        return out


def model_summary(model, device):
    model.to(device)
    summary(model, input_size=(3, 256, 256)) # Change order & num of channels to match grayscale channel 


class SuperResolutionDataset(Dataset):
    def __init__(self, lr_dir, hr_dir, transform=None):
        self.lr_dir = lr_dir
        self.hr_dir = hr_dir
        self.lr_images = os.listdir(lr_dir)
        self.transform = transform

    def __len__(self):
        return len(self.lr_images)

    def __getitem__(self, idx):
        lr_image_path = os.path.join(self.lr_dir, self.lr_images[idx])
        hr_image_path = os.path.join(self.hr_dir, self.lr_images[idx])  # Assuming same naming

        lr_image = Image.open(lr_image_path).convert("RGB")
        hr_image = Image.open(hr_image_path).convert("RGB")

        if self.transform:
            lr_image = self.transform(lr_image)
            hr_image = self.transform(hr_image)

        return {'image': lr_image, 'label': hr_image}


def dataloaders(train_dataset, val_dataset, batch_size=2):
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def setup_training(model, device, train_loader, val_loader, epochs=20, patience=5):
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_function = CharbonnierLoss(epsilon=1e-6)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_function=loss_function,
        device=device,
        epochs=epochs,
        patience=patience,
        val_interval=1,
        lr_scheduler=lr_scheduler,
        output_dir="./model_output"  # Specify the output directory
    )


def main():
    
    # Check if CUDA is available
    if torch.cuda.is_available():
        print("CUDA is available!")
        print(f"Number of available GPUs: {torch.cuda.device_count()}")
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    else:
        print("CUDA is not available.")
    torch.cuda.empty_cache()


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MESR(in_channels=3, mid_channels=64, out_channels=3, num_blocks=12)

    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize to desired dimensions
        transforms.RandomHorizontalFlip(p=0.5),  
        transforms.RandomVerticalFlip(p=0.5),    
        transforms.RandomRotation(90),           
        transforms.ToTensor(),             
        
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Normalize
    ]   )


    # load the dataset
    lr_dir = "/home/user/Desktop/Abul Hasan/Dataset/LRHR dataset/renamedsoho"
    hr_dir = "/home/user/Desktop/Abul Hasan/Dataset/LRHR dataset/renamedsdo"

    # dataset instance
    full_dataset = SuperResolutionDataset(lr_dir=lr_dir, hr_dir=hr_dir, transform=transform)

    # #splitting the dataset
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Use dataloaders function to get train and validation loaders
    train_loader, val_loader = dataloaders(train_dataset, val_dataset, batch_size=32)

    model_summary(model, device)
    setup_training(model, device, train_loader, val_loader)

main()
