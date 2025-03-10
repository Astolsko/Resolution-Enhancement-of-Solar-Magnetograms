import torch
from PIL import Image
from torchvision import transforms as transform

# Predict function
def prediction(model, image_path):
    model.eval()
    image = Image.open(image_path).convert('L')
    image = transform(image).unsqueeze(0).to('cuda')
    with torch.no_grad():
        output = model(image)
    return output.squeeze(0).cpu()