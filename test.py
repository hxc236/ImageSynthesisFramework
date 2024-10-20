from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import torch
from torch import nn

transforms = transforms.Compose([
    # transforms.Resize((256, 256)),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

img_path = r"F:\ML\Dataset\经典风格迁移数据集\monet2photo\trainA\00001.jpg"

img = Image.open(img_path).convert('RGB')
img = transforms(img)
print(img.shape)

