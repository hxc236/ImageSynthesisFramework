from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import torch
from torch import nn
#
# transforms = transforms.Compose([
#     # transforms.Resize((256, 256)),
#     transforms.ToTensor(),
#     # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
# ])
#
# img_path = r"F:\ML\Dataset\经典风格迁移数据集\monet2photo\trainA\00001.jpg"
#
# img = Image.open(img_path).convert('RGB')
# img = transforms(img)
# print(img.shape)
#


# from models.base_model import *
#
# timesteps = torch.randn(5)
# print("输入的时间步:", timesteps)
#
# # 设定输出嵌入的维度
# dim = 16
#
# # 调用timestep_embedding函数获取时间步嵌入
# embedding = timestep_embedding(timesteps, dim)
# print("输出的时间步嵌入形状:", embedding.shape)
# print("输出的时间步嵌入:", embedding)
#
#
if __name__ == '__main__':
    x = torch.randn(1, 3, 64)
    print(x.shape)
    x = x.permute(0, 2, 1)
    linear1 = nn.Linear( 3, 3)

    y = linear1(x)
    print(y.shape)