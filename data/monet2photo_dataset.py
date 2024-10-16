from data.base_dataset import BaseDataset, transform_resize, get_transform
from data.utils import load_path
from PIL import Image
import os
import torch as t
import numpy as np

class Monet2PhotoDataset(BaseDataset):
    def __init__(self, conf):
        BaseDataset.__init__(self, conf)
        self.dirA = os.path.join(conf.dataroot, conf.A)
        self.dirB = os.path.join(conf.dataroot, conf.B)

        self.A_path = load_path(self.dirA)
        self.B_path = load_path(self.dirB)

        self.len_A = len(self.A_path)
        self.len_B = len(self.B_path)

        self.transform_A = get_transform(self.conf, grayscale=False)
        self.transform_B = get_transform(self.conf, grayscale=False)

        self.transform_resize = transform_resize(self.conf)

    def __len__(self):
        return max(self.len_A, self.len_B)
    
    def __getitem__(self, idx):
        A_path = self.A_paths[idx % self.len_A]
        name = os.path.basename(A_path)         # 确保A、B文件名一致
        #
        # B_path = os.path.join(self.dir_B, name)
        B_path = self.B_paths[idx % self.len_B]

        A_img = np.load(A_path)
        B_img = np.load(B_path)

        A_img = A_img / A_img.max() * 255
        B_img = B_img / B_img.max() * 255
        A_img = Image.fromarray(np.uint8(A_img)).convert('RGB')
        B_img = Image.fromarray(np.uint8(B_img)).convert('RGB')

        A = self.transform_A(A_img)
        B = self.transform_B(B_img)

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path, 'name': name}
