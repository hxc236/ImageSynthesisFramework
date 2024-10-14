from data.base_dataset import BaseDataset, transform_resize, get_transform
from utils import load_path
from PIL import Image
import os
import torch as t
import numpy as np

class Monet2PhotoDataset(BaseDataset):
    def __init__(self, conf):
        self.dirA = os.path.join(conf.dataroot, conf.A)
        self.dirB = os.path.join(conf.dataroot, conf.B)

        self.A_path = load_path(self.dirA)
        self.B_path = load_path(self.dirB)

        self.transformA = get_transform(self.conf, grayscale=False)
        self.transformB = get_transform(self.conf, grayscale=False)

        self.transform_resize = transform_resize(self.conf)