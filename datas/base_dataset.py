import random
import numpy as np
import torchvision.transforms as transforms
import torch.utils.data as data
from abc import ABC, abstractmethod
from PIL import Image

class BaseDataset(data.Dataset, ABC):
    """This class is an abstract base class (ABC) for datasets.
    To create a subclass, you need to implement the following four functions:
    1. initialize the dataset by loading the data (self.initialize(self.opt))
    2. get_image_paths and get_label_paths
    3. __getitem__
    4. __len__
    """
    def __init__(self, conf):
       self.conf = conf
       self.root = conf.root

    @abstractmethod
    def __len__(self):
        """Return the total number of images in the dataset."""
        pass

    @abstractmethod
    def __getitem__(self, idx):
        """Return a data point and its metadata information.
        Parameters:
            index - - a random integer for data indexing
        Returns:
            a dictionary of data with their names. It ususally contains the data itself and its metadata information.
        """
        pass


def get_transform(conf, is2d=True, grayscale=False, method=Image.BICUBIC):
    transforms_list = []
    if grayscale:
        transforms_list.append(transforms.Grayscale(1))
    if 'resize' in conf.preprocess:
        osize = [conf.load_size, conf.load_size]
        transforms_list.append(transforms.Resize(osize, method))
    if 'centercrop' in conf.preprocess:
        transforms_list.append(transforms.CenterCrop(conf.crop_size))
    if 'crop' in conf.preprocess:
        transforms_list.append(transforms.RandomCrop(conf.crop_size))

    if conf.flip:
        transforms_list.append(transforms.RandomHorizontalFlip())

    transforms_list.append(transforms.ToTensor())
    if is2d:
        transforms_list.append(transforms.Normalize((0.5,), (0.5,)))
    else:
        transforms_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))

    return transforms.Compose(transforms_list)

