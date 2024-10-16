import torch as t
import warnings
from config.base_config import BaseConfig

class Monet2PhotoConfig(BaseConfig):
    # preprocess
    load_size = 256
    crop_size = 256
    flip = False
    serial_batch = False
    preprocess = 'resize'

    # monet2photo
    dataroot = r'F:\ML\Dataset\经典风格迁移数据集\monet2photo'

    # model
    model = 'CycleGANModel'
    A = 'monet'
    B = 'photo'
    task = 'AtoB'
    in_channel = 3
    out_channel = 3

    # training
    save_dir = './ckpt'