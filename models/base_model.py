import torch
import torch.nn as nn
import numpy as np
from torch.nn import init
from abc import ABC, abstractmethod
from collections import OrderedDict
from torch.optim import lr_scheduler
import functools
import os


class BaseModel(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.conf = conf
        self.isTrain = conf.isTrain
        self.gpu_ids = conf.gpu_ids
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        self.save_dir = os.path.join(conf.save_dir, conf.dataset + conf.task + conf.model)
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
        self.loss_names = []
        self.visual_names = []
        self.optimizers = []
        self.model_names = []
        self.image_paths = []
        self.metric = 0
        self.emaG = None

    @abstractmethod
    def forward(self):
        pass

    @abstractmethod
    def set_input(self, input):
        pass

    @abstractmethod
    def optimize_parameters(self):
        pass

