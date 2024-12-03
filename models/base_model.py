import torch

import torch.nn as nn
import numpy as np
from torch.nn import init
from abc import ABC, abstractmethod
from collections import OrderedDict
from torch.optim import lr_scheduler
import functools
import os

from models.model_utils import *
from models.networks import *

class BaseModel(nn.Module):
    def __init__(self, conf):
        """
        初始化模型配置和状态。

        参数:
        - conf: 配置对象，包含模型训练和运行的各种配置参数。

        在初始化过程中，该方法会：
        - 调用超类的初始化方法进行基本初始化；
        - 根据配置文件设置模型的训练状态、GPU配置、保存目录等；
        - 初始化各种名称列表和优化器列表，用于后续模型组件的注册和管理；
        - 设置初始的度量值和指数移动平均生成器状态。
        """
        super().__init__()
        self.conf = conf
        self.isTrain = conf.isTrain
        self.gpu_ids = conf.gpu_ids
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        self.save_dir = os.path.join(conf.save_dir, conf.dataset + conf.task + conf.model)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
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
        """
        设置输入数据的方法

        该方法用于接收输入数据，并在子类中实现具体的设置逻辑
        由于是抽象方法，所以具体的实现需要在子类中完成

        参数:
        input: 传入的输入数据，类型和内容取决于具体的使用场景

        返回值:
        无返回值，具体的处理结果应体现在对象的状态改变上
        """
        pass

    @abstractmethod
    def optimize_parameters(self):
        """
        抽象方法：优化参数

        此方法的目的是在特定的上下文中实现参数的优化。具体的优化策略或算法需要在子类中实现。
        由于此方法被声明为抽象方法，因此它不会在此基类中提供具体的实现代码；而是强迫子类去实现
        这个方法以满足特定的需求。
        """
        pass

    def setup(self, conf):
        """
        根据配置和训练状态初始化模型。

        设置了：
            isTrain
            load_suffix
        打印了网络信息

        参数:
        - conf: 配置对象，包含了模型训练或测试的各种参数。
        """
        # 如果模型是训练状态，则初始化调度器列表
        if self.isTrain:
            self.schedulers = [get_scheduler(optimizer, conf) for optimizer in self.optimizers]
        # 如果模型不是训练状态或者配置要求继续训练，则尝试加载网络
        if not self.isTrain or conf.continue_train:
            # 根据配置确定要加载的网络迭代次数或epoch数的后缀
            if conf.load_iter == 'latest':
                load_suffix = 'latest'
            else:
                # 确保配置的迭代次数大于0，否则使用epoch数（epoch默认为latest）
                load_suffix = 'iter_%d' % conf.load_iter if conf.load_iter > 0 else conf.epoch
            # 根据确定的后缀加载网络
            self.load_networks(load_suffix)
        # 根据配置的详细程度打印网络信息
        self.print_networks(conf.verbose)

    def eval(self):
        for name in self.model_names:
            if isinstance(name, str):       # 网络名称是否为字符串类型
                net = getattr(self, 'net' + name)   # 假如说name='G_A2B' 那net加载的就是self.netG_A2B
                net.eval()                  # 将网络设置为评估模式

    def test(self):
        """
        测试模型的函数。

        此函数在不进行梯度计算的情况下运行模型的前向传播和可视化计算，
        通常用于评估模型性能或生成模型的可视化结果。
        """
        with torch.no_grad():
            # 运行模型的前向传播，通常用于预测或评估模型性能
            self.forward()
            # 计算和生成模型的可视化结果
            self.compute_visuals()

    def compute_visuals(self):
        """
        计算并生成视觉元素

        此方法负责处理数据，并生成相应的视觉元素或图表。
        它可能涉及到复杂的可视化库操作，以及数据的清洗和计算。
        由于视觉元素的具体生成过程依赖于数据类型和可视化目标，
        因此该方法需要灵活地适应不同的数据结构和可视化需求。
        """
        pass

    def update_learning_rate(self):
        """
        更新学习率。

        此方法根据预定义的学习率策略，调整优化器的学习率。它支持不同的学习率策略，包括但不限于'plateau'，
        并根据当前指标或直接按预定计划调整学习率。在调整前后，它会记录并打印出学习率的变化。
        """
        # 记录更新前的学习率
        old_lr = self.optimizers[0].param_groups[0]['lr']

        print("------------==================-----------------", self.schedulers)

        # 遍历所有学习率调度器
        for scheduler in self.schedulers:
            # 根据配置的策略更新学习率
            if self.conf.lr_policy == 'plateau':
                scheduler.step(self.metric)  # 对于plateau策略，需要传入当前指标作为参数
            else:
                scheduler.step()  # 其他策略直接调用 step方法

        # 记录更新后的学习率
        lr = self.optimizers[0].param_groups[0]['lr']
        # 输出学习率的变化情况
        print('learning rate %.7f -> %.7f' % (old_lr, lr))

    def get_current_visuals(self):
        """
        获取当前实例的视觉表示。

        此方法创建一个有序字典，其中包含实例当前状态下的所有视觉元素。
        它通过遍历视觉元素的名称列表，并从实例本身获取相应的属性值。

        Returns:
            OrderedDict: 一个有序字典，键是视觉元素的名称，值是对应的视觉元素。

        """
        # 初始化一个有序字典来存储视觉元素
        visual_ret = OrderedDict()
        # 遍历视觉元素的名称列表
        for name in self.visual_names:
            # 检查名称是否为字符串，以确保其有效性
            if isinstance(name, str):
                # 从实例中获取对应名称的视觉元素，并添加到有序字典中
                visual_ret[name] = getattr(self, name)
        # 返回包含所有视觉元素的有序字典
        return visual_ret

    def get_current_losses(self):
        """
        获取当前的损失值。

        该方法遍历损失名称列表，并从对象的属性中获取相应的损失值，
        然后将其以字典的形式返回。这样做是为了方便管理和访问不同的损失值。

        Returns:
            OrderedDict: 一个有序字典，包含所有损失的名称和它们的当前值。
        """
        # 初始化一个有序字典，用于存储损失名称和对应的损失值
        errors_ret = OrderedDict()

        # 遍历损失名称列表
        for name in self.loss_names:
            # 检查名称是否为字符串，确保名称格式正确
            if isinstance(name, str):
                # 通过属性名动态获取损失值，并将其转换为浮点类型，然后存储到字典中
                errors_ret[name] = float(getattr(self, 'loss_' + name))
        # 返回包含所有损失的有序字典
        return errors_ret

    def save_networks(self, epoch):
        """
        保存网络模型的参数。

        如果epoch被指定为'latest'且存在指数移动平均(EMA)的G网络，则应用EMA阴影。
        遍历所有模型名称，逐个保存模型的参数。保存路径基于epoch和模型名称。

        参数:
        - self: 实例引用。
        - epoch: 保存的epoch编号，用于构建保存文件名。

        没有返回值。
        """
        # 如果epoch为'latest'且存在EMA的G网络，则应用EMA阴影
        if epoch == 'latest' and self.emaG:
            self.emaG.apply_shadow()
            print('The latest using EMA.')

        # 遍历所有模型名称，逐个保存模型的参数
        for name in self.model_names:
            if isinstance(name, str):
                # 构建保存文件名
                save_filename = '%s_net_%s.pth' % (epoch, name)
                save_path = os.path.join(self.save_dir, save_filename)
                net = getattr(self, 'net' + name)

                # 保存模型参数
                # 注释掉的代码是之前用于多GPU训练时的代码，当前使用的是一些简化的保存逻辑
                torch.save(net.state_dict(), save_path)

    def __patch_instance_norm_state_dict(self, state_dict, module, keys, i=0):
        """
        修复 InstanceNorm 检查点在 0.4 版本之前的不兼容问题

        该函数用于解决 0.4 版本之前 InstanceNorm 检查点的兼容性问题。
        它根据模块类型和键名检查并更新 state_dict，移除不兼容的键。

        参数:
        - state_dict: 模型的状态字典，记录了每一层的状态。
        - module: 当前正在检查的模块（层）。
        - keys: 字符串列表，用于定位状态字典中的特定键。
        - i: 当前在 keys 列表中的索引，用于递归。
        """
        # 获取当前正在检查的键
        key = keys[i]

        # 检查当前键是否指向模块末尾的参数或缓冲区
        if i + 1 == len(keys):
            # 检查当前模块是否为 InstanceNorm 类型，且键名为 running_mean 或 running_var
            if module.__class__.__name__.startswith('InstanceNorm') and \
                    (key == 'running_mean' or key == 'running_var'):
                # 如果当前模块的 running_mean 或 running_var 为 None，则从 state_dict 中移除该键
                if getattr(module, key) is None:
                    state_dict.pop('.'.join(keys))
            # 检查当前模块是否为 InstanceNorm 类型，且键名为 num_batches_tracked
            if module.__class__.__name__.startswith('InstanceNorm') and \
                    (key == 'num_batches_tracked'):
                # 从 state_dict 中移除 num_batches_tracked 键
                state_dict.pop('.'.join(keys))
        else:
            # 如果当前键不是模块末尾的键，则递归检查下一个键
            self.__patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)

    def load_networks(self, epoch):
        """Load all the networks from the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)

        加载了

        """
        for name in self.model_names:
            if isinstance(name, str):
                load_filename = '%s_net_%s.pth' % (epoch, name)
                load_path = os.path.join(self.save_dir, load_filename)
                net = getattr(self, 'net' + name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                print('loading the model from %s' % load_path)
                # if you are using PyTorch newer than 0.4 (e.g., built from
                # GitHub source), you can remove str() on self.device
                state_dict = torch.load(load_path, map_location=str(self.device))
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k[7:]
                    new_state_dict[name] = v
                net.load_state_dict(new_state_dict)

                # if hasattr(state_dict, '_metadata'):
                #     del state_dict._metadata

                # # patch InstanceNorm checkpoints prior to 0.4
                # for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
                #     self.__patch_instance_norm_state_dict(state_dict, net, key.split('.'))
                # net.load_state_dict(state_dict)


    def print_networks(self, verbose):
        """Print the total number of parameters in the network and (if verbose) network architecture

        Parameters:
            verbose (bool) -- if verbose: print the network architecture
        """
        print('---------- Networks initialized -------------')
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                if verbose:
                    print(net)
                print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
        print('-----------------------------------------------')

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad


