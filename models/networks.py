import torch
import torch.nn as nn
import functools

import numpy as np
from abc import ABC, abstractmethod
from collections import OrderedDict
import os

class Identity(nn.Module):
    """
    该类继承自nn.Module，定义了一个最简单的身份映射模型。
    它的作用是将输入直接作为输出返回，不进行任何变换。

    Attributes:
        该类没有定义任何属性。
    """

    def forward(self, x):
        """
        定义了模型的前向传播逻辑。

        参数:
            x (Tensor): 输入数据，可以是任意形状的张量。

        返回:
            Tensor: 输入数据x直接作为输出返回，未经过任何变换。
        """
        return x


class Residual(nn.Module):
    """
    Residual模块，用于在神经网络中实现跳跃连接。

    该模块通过将一个函数（通常是一个或多个层的组合）应用于输入上，并将结果与原始输入相加，
    来实现输入的变换。这种设计模式有助于缓解深层神经网络中的梯度消失问题。

    继承自:
        nn.Module: PyTorch的基类模块，用于构建自定义的神经网络组件。
    """

    def __init__(self, fn):
        """
        初始化Residual模块。

        参数:
            fn (Callable): 一个函数或层的序列，用于对输入数据进行变换。这通常包括一个或多个神经网络层。
        """
        super().__init__()
        self.fn = fn

    def forward(self, x):
        """
        定义前向传播计算。

        通过应用变换函数`self.fn`到输入`x`上，并将结果与原始输入`x`相加，
        来实现残差块的前向传播。

        参数:
            x (Tensor): 输入数据，通常是一个张量，表示神经网络的一层输入。

        返回:
            Tensor: 变换后的数据加上原始输入的和，作为残差块的输出。
        """
        return self.fn(x) + x


class Norm(nn.Module):
    """
    一个用于在神经网络中应用归一化以及后续函数的模块。

    该类继承自`nn.Module`，其主要功能是先对输入数据进行层归一化（Layer Normalization），
    然后应用传入的函数`fn`。层归一化有助于训练深层神经网络，通过归一化数据分布来加速训练过程
    并提高模型的稳定性。

    参数:
    - dim: int，归一化维度。指明输入数据的特征维度，归一化将沿着这个维度进行。
    - fn: function，应用在归一化数据上的函数。这允许用户自定义后续的网络层或操作。

    返回:
    - 该类实例化后不直接返回值，但包含了一个可以对输入数据进行层归一化并应用`fn`函数的方法。
    """
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)  # 初始化层归一化模块，指定归一化维度
        self.fn = fn  # 保存外部传入的函数，后续将应用于归一化后的数据


    # 先对输入 x 进行层归一化处理，然后将结果传递给 fn 函数，并返回最终结果
    def forward(self, x):
        return self.fn(self.norm(x))


class FeedForward(nn.Module):
    """
    前馈神经网络模块。

    该模块主要包含两部分：
    1. 将输入数据从维度 `dim` 映射到 `hidden_dim` 的线性变换。
    2. 将数据从 `hidden_dim` 映射回 `dim` 的线性变换。

    在线性变换之间使用 GELU 激活函数和 dropout 进行非线性和正则化处理。

    参数:
        dim (int): 输入和输出数据的维度。
        hidden_dim (int): 隐藏层的维度。
        dropout (float): dropout 概率，在训练过程中随机将张量中的某些元素设置为零。
    """

    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        # 构建包括线性变换、GELU 激活和 dropout 的顺序模型
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),  # 将数据映射到隐藏层
            nn.GELU(),  # GELU 激活函数
            nn.Dropout(dropout),  # Dropout 层
            nn.Linear(hidden_dim, dim),  # 将数据映射回原始维度
            nn.Dropout(dropout)  # Dropout 层
        )

    def forward(self, x):
        """
        前馈网络的前向传播。

        参数:
            x (Tensor): 输入张量，形状为 (batch_size, dim)。

        返回:
            Tensor: 经过前馈网络后的输出张量，形状为 (batch_size, dim)。
        """
        return self.net(x)  # 将输入通过顺序模型以获得输出


class ResnetBlock(nn.Module):
    """
    定义一个ResNet块，用于深度学习模型中的特征学习.

    ResNet块包含两个卷积层，每个卷积层后接一个标准化层和一个ReLU激活函数.
    如果指定使用dropout，还会在一个卷积层后加入dropout层.这个块的主要作用是
    在神经网络中学习残差函数，有助于缓解深层网络中的梯度消失问题.

    参数:
        dim (int): 输入和输出的通道维度.
        padding_type (str): 卷积层前使用的填充类型，可以是'reflect', 'replicate'或'zero'.
        norm_layer (nn.Module): 使用的标准化层类型.
        use_dropout (bool): 是否使用dropout.
        use_bias (bool): 卷积层是否使用偏差.
    """

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super().__init__()
        # 构建卷积块并保存为成员变量
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """
        构建并返回一个包含两个卷积层的序列化卷积块.

        参数和返回值同__init__方法.
        """
        conv_block = []
        p = 0
        # 根据填充类型添加相应的填充层
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1

        # 第一个卷积层及其后续的标准化层和ReLU激活函数
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        # 如果使用dropout，则添加dropout层
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        # 同样的逻辑适用于第二个卷积层，但没有ReLU激活函数
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        # 返回序列化的卷积块
        return nn.Sequential(*conv_block)

    def forward(self, x):
        """
        定义前向传播计算.

        参数:
            x (Tensor): 输入张量.

        返回:
            Tensor: 输入张量和经过卷积块的输出张量之和，即残差连接的结果.
        """
        # 残差连接：将输入张量与经过卷积块的输出张量相加
        out = x + self.conv_block(x)
        return out


class UnetBlock(nn.Module):
    """
    定义一个Unet块，用于构建Unet模型。

    Unet块是Unet模型的一部分，包含下采样和上采样路径，可以堆叠形成整个Unet结构。
    这个块的特殊之处在于它能接收来自先前路径的输入，并将其与下采样后上采样回来的特征图合并。

    参数:
    - in_channel: 输入通道数。
    - out_channel: 输出通道数，默认为1。
    - hidden_channel: 隐藏通道数，默认为1。
    - pre_module: 前一个模块，用于拼接特征图。
    - inner: 标志表示这是最内部的Unet块，默认为False。
    - outer: 标志表示这是最外部的Unet块，默认为False。
    - norm_layer: 使用的规范化层，默认为nn.BatchNorm2d。
    - use_dropout: 是否使用dropout，默认为False。
    """
    def __init__(self, in_channel=None, out_channel=1, hidden_channel=1, pre_module=None, inner=False, outer=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super().__init__()
        self.outer = outer
        # 判断是否使用偏置，取决于规范化层的类型
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        # 如果没有指定输入通道数，假设它等于输出通道数
        if in_channel == None:
            in_channel = out_channel

        # 定义下采样层
        downconv = nn.Conv2d(in_channel, hidden_channel, kernel_size=4, stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(hidden_channel)
        # 定义上采样层
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(out_channel)

        # 根据Unet块的位置（外层、内层或其他），配置不同的层结构
        if outer:
            #
            upconv = nn.ConvTranspose2d(hidden_channel * 2, out_channel, kernel_size=4, stride=2, padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [pre_module] + up
        elif inner:
            upconv = nn.ConvTranspose2d(hidden_channel, out_channel, kernel_size=4, stride=2, padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            # 定义一个二维转置卷积层，用于增加特征图的尺寸
            upconv = nn.ConvTranspose2d(hidden_channel * 2, out_channel, kernel_size=4, stride=2, padding=1, bias=use_bias)

            # 构建下采样层列表，包含激活函数、卷积和标准化操作，用于减小特征图的尺寸
            down = [downrelu, downconv, downnorm]

            # 构建上采样层列表，包含激活函数、转置卷积和标准化操作，用于增加特征图的尺寸
            up = [uprelu, upconv, upnorm]

            # 根据需求添加dropout层
            if use_dropout:
                model = down + [pre_module] + up + [nn.Dropout(0.5)]
            else:
                model = down + [pre_module] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        """
        定义前向传播过程。

        参数:
        - x: 输入的特征图。

        返回:
        - 如果是外层块，直接返回模型的输出。
        - 否则，将输入特征图与模型输出拼接后返回。
        """
        if self.outer:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], dim=1)


class UnetBlock_with_z(nn.Module):
    """
    U-Net块与z（潜在变量）的自定义模块。

    该模块是U-Net架构的一个扩展，允许在网络中引入潜在变量z，以便在图像生成或翻译任务中使用。

    参数:
    - in_channel: 输入通道数。
    - out_channel: 输出通道数。
    - hidden_channel: 隐藏层通道数。
    - nz: 潜在变量z的维度，默认为0。
    - pre_module: 前置模块，用于处理潜在变量z。
    - inner: 是否为最内层的标记。
    - outer: 是否为最外层的标记。
    - norm_layer: 归一化层，用于网络中的归一化操作。
    - use_dropout: 是否使用dropout的标记。
    """

    def __init__(self, in_channel, out_channel, hidden_channel, nz=0, pre_module=None, inner=False, outer=False,
                 norm_layer=None, use_dropout=False):
        super(UnetBlock_with_z, self).__init__()
        # 初始化下采样卷积层
        downconv = []
        self.inner = inner
        self.outer = outer
        self.nz = nz
        in_channel = in_channel + nz
        downconv += [nn.Conv2d(in_channel, hidden_channel, kernel_size=4, stride=2, padding=1)]
        downrelu = nn.LeakyReLU(0.2, True)
        uprelu = nn.ReLU(True)

        # 根据是外层、内层还是中间层，构建不同的网络结构
        if self.outer:
            upconv = [nn.ConvTranspose2d(hidden_channel * 2, out_channel, kernel_size=4, stride=2, padding=1)]
            down = downconv
            up = [uprelu] + upconv + [nn.Tanh()]
        elif self.inner:
            upconv = [nn.ConvTranspose2d(hidden_channel, out_channel, kernel_size=4, stride=2, padding=1)]
            down = [downrelu] + downconv
            up = [uprelu] + upconv + [norm_layer(out_channel)]
        else:
            upconv = [nn.ConvTranspose2d(hidden_channel * 2, out_channel, kernel_size=4, stride=2, padding=1)]
            down = [downrelu] + downconv + [norm_layer(hidden_channel)]
            up = [uprelu] + upconv + [norm_layer(out_channel)]
            if use_dropout:
                up += [nn.Dropout(0.5)]
        self.down = nn.Sequential(*down)
        self.pre_module = pre_module
        self.up = nn.Sequential(*up)

    def forward(self, x, z):
        """
        前向传播函数。

        参数:
        - x: 输入张量。
        - z: 潜在变量z。

        返回:
        - 输出张量。
        """
        # 根据是否存在潜在变量z，决定是否将z与输入x拼接
        if self.nz > 0:
            z_img = z.view(z.size(0), z.size(1), 1, 1).expand(z.size(0), z.size(1), x.size(2), x.size(3))
            x_and_z = torch.cat([x, z_img], dim=1)
        else:
            x_and_z = x

        # 根据是外层、内层还是中间层，执行不同的前向传播逻辑
        if self.outer:
            x1 = self.down(x_and_z)
            x2 = self.pre_module(x1, z)
            return self.up(x2)
        elif self.inner:
            x1 = self.up(self.down(x_and_z))
            return torch.cat([x1, x], dim=1)
        else:
            x1 = self.down(x_and_z)
            x2 = self.pre_module(x1, z)
            return torch.cat([self.up(x2), x], dim=1)


class D_NLayersMulti(nn.Module):
    def __init__(self, in_channel, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, num_D=1):
        super(D_NLayersMulti, self).__init__()
        self.num_D = num_D
        if num_D == 1:
            layers = self.get_layers(in_channel, ndf, n_layers, norm_layer)
            self.model = nn.Sequential(*layers)
        else:
            layers = self.get_layers(in_channel, ndf, n_layers, norm_layer)
            self.add_module("model_0", nn.Sequential(*layers))
            self.down = nn.AvgPool2d(kernel_size=3, stride=2, padding=[1, 1], count_include_pad=False)

            for i in range(1, num_D):
                ndf_i = int(round(ndf / (2 ** i)))
                layers = self.get_layers(in_channel, ndf_i, n_layers, norm_layer)
                self.add_module("model_%d" % i, nn.Sequential(*layers))

    def get_layers(self, in_channel, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(in_channel, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]
        nf_multi = 1
        nf_multi_prev = 1
        for n in range(1, n_layers):
            nf_multi_prev = nf_multi
            nf_multi = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_multi_prev, ndf * nf_multi, kw, 2, padw),
                norm_layer(ndf * nf_multi),
                nn.LeakyReLU(0.2, True)
            ]

        nf_multi_prev = nf_multi
        nf_multi = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_multi_prev, ndf * nf_multi, kw, 1, padw),
            norm_layer(ndf * nf_multi),
            nn.LeakyReLU(0.2, True)
        ]
        sequence += [
            nn.Conv2d(ndf * nf_multi, 1, kw, 1, padw)
        ]
        return sequence

    def forward(self, x):
        if self.num_D == 1:
            return self.model(x)
        result = []
        down = x
        for i in range(self.num_D):
            model = getattr(self, "model_%d" % i)
            result.append(model(down))
            if i != self.num_D - 1:
                down = self.down(down)
        return result


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, norm_layer=None):
        super(ResBlock, self).__init__()
        layers = []
        if norm_layer is not None:
            layers += [norm_layer(in_channel)]
        layers += [
            nn.ReLU(True),
            nn.Conv2d(in_channel, in_channel, 3, 1, 1, bias=True)
        ]
        if norm_layer is not None:
            layers += [norm_layer(in_channel)]
        layers += [
            nn.ReLU(True),
            nn.Conv2d(in_channel, out_channel, 3, 1, 1, bias=True),
            nn.AvgPool2d(2, 2)
        ]

        self.conv = nn.Sequential(*layers)
        self.shortcut = nn.Sequential(
            nn.AvgPool2d(2, 2),
            nn.Conv2d(in_channel, out_channel, 1, 1, 0, bias=True)
        )

    def forward(self, x):
        out = self.conv(x) + self.shortcut(x)
        return out


class E_ResNet(nn.Module):
    def __init__(self, in_channel=1, out_channel=1, ndf=64, n_blocks=4, norm_layer=None, vaeLike=False):
        super(E_ResNet, self).__init__()
        self.vaeLike = vaeLike
        max_ndf = 4
        conv_layers = [
            nn.Conv2d(in_channel, ndf, 4, 2, 1, bias=True)
        ]
        for n in range(1, n_blocks):
            in_ndf = ndf * min(max_ndf, n)
            out_ndf = ndf * min(max_ndf, n + 1)
            conv_layers += [
                ResBlock(in_ndf, out_ndf, norm_layer)
            ]
        conv_layers += [
            nn.ReLU(True),
            nn.AvgPool2d(8)
        ]

        if vaeLike:
            self.fc = nn.Sequential(*[nn.Linear(out_ndf, out_channel)])
            self.fcVar = nn.Sequential(*[nn.Linear(out_ndf, out_channel)])
        else:
            self.fc = nn.Sequential(*[nn.Linear(out_ndf, out_channel)])
        self.conv = nn.Sequential(*conv_layers)

    def forward(self, x):
        x_conv = self.conv(x)
        conv_flat = x_conv.view(x.size(0), -1)
        output = self.fc(conv_flat)
        if self.vaeLike:
            outputVar = self.fcVar(conv_flat)
            return output, outputVar
        else:
            return output


class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=1, stride=1, groups=1, norm_layer=None):
        super().__init__()
        padding = (kernel_size - 1) // 2
        conv = []
        conv += [
            norm_layer(in_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups,
                      bias=False)
        ]

        self.conv = nn.Sequential(*conv)

    def forward(self, x):
        return self.conv(x)


class DilatedConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=(1, 1), stride=1, groups=1, dilated=(1, 1),
                 norm_layer=None):
        super().__init__()
        padding = tuple(
            [(k - 1) // 2 * d for k, d in zip(kernel_size, dilated)]
        )
        conv = []
        conv += [
            norm_layer(in_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups,
                      dilation=dilated, bias=False)
        ]
        self.conv = nn.Sequential(*conv)

    def forward(self, x):
        return self.conv(x)


class ADBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, groups=1, dilated=[1, 3, 5], norm_layer=None):
        super().__init__()
        mid = in_channel if in_channel <= out_channel else out_channel
        self.w1 = nn.Parameter(torch.ones(1), requires_grad=True)
        self.w2 = nn.Parameter(torch.ones(1), requires_grad=True)
        self.w3 = nn.Parameter(torch.ones(1), requires_grad=True)

        self.conv_1 = ConvBlock(in_channel, in_channel // 4, kernel_size=1, stride=1, norm_layer=norm_layer)
        self.conv_2 = ConvBlock(in_channel // 4, mid, kernel_size=1, stride=1, norm_layer=norm_layer)

        self.d_conv = nn.ModuleList()
        for i in range(3):
            self.d_conv.append(
                DilatedConvBlock(mid, out_channel, kernel_size=(3, 3), stride=stride, groups=groups,
                                 dilated=(dilated[i], dilated[i]), norm_layer=norm_layer)
            )
        self.gconv_3 = DilatedConvBlock(out_channel, out_channel, kernel_size=(1, 3), groups=groups, stride=(1, 1),
                                        norm_layer=norm_layer)
        if stride == 1:
            self.res = ConvBlock(mid, out_channel, kernel_size=1, stride=1, norm_layer=norm_layer)
        if stride == 2:
            self.res = ConvBlock(mid, out_channel, kernel_size=2, stride=2, norm_layer=norm_layer)

        # skip connection
        if in_channel != out_channel or stride != 1:
            if stride == 1:
                self.conv_res = ConvBlock(in_channel, out_channel, kernel_size=1, stride=1, norm_layer=norm_layer)
            if stride == 2:
                self.conv_res = ConvBlock(in_channel, out_channel, kernel_size=2, stride=2, norm_layer=norm_layer)

    def forward(self, x):
        res = x
        x = self.conv_1(x)
        x = self.conv_2(x)
        xd = self.w1 * self.d_conv[0](x) + self.w2 * self.d_conv[1](x) + self.w3 * self.d_conv[2](x)
        x = xd + self.res(x)  # residual connection
        x = self.gconv_3(x)
        if hasattr(self, 'conv_res'):
            res = self.conv_res(res)
        return x + res


class UpADBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, groups=1, dilated=[1, 3, 5], norm_layer=None):
        super().__init__()
        mid = in_channel if in_channel <= out_channel else out_channel
        self.conv_1 = ConvBlock(in_channel, in_channel // 4, kernel_size=1, stride=1, norm_layer=norm_layer)
        self.conv_2 = ConvBlock(in_channel // 4, mid, kernel_size=1, stride=1, norm_layer=norm_layer)

        self.g_conv = nn.Sequential(
            ConvBlock(mid, out_channel, kernel_size=3, stride=stride, groups=groups, norm_layer=norm_layer),
            ConvBlock(out_channel, out_channel, kernel_size=3, stride=1, groups=groups, norm_layer=norm_layer)
        )
        self.gconv_3 = DilatedConvBlock(out_channel, out_channel, kernel_size=(1, 3), groups=groups, stride=(1, 1),
                                        norm_layer=norm_layer)
        # skip connection
        if in_channel != out_channel or stride != 1:
            if stride == 1:
                self.conv_res = ConvBlock(in_channel, out_channel, kernel_size=1, stride=1, norm_layer=norm_layer)
            if stride == 2:
                self.conv_res = ConvBlock(in_channel, out_channel, kernel_size=2, stride=2, norm_layer=norm_layer)

    def forward(self, x):
        res = x
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.g_conv(x)

        if hasattr(self, 'conv_res'):
            res = self.conv_res(res)

        return x + res
