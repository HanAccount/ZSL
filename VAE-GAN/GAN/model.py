# GAN模型

import torch.nn as nn
import torch
import torch.nn.functional as F
from GAN.config import opt
'''
    生成对抗网络分为 生成网络(generator) 对抗网络(discriminator)
    生成网络(generator):通过随机输入的noise噪声生成一个图像
    对抗网络(discriminator):对抗网络就是一个二分类器  输出的生成的图像是否为真 real/fake
'''

class Generator(nn.Module):
    def __init__(self, opt):
        super(Generator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(opt.noise_size, opt.hidden_size),
            nn.ReLU(),
            nn.Linear(opt.hidden_size, opt.hidden_size),
            nn.ReLU(),
            nn.Linear(opt.hidden_size, opt.img_size),
            nn.Tanh(),
        )
    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), -1)

        return img


class Discriminator(nn.Module):
    def __init__(self, opt):
        super(Discriminator, self).__init__()

        # 判别器
        self.model = nn.Sequential(
            # 748 -> 256
            nn.Linear(opt.img_size, opt.hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(opt.hidden_size, opt.hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(opt.hidden_size, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        h = self.model(x)
        return h