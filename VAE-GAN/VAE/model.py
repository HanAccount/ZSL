# VAE模型



import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import opt

class VAE(nn.Module):
    def __init__(self, opt):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(opt.img_size, opt.hidden_size)
        # 均值
        self.fc2 = nn.Linear(opt.hidden_size, opt.z_dim)
        # 方差
        self.fc3 = nn.Linear(opt.hidden_size, opt.z_dim)

        # decoder
        self.fc4 = nn.Linear(opt.z_dim, opt.hidden_size)
        self.fc5 = nn.Linear(opt.hidden_size, opt.img_size)

    # 编码层
    def encoder(self, x):
        h = F.relu(self.fc1(x))
        mu = self.fc2(h)
        log_var = self.fc3(h)
        return mu, log_var

    def decoder(self, z):
        h = F.relu(self.fc4(z))
        return F.sigmoid(self.fc5(h))

    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var / 2)
        eps = torch.rand_like(std)
        return std * eps + mu

    def forward(self, x):
        # 编码
        mu, log_var = self.encoder(x)
        # 采样
        z = self.reparameterize(mu, log_var)

        # 解码 同时输出解码的值、均值、方差
        return self.decoder(z), mu, log_var
