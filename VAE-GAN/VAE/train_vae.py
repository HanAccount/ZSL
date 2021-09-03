# 训练VAE
import sys
sys.path.append('/home/zutnlp/zut_hay/VAE-GAN/')
sys.path.append('/home/zutnlp/zut_hay/VAE-GAN/data/')
sys.path.append('/home/zutnlp/zut_hay/VAE-GAN/data/dataloader.py')
# from data.dataloader import train_loader
from VAE import model
from VAE.config import opt
import torch
import torch.nn.functional as F
from tqdm import tqdm
import os
from torchvision.utils import save_image
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

train_data = datasets.MNIST('/home/zutnlp/zut_hay/VAE-GAN/data/mnist', train=True, transform=transform, download=True)

train_loader = DataLoader(dataset=train_data, batch_size=opt.batch_size, shuffle=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
sample_dir = '/home/zutnlp/zut_hay/VAE-GAN/samples'
if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)


net = model.VAE(opt).to(device)

# 优化器
optimizer = torch.optim.Adam(net.parameters(), lr=opt.learning_rate)
# loss
def loss_fn(recon_x, x, mu, log_var):
    # 重构损失和KL散度
    reconst_loss = F.binary_cross_entropy(recon_x, x, size_average=False)
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

    return reconst_loss + KLD


def main():
    for epoch in range(opt.num_epochs):
        # 获取样本
        for i, (x, _) in enumerate(tqdm(train_loader)):
            x = x.view(-1, opt.img_size).to(device)

            x_reconst, mu, log_var = net(x)

            # loss
            loss = loss_fn(x_reconst, x, mu, log_var).to(device)

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 100 == 0:
                print('Epoch:[{}/{}] | Step:[{}/{}] | Reconst Loss + KL Div:{:.4f}'
                      .format(epoch+1, opt.num_epochs, i+1, len(train_loader), loss.item()))

        # 利用训练的模型进行测试
        with torch.no_grad():
            # 随机生成图像
            z = torch.randn(opt.batch_size, opt.z_dim).cuda()
            out = net.decoder(z).view(-1, 1, 28, 28)
            save_image(out, os.path.join(sample_dir, 'sample-{}.png'.format(epoch+1)))

            # 重构图像
            out,_,_ = net(x)
            x_concat = torch.cat([x.view(-1, 1, 28, 28), out.view(-1, 1, 28, 28)], dim=3)
            save_image(x_concat, os.path.join(sample_dir, 'reconst-{}.png'.format(epoch+1)))

if __name__ == '__main__':
    main()













