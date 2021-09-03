# GAN训练
import torch
from torch.autograd import Variable
from tqdm import tqdm
from GAN.config import opt
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import os
from GAN.model import Generator, Discriminator
from torchvision.utils import save_image

# 准备数据
transform = transforms.Compose([
    # transforms.Resize(opt.img_size),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

data = datasets.MNIST('/home/zutnlp/zut_hay/VAE-GAN/data/mnist', train=True, transform=transform, download=True)

data_loader = DataLoader(dataset=data, batch_size=opt.batch_size, shuffle=True, drop_last=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


sample_dir = '/home/zutnlp/zut_hay/VAE-GAN/GAN/samples'
if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)

# 实例模型
D_net = Discriminator(opt).to(device)
G_net = Generator(opt).to(device)
print(D_net)
print(G_net)
# 优化器
optimizer_G = torch.optim.Adam(G_net.parameters(), lr=opt.learning_rate, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(D_net.parameters(), lr=opt.learning_rate, betas=(0.5, 0.999))

# loss
adversarial_loss = torch.nn.BCELoss()

total_step = len(data_loader)

def main():
    for epoch in range(opt.num_epochs):
        for i, (img,_) in enumerate(tqdm(data_loader)):
            bs = opt.batch_size
            # 真/假标签
            valid = Variable(torch.Tensor(bs, 1).fill_(1.0), requires_grad=False).to(device)
            fake = Variable(torch.Tensor(bs, 1).fill_(0.0), requires_grad=False).to(device)

            real_img = Variable(img.view(bs, -1)).to(device)
            # print('real_img shape:',real_img.shape)   # [64,784]

            # training Generator
            optimizer_G.zero_grad()
            # 生成随机噪声
            z = Variable(torch.FloatTensor(torch.normal(0, 1, (bs, opt.noise_size)))).to(device)
            # 生成的图像
            gen_img = G_net(z)
            # loss
            g_loss = adversarial_loss(D_net(gen_img), valid)

            g_loss.backward()
            optimizer_G.step()

            # training Discriminator
            optimizer_D.zero_grad()
            real_loss = adversarial_loss(D_net(real_img), valid)
            fake_loss = adversarial_loss(D_net(gen_img.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2

            d_loss.backward()
            optimizer_D.step()

            if (i+1) % 200 == 0:
                print(
                    'Epoch:[{}/{}] | Step:[{}/{}] | d_loss:{:.4f} | g_loss:{:.4f} | D(x):{:.2f} | D(G(x)):{:.2f}'
                    .format(epoch+1, opt.num_epochs, i+1, total_step, d_loss.item(), g_loss.item(),
                            D_net(real_img).mean().item(), D_net(gen_img).mean().item())
                )
        # 在第一轮保存训练数据图像
        if (epoch+1) == 1:
            img = img.reshape(img.shape[0],1,28,28)
            save_image(img, os.path.join(sample_dir, 'real-img.png'))

        # 每一轮保存生成的图像
        fake_img = gen_img.reshape(img.shape[0],1,28,28)
        save_image(fake_img, os.path.join(sample_dir, 'fake-img-{}.png'.format(epoch+1)))

if __name__ == '__main__':
    main()



