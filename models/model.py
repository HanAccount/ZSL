
# Unet模型
import torch
import torch.nn as nn
import torchvision.transforms.functional as F

# Unet中每个节点都是双卷积构成的
class DoubleConv(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            # 将bias设置为False是为了使用batch normal
            nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self,x):
        return self.conv(x)

class UNET(nn.Module):
    def __init__(self,in_channels = 3 , out_channels = 1, features = [64,128,256,512]):
        """
        :param in_channels: 输入channel
        :param out_channels: 输出channel
        :param features: 会使用到的channel数
        """
        super(UNET, self).__init__()
        # 模型整体结构分为下采样，上采样，池化操作
        # ModuleList():可以存储不同的Module，并自动将每个module的parameters添加到网络之中的容器
        # 但是内部并没有实现forward()
        self.downs = nn.ModuleList()    # 下采样的操作
        self.ups = nn.ModuleList()      # 上采样的操作
        self.pool = nn.MaxPool2d(kernel_size=2,stride=2)

        # UNET中的下采样:使用的是双层卷积的方式
        for feature in features:
            # 下采样通道数的变化:[in_channels->64->128->256->512]
            self.downs.append(
                DoubleConv(in_channels,feature)
            )
            in_channels = feature

        # UNET中的上采样:使用的是反卷积的方式
        # 上采样图像尺寸逐渐变大通道逐渐减少,所以要翻转
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature * 2, feature,kernel_size=2,stride=2)
            )
            self.ups.append(DoubleConv(feature * 2, feature))

        # 模型底部实现 channel:512->1024
        self.bottleneck = DoubleConv(features[-1],features[-1] * 2)
        # 最后一个卷积 channel:61->1
        self.final_conv = nn.Conv2d(features[0],out_channels,kernel_size=1)


    def forward(self,x):
        skip_connections = []

        # 下采样
        for down in self.downs:
            x = down(x)
            # 保存每一层的结果 为了跟上采样的结果进行concat
            skip_connections.append(x)
            x = self.pool(x)

        # 底部
        x = self.bottleneck(x)

        # 将下采样的输出进行翻转 为了和后面反卷积结果拼接
        skip_connections = skip_connections[::-1]

        # 上采样
        # 在上采样模型列表中 偶数是反卷积操作 奇数对应是双卷积
        for idx in range(0, len(self.ups), 2):
            # idx取值为偶数 对应的是反卷积的操作
            x = self.ups[idx](x)
            # 在下采样输出中找到和反卷积输出对应位置的对应的输出
            skip_connection = skip_connections[idx // 2]
            # print(skip_connection.shape)
            # 保证拼接的数据维度一致
            if x.shape != skip_connection.shape:
                # print(skip_connection.shape)
                # skip_connection.shape:[batchsize,channel,h.w]
                x = F.resize(x,size=skip_connection.shape[2:])

            # 在进行反卷积后跟下采样得到的结果进行拼接
            concat_skip = torch.cat((skip_connection,x), dim=1)
            # 将拼接的数据输入到双卷积
            x = self.ups[idx + 1](concat_skip)

        return self.final_conv(x)

def test():
    x = torch.randn((3,1,224,224))
    model = UNET(in_channels=1,out_channels=1)
    preds = model(x)
    print(preds.shape)
    print(x.shape)
    assert preds.shape == x.shape

if __name__ == '__main__':
    test()

