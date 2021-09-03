# 参数设置
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--img_size', default=784, type=int, help='size of image')
parser.add_argument('--hidden_size', default=256, type=int, help='size of hidden layer')
parser.add_argument('--noise_size',default=784, type=int, help='size of noise(input of G)')
parser.add_argument('--z_dim', default=20, type=int)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--learning_rate', default=0.001, type=float)
parser.add_argument('--num_epochs',default=200, type=int)
opt = parser.parse_args()

print(opt)

