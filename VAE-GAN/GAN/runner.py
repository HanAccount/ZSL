import os

os.system('''OMP_NUM_THREADS=8 python /home/zutnlp/zut_hay/VAE-GAN/GAN/train_gan.py --img_size 784 --hidden_size 256 --noise_size 784\
            --batch_size 128 --learning_rate 1e-3 --num_epochs 30''')