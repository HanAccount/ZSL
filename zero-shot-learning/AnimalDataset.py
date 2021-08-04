# AWA2-DATA
import numpy as np
from torch.utils.data import Dataset
import os
import torch
from glob import glob
from PIL import Image

class AnimalDataset(Dataset):

    def __init__(self, classes_file, transform):
        path = '../AWA2_Data/AwA2-data/Animals_with_Attributes2/'
        predicate_binary_mat = np.array(np.genfromtxt(path + 'predicate-matrix-binary.txt',dtype='int'))
        # print(predicate_binary_mat)
        self.predicate_binary_mat = predicate_binary_mat

        self.transform = transform
        class_to_index = dict()
        # 建立索引到类名的字典
        with open(path + 'classes.txt') as f:
            index = 0
            for line in f:
                # strip() 方法用于移除字符串头尾指定的字符（默认为空格或换行符）或字符序列
                class_name = line.split('\t')[1].strip()
                class_to_index[class_name] = index
                index += 1
        self.class_to_index = class_to_index

        # 存放classes_file中每个类的图片路径
        img_names = []
        # 存放classes_file中每个类在classes.txt中的索引
        img_index = []
        with open(path + classes_file) as f:
            for line in f:
                # 类的名称
                class_name = line.strip()
                # FOLDER_DIR:存放所属类图片的目录名
                FOLDER_DIR = os.path.join(path + 'JPEGImages', class_name)
                file_descriptor = os.path.join(FOLDER_DIR, '*.jpg')
                # 存放每个文件的文件名
                files = glob(file_descriptor)

                class_index = class_to_index[class_name]
                for file_name in files:
                    img_names.append(file_name)
                    img_index.append(class_index)
        self.img_names = img_names
        self.img_index = img_index



    def __getitem__(self, index):
        im = Image.open(self.img_names[index]).convert('RGB')
        # if im.getbands()[0] == 'L':
        #     im.convert('RGB')
        if self.transform:
            im = self.transform(im)
        if im.shape != (3,224,224):
            print(self.img_names[index])
        im_index = self.img_index[index]
        # 在predicate-matrix-binary.txt文件中的一行
        im_predicate = self.predicate_binary_mat[im_index,:]

        return im,im_predicate,self.img_names[index],im_index

    def __len__(self):
        return len(self.img_names)

if __name__ == '__main__':
    data = AnimalDataset('trainclasses.txt',transform=None)
    # print(data[0])
    # print(data[1])
    # print(data[2])
