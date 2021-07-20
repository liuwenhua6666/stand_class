import sys
sys.path.append('./data')
sys.path.append('./model')

import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from model import MobileNetV3_large
from model import MobileNetV3_small
import torchvision
from torch.autograd import Variable
import cv2
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.nn import functional as F

class sunDatasetInfer(Dataset):
    def __init__(self, data_dir):
        c_dir=data_dir
        # self.c_paths=sorted(glob.glob(c_dir+'/*'))
        self.c_paths=sorted([data_dir])
        self.transform_valid = A.Compose([
            A.Resize(height=224, width=112),
            A.Normalize(mean=(0.453 ,0.422 ,0.428), std=(0.158, 0.163, 0.161), max_pixel_value=255.0, p=1.0),
            # A.Normalize(mean=(115.515, 107.61, 109.14), std=(40.29, 41.565, 41.055), max_pixel_value=1, p=1.0),
            ToTensorV2(p=1.0),
        ])
        self.data_transforms = self.transform_valid
    def __getitem__(self,index):
        #第index个样本
        sample_path1 = self.c_paths[index]
        img1 = Image.open(sample_path1)
        # img1 = pad_image(sample_path1)
        # img1 = Image.open(sample_path1)
        img1=np.array(img1)
        img = self.data_transforms(image=img1)['image']
        return img ,sample_path1

    def __len__(self):
        return len(self.c_paths)
# 创建一个检测器类，包含了图片的读取，检测等方法
class Detector(object):
    # netkind为'large'或'small'可以选择加载MobileNetV3_large或MobileNetV3_small
    # 需要事先训练好对应网络的权重
    def __init__(self,net_kind,num_classes=17):
        super(Detector, self).__init__()
        kind=net_kind.lower()
        if kind=='large':
            self.net = MobileNetV3_large(num_classes=num_classes)
        elif kind=='small':
            self.net = MobileNetV3_large(num_classes=num_classes)
        self.net.eval()
        if torch.cuda.is_available():
            self.net.cuda()

    def load_weights(self,weight_path):
        self.net.load_state_dict(torch.load(weight_path,map_location='cpu'))

    # 检测器主体
    def detect(self,weight_path,path_img_test):
        # 先加载权重
        self.load_weights(weight_path=weight_path)
        self.net.eval()
        trace_model = torch.jit.trace(self.net, torch.Tensor(1, 3, 224, 112))
        trace_model.save('./mobilenetv3.pt')

        # 读取图片

        image_datasets = sunDatasetInfer(path_img_test)
        dataset_loaders = torch.utils.data.DataLoader(image_datasets, batch_size=8, shuffle=False,
                                                      num_workers=0)

        for data in dataset_loaders:
            img_tensor, sample_path = data


            if torch.cuda.is_available():
                img_tensor=img_tensor.cuda()
            net_output = self.net(img_tensor)
            preds = F.softmax(net_output, dim=1)
            print(preds)
            print(net_output)
            _, predicted = torch.max(net_output.data, 1)

            result = predicted[0].item()
            print("预测的结果为：",class_dict[result])

if __name__=='__main__':
    class_dict = {0: 'teacher', 1: 'student', 2: 'other'}
    detector=Detector('large',num_classes=3)
    detector.detect('E:\python_project\stand_class\my_mobilenetv3\MobileNetV3_large_best.pth',r'E:\python_project\stand_class\my_mobilenetv3\00008744.jpg')







