import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import glob
import os
import cv2
import numpy as np
from torchvision import datasets, models, transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations import FancyPCA
class sunDataset(Dataset):
    def __init__(self, data_dir,is_train=True):
        c_dir=data_dir
        self.cls_dic={'teacher':0,'student':1,"other":2}
        self.c_paths=sorted(glob.glob(c_dir+'/*/*'))
        self.transform_train = A.Compose([
            A.Resize(height=224, width=112),
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.2),
                A.RandomBrightness(limit=0.1, p=0.2),
                A.RandomGamma(gamma_limit=(80, 120), p=0.2),  # +
            ], p=1),
            A.GaussNoise(),
            A.HorizontalFlip(p=0.2),  # 水平翻转
            # A.VerticalFlip(p=0.5), #垂直翻转
            # A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(rotate_limit=1, p=0.2),

            # FancyPCA(alpha=0.1, p=0.5),
            # blur
            A.OneOf([
                A.MotionBlur(blur_limit=3), A.MedianBlur(blur_limit=3), A.GaussianBlur(blur_limit=(1, 3)),
            ], p=0.2),
            # Pixels
            A.OneOf([
                A.IAAEmboss(p=0.2),
                A.IAASharpen(p=0.2),
            ], p=1),
            # Affine
            A.OneOf([
                A.ElasticTransform(p=0.2),
                A.IAAPiecewiseAffine(p=0.2),
            ], p=1),
            A.Normalize(mean=(0.453, 0.422, 0.428), std=(0.158, 0.163, 0.161), max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),
        ])

        self.transform_valid = A.Compose([
            A.Resize(height=224, width=112),
            A.Normalize(mean=(0.453, 0.422, 0.428), std=(0.158, 0.163, 0.161), max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),
        ])
        if is_train:
            self.data_transforms=self.transform_train
        else:
            self.data_transforms = self.transform_valid

    def __getitem__(self,index):
        #第index个样本
        sample_path1 = self.c_paths[index]
        # sample_path2 = self.m_paths[index]
        #
        cls = sample_path1.split('/')[-2]
        label = self.cls_dic[cls]

        img1 = Image.open(sample_path1)
        img1 = img1.convert("RGB")
        img1=np.array(img1)
        img = self.data_transforms(image=img1)['image']
        return img,label,sample_path1

    def __len__(self):
        return len(self.c_paths)
