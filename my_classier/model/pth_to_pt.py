# -*- coding: utf-8 -*-
"""
@Time ： 2021/7/2 9:48
@Auth ： 刘文华
@File ：pth_to_pt.py
@IDE ：PyCharm
@Motto：coding and code (Always Be Coding)
"""
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn.functional as F
import os
import numpy as np

net_weight = r'E:\python_project\stand_class\my_mobilenetv3\MobileNetV3_large_best.pth'
model = torch.load(net_weight, map_location='cpu')
model.eval()
trace_model = torch.jit.trace(model, torch.Tensor(1,3,240,112))
trace_model.save('./mobilenet_v2_best.pt')