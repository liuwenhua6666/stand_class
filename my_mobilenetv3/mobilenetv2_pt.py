# -*- coding: utf-8 -*-
"""
@Time ： 2021/7/13 10:02
@Auth ： 刘文华
@File ：mobilenetv2_pt.py
@IDE ：PyCharm
@Motto：coding and code (Always Be Coding)
"""
import torch
import sys
sys.path.append('./model')
from MobileNetV2 import mobilenetv2

weight_path = r'E:\python_project\stand_class\my_mobilenetv3\MobileNetV2_best.pth'
net = mobilenetv2(num_classes=3)
net.load_state_dict(torch.load(weight_path,map_location='cpu'))
net.eval()
trace_model = torch.jit.trace(net, torch.Tensor(1, 3, 224, 112))
trace_model.save('./mobilenetv2.pt')

