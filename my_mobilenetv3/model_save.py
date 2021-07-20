# -*- coding: utf-8 -*-
"""
@Time ： 2021/7/12 16:54
@Auth ： 刘文华
@File ：model_save.py
@IDE ：PyCharm
@Motto：coding and code (Always Be Coding)
"""
import sys
sys.path.append('./model')
from model import MobileNetV3_large
from model import MobileNetV3_small
import torch

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

if __name__ == '__main__':
    weight_path = r'E:\python_project\stand_class\my_mobilenetv3\MobileNetV3_large_best.pth'
    detector=Detector('large',num_classes=3)
    detector.load_weights(weight_path=weight_path)
    detector.net.eval()
    trace_model = torch.jit.trace(detector.net, torch.Tensor(1, 3, 224, 112))
    trace_model.save('./mobilenetv3.pt')