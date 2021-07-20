import torch.nn as nn
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import scipy.io
import yaml
import math

#
# nclasses = 751
# stride = 2
#
# net = ft_net(nclasses, stride = stride)
# # net.classifier.classifier = nn.Sequential()
#

onnx_model_path = './se_resnext50_32x4d_224_112.onnx'
#
#
# save_path = "./model/warm5_s1_b8_lr2_p0.5/net_last.pth"
#
# net.load_state_dict(torch.load(save_path))
# net.classifier.classifier = nn.Sequential()
#
# net.cuda()
# net.eval()
#

# # dummy_input = torch.randn(1, 3, 64, 128)
# dummy_input = torch.randn(1, 3, 256, 128).to("cuda")
# torch.onnx.export(net, (dummy_input), onnx_model_path, verbose=True, output_names=['classifier'])
net_weight =  r"/workspace/stand_class/output/se_resnext50_32x4d/se_resnext50_32x4d_best.pth"
model = torch.load(net_weight, map_location='cpu')
model.cuda()
model.eval()
dummy_input = torch.randn(1, 3,224, 112).to("cuda")
torch.onnx.export(model, (dummy_input), onnx_model_path, verbose=True, output_names=['_classifier'])