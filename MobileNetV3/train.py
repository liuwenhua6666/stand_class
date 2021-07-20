# -*- coding: utf-8 -*-

import sys
sys.path.append('./data')
sys.path.append('./model')
import os
os.environ['NLS_LANG'] = 'AMERICAN_AMERICA.AL32UTF8'
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
from matplotlib import pyplot as plt
from dataset import flowerDataset
from model import MobileNetV3_large
from model import MobileNetV3_small
import torchvision
from torch.autograd import Variable

#宏定义一些数据，如epoch数，batchsize等
MAX_EPOCH=50
BATCH_SIZE=64
LR=0.0001
log_interval=3
val_interval=1

# ============================ step 1/5 数据 ============================
split_dir=os.path.join(".","data","dataset")
train_dir=os.path.join(split_dir,"train")
valid_dir=os.path.join(split_dir,"valid")

#对训练集所需要做的预处理
train_transform=transforms.Compose([
    transforms.Resize((224,112)),
    transforms.ToTensor(),
])

#对验证集所需要做的预处理
valid_transform=transforms.Compose([
    transforms.Resize((224,112)),
    transforms.ToTensor(),
])

# 构建MyDataset实例
train_data=flowerDataset(data_dir=train_dir,transform=train_transform)
valid_data=flowerDataset(data_dir=valid_dir,transform=valid_transform)

# 构建DataLoader
# 训练集数据最好打乱
# DataLoader的实质就是把数据集加上一个索引号，再返回
train_loader=DataLoader(dataset=train_data,batch_size=BATCH_SIZE,shuffle=True)
valid_loader=DataLoader(dataset=valid_data,batch_size=BATCH_SIZE)

# ============================ step 2/5 模型 ============================
net=MobileNetV3_large(num_classes=3)
if torch.cuda.is_available():
    net.cuda()
# ============================ step 3/5 损失函数 ============================
criterion=nn.CrossEntropyLoss()
# ============================ step 4/5 优化器 ============================
optimizer=optim.Adam(net.parameters(),lr=LR, betas=(0.9, 0.99))# 选择优化器
# ============================ step 5/5 训练 ============================
# 记录每一次的数据，方便绘图
train_curve=list()
valid_curve=list()
net.train()
accurancy_global=0.0
for epoch in range(MAX_EPOCH):
    loss_mean=0.
    correct=0.
    total=0.
    running_loss = 0.0

    for i,data in enumerate(train_loader):
        img,label=data
        img = Variable(img)
        label = Variable(label)
        if torch.cuda.is_available():
            img=img.cuda()
            label=label.cuda()
        # 前向传播
        out=net(img)
        optimizer.zero_grad()  # 归0梯度
        loss=criterion(out,label)#得到损失函数

        print_loss=loss.data.item()

        loss.backward()#反向传播
        optimizer.step()#优化
        if (i+1)%log_interval==0:
            print('epoch:{},loss:{:.4f}'.format(epoch+1,loss.data.item()))
        _, predicted = torch.max(out.data, 1)
        total += label.size(0)
        # print("============================================")
        # print("源数据标签：",label)
        # print("============================================")
        # print("预测结果：",predicted)
        # print("相等的结果为：",predicted == label)
        correct += (predicted == label).sum()

    print("============================================")
    correct = correct.cpu().numpy()
    accurancy=correct / total
    print(accurancy)
    if accurancy>accurancy_global:
        torch.save(net.state_dict(), './weights/best.pkl')
        print("accurancy:", accurancy_global, "up:", accurancy, "save_weights/best.pkl")
        accurancy_global=accurancy
    print('%depoch_accurancy:%d' % (epoch + 1, 100*accurancy))
torch.save(net.state_dict(), './weights/last.pkl')
print("train_finish，save_weights:weights/last.pkl")
