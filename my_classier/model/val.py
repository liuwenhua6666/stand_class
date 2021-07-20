# -*- coding: utf-8 -*-
"""
@Time ： 2021/7/3 10:04
@Auth ： 刘文华
@File ：val.py
@IDE ：PyCharm
@Motto：coding and code (Always Be Coding)
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
import time
import os
import pdb
import numpy as np
import torchvision
import matplotlib.pyplot as plt
from torch.autograd import Variable
import albumentations as A
from torch.nn import functional as F
from sklearn.metrics import f1_score,precision_recall_fscore_support, accuracy_score
from log import get_logger
from albumentations.pytorch import ToTensorV2
from albumentations import FancyPCA
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from dataset import sunDataset
from cnn_finetune import make_model
import glob
import time
from shutil import copyfile

torch.cuda.empty_cache()
GPU_ID = 1
os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(GPU_ID)
criterion = nn.CrossEntropyLoss().cuda()
val_dir = 'dataset' + '/val'
# val_dir = 'val_data'

def loaddata(train_dir, batch_size, shuffle,is_train=True):
    image_datasets = sunDataset(train_dir,is_train=is_train)
    dataset_loaders = torch.utils.data.DataLoader(image_datasets, batch_size=batch_size, shuffle=shuffle, num_workers=2)
    data_set_sizes = len(image_datasets)
    return dataset_loaders, data_set_sizes

def test_model(model, criterion,val_dir=None):
    # os.mkdir("filter_teacher")
    # os.mkdir("filter_student")
    # os.mkdir("filter_other")
    start_time = time.time()
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    cont = 0
    outPre = []
    outLabel = []
    pres_list=[]
    labels_list=[]
    data_loaders, dset_sizes = loaddata(train_dir=val_dir, batch_size=8,  shuffle=False, is_train=False)
    allCnt = 0
    valideCnt = 0
    for data in data_loaders:
        inputs, labels ,smpth= data
        labels = labels.type(torch.LongTensor)
        inputs, labels = inputs.cuda(), labels.cuda()
        outputs = model(inputs)
        preds = F.softmax(outputs, dim=1)

        #smpth = list(smpth)
        #kk = preds.cpu()
        #xx = kk.detach().numpy().tolist()
        #xx = np.column_stack((xx,smpth))
        # for x in xx:
        #     allCnt += 1
        #     if(float(x[0])>0.90 ):
        #         valideCnt += 1
        #         copyfile(x[3],"./filter_teacher/"+os.path.split(x[3])[1])
        #     elif( float(x[1])>0.90):
        #         valideCnt += 1
        #         #print(x[0],x[1],os.path.split(x[2])[1] )
        #         copyfile(x[3],"./filter_student/"+os.path.split(x[3])[1])
        #     elif ( float(x[2])>0.50):
        #         copyfile(x[3], "./filter_other/" + os.path.split(x[3])[1])



        # _, preds = torch.max(preds, 1)
        _, preds = torch.max(outputs.data, 1)

        #print("----",preds)
        loss = criterion(outputs, labels)
        if cont == 0:
            outPre = outputs.data.cpu()
            outLabel = labels.data.cpu()
        else:
            outPre = torch.cat((outPre, outputs.data.cpu()), 0)
            outLabel = torch.cat((outLabel, labels.data.cpu()), 0)
        pres_list+=preds.cpu().numpy().tolist()
        # pres_list+=preds.numpy().tolist()
        labels_list+=labels.data.cpu().numpy().tolist()
        running_loss += loss.item() * inputs.size(0)
        # running_corrects += torch.sum(preds == labels.data.cpu())
        running_corrects += torch.sum(preds == labels.data)
        cont += 1
    _,_, f_class, _= precision_recall_fscore_support(y_true=labels_list, y_pred=pres_list,labels=[0,1,2], average=None)
    fper_class = {'teacher': f_class[0], 'student': f_class[1],'other': f_class[2]}
    print('clssse_F1:{}  class_F1_average:{}'.format(fper_class, f_class.mean()))
    print('val_size: {}  valLoss: {:.4f} valAcc: {:.4f}'.format(dset_sizes, running_loss / dset_sizes, running_corrects.double() / dset_sizes))
    print("spend_time is",time.time()-start_time)
def load_model(net_weight):
    # net_weight = r"/workspace/stand_class/output/se_resnext50_32x4d/se_resnext50_32x4d_best.pth"
    model = torch.load(net_weight)
    return model

if __name__ == '__main__':
    model_dir = r"/workspace/stand_class/output_mixup"
    model_list = glob.glob(model_dir + '/*/*')
    for model_path in model_list:
        if model_path.endswith(".pth"):
            model = load_model(model_path)
            model_name = os.path.split(model_path)[-1]
            print(model_name)
            test_model(model,criterion,val_dir)


