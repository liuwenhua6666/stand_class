# -*- coding: utf-8 -*-
"""
@Time ： 2021/7/12 13:57
@Auth ： 刘文华
@File ：train.py
@IDE ：PyCharm
@Motto：coding and code (Always Be Coding)
"""
# -*- coding: utf-8 -*-
import sys
sys.path.append('./data')
sys.path.append('./model')
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
import time
import os
import numpy as np
import torchvision
from torch.autograd import Variable
import albumentations as A
from torch.nn import functional as F
from sklearn.metrics import f1_score,precision_recall_fscore_support, accuracy_score
from albumentations.pytorch import ToTensorV2
from albumentations import FancyPCA
from torch.utils.data import Dataset, DataLoader
from log import get_logger
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from dataset import sunDataset
from model import MobileNetV3_large
from MobileNetV2 import mobilenetv2
from resnet import seResNet18
import argparse

parser = argparse.ArgumentParser(description='PyTorch stand Training')

parser.add_argument('--model', default="mobilenetv2", type=str,
                    help='model type (default: mobilenetv2)')
parser.add_argument('--batch_size', default=16, type=int, help='batch size')
parser.add_argument('--epoch', default=200, type=int,help='total epochs to run')
parser.add_argument('--GPU', default=1, type=int,help='total epochs to run')
parser.add_argument('--lr', default=1, type=float,help='total epochs to run')
args = parser.parse_args()




#宏定义一些数据，如epoch数，batchsize等
NUM_EPOCHS=args.epoch
TRAIN_BATCH_SIZE = args.batch_size
VAL_BATCH_SIZE = args.batch_size
LR=args.lr
log_interval=3
val_interval=1
LOG_DIR='logKfold/'
torch.cuda.empty_cache()
GPU_ID = args.GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(GPU_ID)

LOG_DIR='logKfold/'
if not os.path.exists(LOG_DIR):
   os.makedirs(LOG_DIR)
# 保存模型路径
OUT_DIR = 'output/'

AL_INTERVAl = 1
# 打印间隔STEP
PRINT_INTERVAL = 40
MIN_SAVE_EPOCH = 5

# ============================ step 1/5 数据 ============================
split_dir=os.path.join(".","data","dataset")
train_dir=os.path.join(split_dir,"train")
valid_dir=os.path.join(split_dir,"val")
#对训练集所需要做的预处理

def loaddata(train_dir, batch_size, shuffle,is_train=True):
    image_datasets = sunDataset(train_dir,is_train=is_train)
    dataset_loaders = torch.utils.data.DataLoader(image_datasets, batch_size=batch_size, shuffle=True, num_workers=1)
    data_set_sizes = len(image_datasets)
    return dataset_loaders, data_set_sizes


# ============================ step 2/5 模型训练 ============================

def train_model(model, criterion,cos_scheduler, optimizer,optimizer_lr, num_epochs=NUM_EPOCHS, model_name=None, train_dir=None,
                val_dir=None):
    train_loss = []
    since = time.time()
    best_model_wts = model.state_dict()
    best_acc = 0.0
    model.train()
    logger.info('start training...')
    for epoch in range(1, NUM_EPOCHS + 1):
        begin_time = time.time()
        data_loaders, dset_sizes = loaddata(train_dir=train_dir, batch_size=TRAIN_BATCH_SIZE, shuffle=True,
                                            is_train=True)
        logger.info('Epoch {}/{}'.format(epoch, NUM_EPOCHS))
        logger.info('-' * 10)
        optimizer = optimizer_lr(optimizer, epoch)
        running_loss = 0.0
        running_corrects = 0
        count = 0
        for i, data in enumerate(data_loaders):
            count += 1
            inputs, labels, img_path = data
            labels = labels.type(torch.LongTensor)
            inputs, labels = inputs.cuda(), labels.cuda()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs.data, 1)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # cos_scheduler.step()

            if i % PRINT_INTERVAL == 0 or outputs.size()[0] < TRAIN_BATCH_SIZE:
                spend_time = time.time() - begin_time
                logger.info(' Epoch:{}({}/{})-{} loss:{:.3f} epoch_Time:{}min:'.format(epoch, count,
                                                                                    dset_sizes // TRAIN_BATCH_SIZE,
                                                                                    model_name,
                                                                                    loss.item(),
                                                                                    spend_time / count * dset_sizes / TRAIN_BATCH_SIZE // 60 - spend_time // 60))

                train_loss.append(loss.item())
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        # print("%d-epoch-lr:%f" % (epoch, optimizer.param_groups[0]['lr']))

        val_acc = test_model(model, criterion, val_dir)
        epoch_loss = running_loss / dset_sizes
        epoch_acc = running_corrects.double() / dset_sizes
        logger.info('Epoch:[{}/{}]\t Loss={:.5f}\t Acc={:.3f}'.format(epoch, NUM_EPOCHS, epoch_loss, epoch_acc))
        if val_acc > best_acc and epoch > MIN_SAVE_EPOCH:
            best_acc = val_acc
            best_model_wts = model.state_dict()
        if val_acc > 0.999:
            break
        save_dir = os.path.join(OUT_DIR, model_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        model_out_path = save_dir + "/" + '{}_'.format(model_name) + str(epoch) + '.pth'
        if epoch % 1 == 0 and epoch > MIN_SAVE_EPOCH:
            # 只保存最好的模型，占空间太大
            pass
            # torch.save(model, model_out_path)
    # save best model
    logger.info('Best Accuracy: {:.3f}'.format(best_acc))
    model.load_state_dict(best_model_wts)
    model_out_path = save_dir + "/" + '{}_best.pth'.format(model_name)
    torch.save(best_model_wts, model_out_path)
    time_elapsed = time.time() - since
    logger.info('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    return model_out_path

# ============================ step 3/5 模型验证 ============================
def test_model(model, criterion,val_dir=None):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    cont = 0
    outPre = []
    outLabel = []
    pres_list=[]
    labels_list=[]
    data_loaders, dset_sizes = loaddata(train_dir=val_dir, batch_size=VAL_BATCH_SIZE,  shuffle=False, is_train=False)
    for data in data_loaders:
        inputs, labels ,img_path = data
        labels = labels.type(torch.LongTensor)
        inputs, labels = inputs.cuda(), labels.cuda()
        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)
        loss = criterion(outputs, labels)
        if cont == 0:
            outPre = outputs.data.cpu()
            outLabel = labels.data.cpu()
        else:
            outPre = torch.cat((outPre, outputs.data.cpu()), 0)
            outLabel = torch.cat((outLabel, labels.data.cpu()), 0)
        pres_list+=preds.cpu().numpy().tolist()
        labels_list+=labels.data.cpu().numpy().tolist()
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        cont += 1
    _,_, f_class, _= precision_recall_fscore_support(y_true=labels_list, y_pred=pres_list,labels=[0,1,2], average=None)
    fper_class = {'teacher': f_class[0], 'student': f_class[1],'other': f_class[2]}
    logger.info('clssse_F1:{}  class_F1_average:{}'.format(fper_class, f_class.mean()))
    logger.info('val_size: {}  valLoss: {:.4f} valAcc: {:.4f}'.format(dset_sizes, running_loss / dset_sizes, running_corrects.double() / dset_sizes))
    return running_corrects.double() / dset_sizes

def exp_lr_scheduler(optimizer, epoch, init_lr=LR, lr_decay_epoch=5):
    LR = init_lr * (0.8**(epoch / lr_decay_epoch))
    logger.info('Learning Rate is {:.5f}'.format(LR))
    print(logger)
    for param_group in optimizer.param_groups:
        param_group['LR'] = LR
    return optimizer

if __name__ == '__main__':
    # ============================ step 2/5 模型 ============================
    model_name = args.model
    if model_name == 'mobilenetv2':
        model = mobilenetv2(num_classes=3)
    elif model_name =='MobileNetV3_large':
        model = MobileNetV3_large(num_classes=3)
    else:
        print(2222)
        model_name = "seResNet18"
        model = seResNet18()
    if torch.cuda.is_available():
        model.cuda()
    # ============================ step 3/5 损失函数 ============================
    criterion = nn.CrossEntropyLoss()
    # ============================ step 4/5 优化器 ============================
    # exp_lr_scheduler = optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.99))  # 选择优化器
    MOMENTUM = 0.9
    optimizer = optim.SGD((model.parameters()), lr=LR, momentum=MOMENTUM, weight_decay=0.0004)


    cos_scheduler = CosineAnnealingWarmRestarts(optimizer,T_0=5, T_mult=2)

    # ============================ step 5/5 训练 ============================
    logger = get_logger(LOG_DIR + model_name + '.log')
    model_out_path = train_model(model, criterion, cos_scheduler,optimizer,optimizer_lr=exp_lr_scheduler
                                 , num_epochs=NUM_EPOCHS, model_name=model_name
                                 , train_dir=train_dir, val_dir=valid_dir)
    model_out_path = os.path.join('output/', model_name) + "/" + '{}_best.pth'.format(model_name)
    torch.cuda.empty_cache()

