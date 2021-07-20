# -*- coding: utf-8 -*-
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
from albumentations.pytorch import ToTensorV2
from albumentations import FancyPCA
from torch.utils.data import Dataset, DataLoader
from log import get_logger
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from dataset import sunDataset
from cnn_finetune import make_model
# from efficientnet_pytorch import EfficientNet

torch.cuda.empty_cache()
GPU_ID = 0
os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(GPU_ID)

LOG_DIR='logKfold/'
if not os.path.exists(LOG_DIR):
   os.makedirs(LOG_DIR)
# 保存模型路径
OUT_DIR = 'output/'
#
TRAIN_BATCH_SIZE = 16
VAL_BATCH_SIZE= 8
TEST_BATCH_SIZE = 8
MOMENTUM = 0.9
NUM_EPOCHS = 50
LR = 0.0005
VAL_INTERVAl = 1
# 打印间隔STEP
PRINT_INTERVAL = 40
# 最低保存模型/计算最优模型epohc阈值
MIN_SAVE_EPOCH = 5
def loaddata(train_dir, batch_size, shuffle,is_train=True):
    image_datasets = sunDataset(train_dir,is_train=is_train)
    dataset_loaders = torch.utils.data.DataLoader(image_datasets, batch_size=batch_size, shuffle=True, num_workers=2)
    data_set_sizes = len(image_datasets)
    return dataset_loaders, data_set_sizes

def train_model(model, criterion, optimizer, lr_scheduler, num_epochs=NUM_EPOCHS,model_name=None,train_dir=None,val_dir=None):
    train_loss = []
    since = time.time()
    best_model_wts = model.state_dict()
    best_acc = 0.0
    model.train(True)
    logger.info('start training...')
    for epoch in range(1,NUM_EPOCHS+1):
        begin_time=time.time()
        data_loaders, dset_sizes = loaddata(train_dir=train_dir, batch_size=TRAIN_BATCH_SIZE, shuffle=True, is_train=True)
        logger.info('Epoch {}/{}'.format(epoch, NUM_EPOCHS))
        logger.info('-' * 10)
        optimizer = lr_scheduler(optimizer, epoch)
        running_loss = 0.0
        running_corrects = 0
        count=0
        for i, data in enumerate(data_loaders):
            count+=1
            inputs, labels,img_path = data
            labels = labels.type(torch.LongTensor)
            inputs, labels = inputs.cuda(), labels.cuda()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs.data, 1)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if i % PRINT_INTERVAL == 0 or outputs.size()[0] < TRAIN_BATCH_SIZE:
                spend_time = time.time() - begin_time
                logger.info(' Epoch:{}({}/{}) loss:{:.3f} epoch_Time:{}min:'.format(epoch, count, dset_sizes // TRAIN_BATCH_SIZE,
                                                                         loss.item(),
                                                                         spend_time / count * dset_sizes / TRAIN_BATCH_SIZE // 60-spend_time//60))
        
                train_loss.append(loss.item())
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
           
        val_acc = test_model(model, criterion,val_dir)
        epoch_loss = running_loss / dset_sizes
        epoch_acc = running_corrects.double() / dset_sizes
        logger.info('Epoch:[{}/{}]\t Loss={:.5f}\t Acc={:.3f}'.format(epoch , NUM_EPOCHS, epoch_loss, epoch_acc))
        if val_acc > best_acc and epoch > MIN_SAVE_EPOCH:
            best_acc = val_acc
            best_model_wts = model.state_dict()
        if val_acc > 0.999:
            break
        save_dir = os.path.join(OUT_DIR,model_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        model_out_path = save_dir + "/" + '{}_'.format(model_name)+str(epoch) + '.pth'
        if epoch % 1 == 0 and epoch > MIN_SAVE_EPOCH:
            #只保存最好的模型，占空间太大
            pass
            # torch.save(model, model_out_path)
    # save best model
    logger.info('Best Accuracy: {:.3f}'.format(best_acc))
    model.load_state_dict(best_model_wts)
    model_out_path = save_dir + "/" + '{}_best.pth'.format(model_name)
    torch.save(model, model_out_path)
    time_elapsed = time.time() - since
    logger.info('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    return model_out_path


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
                                                  
def view_train_pic():
    data_loaders, _ = loaddata(train_dir=None, batch_size=TRAIN_BATCH_SIZE, shuffle=True, is_train=True)
    pic, label = next(iter(data_loaders))
    img = torchvision.utils.make_grid(pic)
    img = img.numpy().transpose([1,2,0])
    plt.imshow(img)
    plt.show()

def exp_lr_scheduler(optimizer, epoch, init_lr=LR, lr_decay_epoch=5):
    LR = init_lr * (0.8**(epoch / lr_decay_epoch))
    logger.info('Learning Rate is {:.5f}'.format(LR))
    for param_group in optimizer.param_groups:
        param_group['LR'] = LR
    return optimizer

def make_classifier(in_features, num_classes):
    return nn.Sequential(nn.Linear(in_features, 64),nn.ReLU(inplace=True),nn.Linear(32, num_classes))


if __name__ == "__main__":
    # model_list = ['se_resnext50_32x4d','resnet34','resnet18','se_resnet50','densenet121','xception','mobilenet_v2', 'EfficientNet'] #'shufflenet_v2_x1_0'
    # model_list = ['inceptionresnetv2','polynet','dpn68','senet154'] #'shufflenet_v2_x1_0'
    # model_list = ['se_resnet50'] #'shufflenet_v2_x1_0','squeezenet1_1','vgg11',
    # model_list = ['resnet18']
    model_list = ['se_resnext50_32x4d']
    for model_name in model_list:
        # model_name = 'mobilenet_v2'
        logger = get_logger(LOG_DIR + model_name+'.log')
        train_dir = 'dataset' + '/train_224_224'
        val_dir = 'dataset' + '/val'
        logger.info('Using: {}'.format(model_name))
        print(model_name)

        model  = make_model('{}'.format(model_name), num_classes=3, pretrained=True, input_size=(224,112))
        # model  = make_model('{}'.format(model_name), num_classes=2, pretrained=True, input_size=(300,300),classifier_factory = make_classifier)


        criterion = nn.CrossEntropyLoss().cuda()
        model = model.cuda()
        optimizer = optim.SGD((model.parameters()), lr=LR, momentum=MOMENTUM, weight_decay=0.0004)
        cos_scheduler = CosineAnnealingWarmRestarts(optimizer, 3)
        model_out_path= train_model(model, criterion, optimizer, lr_scheduler=exp_lr_scheduler
                                    , num_epochs=NUM_EPOCHS,model_name=model_name
                                    ,train_dir=train_dir,val_dir=val_dir)
        model_out_path=os.path.join('output/',model_name) + "/" + '{}_best.pth'.format(model_name)
        torch.cuda.empty_cache()


