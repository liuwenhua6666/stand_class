# -*- coding: utf-8 -*-
"""
@Time ： 2021/5/21 14:47
@Auth ： 刘文华
@File ：demo_qt.py
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
import glob
from PIL import Image
import tkinter as tk
from tkinter import filedialog
from tkinter import *
import cv2
from PIL import Image, ImageFont, ImageDraw,ImageTk
import time



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
        img1 = pad_image(sample_path1)
        # img1 = Image.open(sample_path1)
        img1=np.array(img1)
        img = self.data_transforms(image=img1)['image']
        return img ,sample_path1

    def __len__(self):
        return len(self.c_paths)

def pad_image(image):
    new_image = Image.open(image)
    # target_size = (IMAGE_SIZE,IMAGE_SIZE)
    # iw, ih = image.size  # 原始图像的尺寸
    # w, h = target_size  # 目标图像的尺寸
    # scale = min(float(w) / float(iw), float(h) / float(ih))  # 转换的最小比例
    #
    # # 保证长或宽，至少一个符合目标图像的尺寸
    # nw = int(iw * scale)
    # nh = int(ih * scale)
    # image = image.resize((nw, nh), Image.BICUBIC)  # 采用双三次插值算法缩小图像
    # new_image = Image.new('RGB', target_size, (128, 128, 128))  # 生成灰色图像
    # # // 为整数除法，计算图像的位置
    # new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))  # 将图像填充为中间图像，两侧为灰色的样式
    return new_image


def pred_data(net_weight,path_img_test):
    image_datasets = sunDatasetInfer(path_img_test)
    dataset_loaders = torch.utils.data.DataLoader(image_datasets, batch_size=8, shuffle=False,
                                                  num_workers=0)
    # 加载最佳模型
    print(net_weight)
    model = torch.load(net_weight, map_location='cpu')
    # model.eval()
    pres_list = []
    name_list = []
    for data in dataset_loaders:
        inputs, sample_path= data
        pic_names = [os.path.split(path)[1][:-4] for path in sample_path]
        # inputs = Variable(inputs.cuda())
        xx = inputs.numpy()
        print(inputs.dtype)
        start_time = time.time()
        outputs = model(inputs)
        print(time.time()-start_time)
        print(outputs)
        preds = F.softmax(outputs, dim=1)
        print(preds)
        _, preds = torch.max(preds, 1)

        result = preds.cpu().numpy().tolist()
    return result[0]

def my_modelname():
    currentpath = os.path.abspath(os.path.dirname(__file__))
    model = os.path.join(currentpath, 'mobilenetv3.pt')
    return model

def slect_dir(event):

    global tkImage
    image_path = filedialog.askopenfilename()
    tkImage = open_img(image_path)
    w.create_image(0, 0, anchor=NW, image=tkImage)

def open_img(image_path):
    class_dict = {0: 'teacher', 1: 'student',2: 'other'}
    font = ImageFont.truetype(font='font/simsun.ttc', size=25)
    imgbak1 = Image.new("RGB", (900, 800), (255, 255, 255))
    img_PIL = Image.open(image_path)
    w,h =  img_PIL.size

    label = pred_data(model,image_path)
    result = class_dict[label]


    draw = ImageDraw.Draw(imgbak1)
    new_w = int((500/h) * w)
    pilImage640 = img_PIL.resize((new_w, 500))
    draw.text((380, 600), "识别结果："+ result, font=font, fill=(255, 0, 0))
    box = (300, 80, new_w + 300, 500 + 80)

    imgbak1.paste(pilImage640, box)
    imgbak = ImageTk.PhotoImage(image=imgbak1)
    return imgbak

def call_back(event):
    global g_current_pt
    # 按哪个键，在console中打印
    master.update()

if __name__ == '__main__':
    model = my_modelname()
    master = Tk()
    master.title("智能识别")
    w = Canvas(master,
               width=900,
               height=640)
    w.pack(expand=YES, fill=BOTH)
    w.bind("<Button-1>", call_back)
    b11 = Button(master, text="打开图片", width=10, height=2)
    b11.bind("<ButtonRelease-1>", slect_dir)
    b11.place(x=10, y=280)
    L2 = Label(master, text="学生老师识别", font=('隶书', 20), foreground=f'#{500:06x}')
    L2.place(x=350, y=0, width=250, height=50)
    mainloop()
