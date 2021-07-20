# -*- coding: utf-8 -*-
"""
@Time ： 2021/2/1 17:00
@Auth ： 刘文华
@File ：data_split.py
@IDE ：PyCharm
@Motto：coding and code (Always Be Coding)
"""
import os
import glob
import numpy as np
import shutil
from PIL import Image


def pad_image(img_path, target_size, output):
    image = Image.open(img_path)
    iw, ih = image.size  # 原始图像的尺寸
    w, h = target_size  # 目标图像的尺寸
    scale = min(float(w) / float(iw), float(h) / float(ih))  # 转换的最小比例

    # 保证长或宽，至少一个符合目标图像的尺寸
    nw = int(iw * scale)
    nh = int(ih * scale)
    image = image.resize((nw, nh), Image.BICUBIC)  # 采用双三次插值算法缩小图像
    new_image = Image.new('RGB', target_size, (128, 128, 128))  # 生成灰色图像
    # // 为整数除法，计算图像的位置
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))  # 将图像填充为中间图像，两侧为灰色的样式
    # new_image.show()
    new_image.save(output)
def main():
    pad = False
    size = (224, 224)

    #获取工作目录的绝对路径
    wd = os.getcwd()
    #判断是否存在保存路劲
    dataset_path = os.path.join(wd,"dataset")
    train_stu = "dataset/train/student"
    train_tea = "dataset/train/teacher"
    train_oth = "dataset/train/other"
    val_stu = "dataset/val/student"
    val_tea = "dataset/val/teacher"
    val_oth = "dataset/val/other"
    if not os.path.exists(train_stu):
        os.makedirs(train_stu)
    if not os.path.exists(train_tea):
        os.makedirs(train_tea)
    if not os.path.exists(train_oth):
        os.makedirs(train_oth)
    if not os.path.exists(val_stu):
        os.makedirs(val_stu)
    if not os.path.exists(val_tea):
        os.makedirs(val_tea)
    if not os.path.exists(val_oth):
        os.makedirs(val_oth)

    c_paths = glob.glob(dataset_path + '/*/*.jpg')
    c_paths = np.array(c_paths)
    np.random.seed(23)
    np.random.shuffle(c_paths)
    data_path_split = np.array_split(c_paths, 6)
    for index, data in enumerate(data_path_split):
        if index <5:
            for img_path in data:
                catogorya_name = img_path.split("/")[-2]
                img_name = os.path.split(img_path)[1]
                train = "dataset/train/"+catogorya_name+"/"+img_name
                if pad:
                    pad_image(img_path, size, train)
                else:
                    shutil.copy(img_path, "dataset/train/"+catogorya_name)


        else:
            for img_path in data:
                catogorya_name = img_path.split("/")[-2]
                img_name = os.path.split(img_path)[1]
                val = "dataset/val/" + catogorya_name +"/"+img_name
                if pad:
                    pad_image(img_path, size, val)
                else:
                    shutil.copy(img_path, "dataset/val/" + catogorya_name)


if __name__ == '__main__':
    main()