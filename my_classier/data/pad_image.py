# -*- coding: utf-8 -*-

import os

from PIL import Image

import numpy as np


def pad_image(image, target_size, output):
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
    size = (240, 240)
    img_list = os.listdir(input_dir)

    for img in img_list:
        img_path = os.path.join(input_dir, img)
        image = Image.open(img_path)
        output_path = os.path.join(output_dir, img)
        print(output_path)
        pad_image(image, size, output_path)


if __name__ == '__main__':
    input_dir = r"F:\student_data\test_out"
    output_dir = r"F:\student_data\val_img"
    main()
