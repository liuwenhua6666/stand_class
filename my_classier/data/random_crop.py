# -*- coding: utf-8 -*-
"""
@Time ： 2021/7/3 15:26
@Auth ： 刘文华
@File ：random_crop.py
@IDE ：PyCharm
@Motto：coding and code (Always Be Coding)

"""
import cv2
import os
import random
def crop():
    index = 0
    for img in os.listdir(img_dir):
        img_path = os.path.join(img_dir,img)
        image = cv2.imread(img_path)
        for i in range(10):
            x = random.randint(250, 1600)
            y = random.randint(250, 800)
            crop_img = image[y:y+240,x:x+120]
            pic_name = str(index).zfill(8) + ".jpg"
            cv2.imwrite(os.path.join(out_dir,pic_name),crop_img)
            index += 1



if __name__ == '__main__':
    img_dir = r"F:\student_data\test_input"
    out_dir = r"F:\student_data\test_out"
    crop()