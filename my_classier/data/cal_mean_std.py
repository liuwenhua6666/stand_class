# -*- coding: utf-8 -*-
"""
@Time ： 2021/2/2 13:47
@Auth ： 刘文华
@File ：cal_mean_std.py
@IDE ：PyCharm
@Motto：coding and code (Always Be Coding)
"""

import cv2, os, argparse
import numpy as np
from tqdm import tqdm
import glob

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str,default=r"/workspace/stand_class/dataset/train_224_224")
    args = parser.parse_args()
    return args

def main():
    opt = parse_args()
    # img_filenames = os.listdir(opt.dir)
    img_filenames = glob.glob(opt.dir + '/*/*')
    m_list, s_list = [], []
    for img_filename in tqdm(img_filenames):
        img = cv2.imread(img_filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img / 255.0
        m, s = cv2.meanStdDev(img)
        m_list.append(m.reshape((3,)))
        s_list.append(s.reshape((3,)))
    m_array = np.array(m_list)
    s_array = np.array(s_list)
    m = m_array.mean(axis=0, keepdims=True)
    s = s_array.mean(axis=0, keepdims=True)
    print(m[0][::-1])
    print(s[0][::-1])

if __name__ == '__main__':
    main()
