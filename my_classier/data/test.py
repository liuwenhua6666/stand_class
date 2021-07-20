import numpy as np
from scipy import misc
import random
import os
import glob
import shutil
x = random.randint(250,1600)
y = random.randint(250,800)
print(x,y)
index = 0
index = + 1
# print(index)
# print([0.453 ,0.422 ,0.428]*255)
# x = [0.158, 0.163, 0.161]
# print([a * 255 for a in x])


def img_read(img):
    img1 = misc.imread(img)
    #img1 = misc.imresize(img1,(256,128),interp = "bicubic")
    img1 = np.float32(img1)
    #mean=(0.453 ,0.422 ,0.428), std=(0.158, 0.163, 0.161)
    chanel_0 = img1[:, :, 0] /255.0
    chanel_1 = img1[:, :, 1] / (255.0)
    chanel_2 = img1[:, :, 2] / (255.0)

    i0 = (chanel_0 - 115.515) / 40.29
    # i0 = (chanel_0 - 0.453) / 0.158
    i1 = (chanel_1 - 0.422) / 0.163
    i2 = (chanel_2 - 0.428) / 0.161
    print(i0)
for i in range(100):
    lam = np.random.beta(0.1, 0.1)
    print(22222222222)
    print(lam)


