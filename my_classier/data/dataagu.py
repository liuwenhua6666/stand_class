# -*- coding: utf-8 -*-
"""
@Time ： 2021/7/1 9:39
@Auth ： 刘文华
@File ：dataagu.py
@IDE ：PyCharm
@Motto：coding and code (Always Be Coding)
"""


import albumentations as A
import cv2

# Declare an augmentation pipeline
transform = A.Compose([
    A.RandomCrop(width=120, height=240),
    # A.HorizontalFlip(p=0.5),
    # A.RandomBrightnessContrast(p=0.2)
    A.RandomGamma(gamma_limit=(10, 200), p=1),
    A.ShiftScaleRotate(rotate_limit=1, p=1),
    # A.VerticalFlip(p=1)
    # A.GaussNoise()
])

# Read an image with OpenCV and convert it to the RGB colorspace
image = cv2.imread("./00006894.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Augment an image
transformed = transform(image=image)
transformed_image = transformed["image"]
transformed_image = cv2.cvtColor(transformed_image, cv2.COLOR_RGB2BGR)
cv2.imwrite("2.jpg",transformed_image)
