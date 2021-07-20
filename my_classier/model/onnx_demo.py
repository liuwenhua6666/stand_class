# -*- coding: utf-8 -*-
"""
@Time ： 2021/7/7 13:44
@Auth ： 刘文华
@File ：onnx_demo.py
@IDE ：PyCharm
@Motto：coding and code (Always Be Coding)
"""

from scipy import misc
import numpy as np
import onnx

import onnxruntime as ort

def load_model(model_path):
    # predictor = onnx.load(model_path)
    # onnx.checker.check_model(predictor)
    # onnx.helper.printable_graph(predictor.graph)
    ort_session = ort.InferenceSession(model_path)

    return  ort_session

def img_read(img):
    img1 = misc.imread(img)
    img1 = misc.imresize(img1, (224, 112), interp="bicubic")
    img1 = np.float32(img1)
    # mean=(0.453 ,0.422 ,0.428), std=(0.158, 0.163, 0.161)

    chanel_0 = img1[:, :, 0] / (255.0)
    chanel_1 = img1[:, :, 1] / (255.0)
    chanel_2 = img1[:, :, 2] / (255.0)

    i0 = (chanel_0 - 0.453) / 0.158
    i1 = (chanel_1 - 0.422) / 0.163
    i2 = (chanel_2 - 0.428) / 0.161

    img3 = np.stack([i0, i1, i2], axis=2)
    img3 = img3[np.newaxis, :, :]
    image = np.transpose(img3, [0, 3, 1, 2])
    return image


def softmax(x):
    return np.exp(x) / sum(np.exp(x))

if __name__ == '__main__':
    img1 = r"E:\python_project\stand_class\my_mobilenetv3\1.jpg"
    model_path = r"E:\python_project\stand_class\my_classier\model\se_resnext50_32x4d_224_112.onnx"
    img = img_read(img1)
    print(img.shape)
    model = load_model(model_path)
    input_name = model.get_inputs()[0].name
    out = model.run(None, {input_name: img})
    print(out)