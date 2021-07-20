# -*- coding: utf-8 -*-
"""
@Time ： 2021/7/2 11:29
@Auth ： 刘文华
@File ：pt_to_rknn.py
@IDE ：PyCharm
@Motto：coding and code (Always Be Coding)
"""
from rknn.api import RKNN


if __name__ == '__main__':


    model = './resnet18.pt'
    input_size_list = [[3, 240, 240]]

    # Create RKNN object
    rknn = RKNN()

    # pre-process config
    print('--> Config model')
    rknn.config(channel_mean_value='0 0 0 1', reorder_channel='0 1 2', batch_size=10)
    print('done')

    # Load Pytorch model
    print('--> Loading model')
    ret = rknn.load_pytorch(model=model, input_size_list=input_size_list)
    if ret != 0:
        print('Load Pytorch model failed!')
        exit(ret)
    print('done')

    # Build model
    print('--> Building model')
    ret = rknn.build(do_quantization=False)
    # ret = rknn.build(do_quantization=True, dataset='./dataset.txt')
    if ret != 0:
        print('Build model failed!')
        exit(ret)
    print('done')

    # Export RKNN model
    print('--> Export RKNN model')
    ret = rknn.export_rknn('./resnet_18.rknn')
    if ret != 0:
        print('Export resnet_18.rknn failed!')
        exit(ret)
    print('done')