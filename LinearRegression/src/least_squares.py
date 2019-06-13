# -*- coding: utf-8 -*-
# @Time     : 2019/6/12 20:53
# @Author   : gaol
# @Project  : ml
# @File     : least_squares.py

import sys
import os
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# print(sys.path)
import time
import numpy as np
from data_processing import *

def least_squares(feature, label):
    w = (feature.T * feature).I * feature.T * label
    return w

def predict(data, w):
    return data * w

if __name__ == '__main__':
    # 训练
    feature, label, train_data = load_data('./../data/data.txt','train')
    start_time = time.time()
    w = least_squares(feature, label)
    print(f'least_squares训练耗时：{time.time() - start_time} sec')  # 0.0 sec
    print(w)    # w0 = 0.00310499,w1 = 0.99450247

    # 测试
    test_data = load_data('./../data/data.txt','test')
    res = predict(test_data, w)

    # 绘图
    plot(train_data, w, test_data, res, '最小二乘法')