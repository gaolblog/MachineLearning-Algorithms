# -*- coding: utf-8 -*-
# @Time     : 2019/6/13 10:32
# @Author   : gaol
# @Project  : ml
# @File     : gradient_descent.py

import time
import numpy as np
from data_processing import *

# 损失函数值
def err_rate(label, h):
    m = np.shape(h)[0]

    sum_err = 0
    for i in range(m):
        sum_err += (h[i,0] - label[i,0]) ** 2
    return sum_err / 2

# 梯度下降法求解模型参数。BGD:每迭代一次，最小化所有训练样本的损失函数
def lr_train_bgd(dataMat,labelVect,maxCycle,alpha):
    n = np.shape(dataMat)[1]            # 特征数目
    w = np.mat(np.ones((n, 1)))         # 初始化权重

    # 权值更新是用矩阵运算的
    i = 0
    while i < maxCycle:
        i += 1
        h = dataMat * w                 # “dataMat * w”就是WX(i)+b（i为第i个样本），得到的是(m,1)的矩阵（m为样本数目）
        err = h - labelVect             # “err”就是y(i)-hW,b(X(i))，得到的也是(m,1)的矩阵

        # 每做一次迭代更新都需要全部的训练数据参与（是一种批处理算法）
        w -=  alpha * dataMat.T * err   #  权重更新

        # 每迭代100次，输出一次损失函数的值
        if i % 100 == 0:
            print(f'iter={i}，损失函数值={err_rate(labelVect, h)}')
    return w

def predict(data, w):
    return data * w

if __name__ == '__main__':
    # 训练
    feature, label, train_data = load_data('./../data/data.txt','train')
    start_time = time.time()
    w = lr_train_bgd(feature, label, 1000, 0.001)   # 学习率α一定要选取恰当，否则会得出严重错误结果
    print(f'BGD训练耗时：{time.time() - start_time} sec')    # 0.018999576568603516 sec
    print(w)    # w0 = 0.0032078,w1 = 0.99420662
    # 测试
    test_data = load_data('./../data/data.txt','test')
    res = predict(test_data, w)
    # 绘图
    plot(train_data, w, test_data, res, '全批量梯度下降法')
