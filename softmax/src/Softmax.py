# -*- coding: utf-8 -*-
# @Time     : 2019/6/3 19:52
# @Author   : gaol
# @Project  : ml
# @File     : Softmax.py

import numpy as np
import random as rd

def load_train_data(inputfile):
    feature_data = [];label_data = []

    f = open(inputfile, 'r')
    for line in f:
        feature_tmp = [1]
        lineArr = line.strip().split()
        for i in range(len(lineArr)-1):
            feature_tmp.append(float(lineArr[i]))
        label_data.append(int(lineArr[-1]))
        
        feature_data.append(feature_tmp)
    return np.mat(feature_data),np.mat(label_data).T,len(set(label_data))

def generate_test_data(num,n):  # 测试样本矩阵(num×n)
    testData = np.mat(np.ones((num,n)))
    for i in range(num):
        testData[i,1] = rd.random()*6-3
        testData[i,2] = rd.random()*15
    return testData

def gradientDescent(feature_data,label_data,k,maxCycle,alpha):  # k表示类别数目
    m, n = np.shape(feature_data)   # m = 135，n = 3

    # 初始化权重
    weights = np.mat(np.ones((n,k)))
    i = 0
    while i <= maxCycle:
        # ezj = e^({θ_j^T}X(i))
        ezj = np.exp(feature_data * weights)    # (135×4)的矩阵

        if i % 100 == 0:
            print(f'iter:{i},cost:{cost(ezj, label_data)}')

        rowsum = -ezj.sum(axis=1)
        rowsum = rowsum.repeat(k, axis=1)
        err = ezj / rowsum
        for x in range(m):
            err[x, label_data[x,0]] += 1    # 只有当“y(i)=j”时才加1

        weights += (alpha / m) * feature_data.T * err
        i += 1
    return weights

def cost(err,label_data):
    m = np.shape(err)[0]
    sum_cost = 0

    for i in range(m):
        if err[i,label_data[i,0]] / np.sum(err[i,:]) > 0:   # “err[i,label_data[i,0]”表示第i个样本在其分类下的误差
            sum_cost -= np.log(err[i,label_data[i,0]] / np.sum(err[i,:]))
        else:
            sum_cost -= 0
    return sum_cost / m

def predict(test_data,weights):
    h = test_data * weights # (4000×4)的矩阵
    return h.argmax(axis=1)

if __name__ == '__main__':
    feature, label, k = load_train_data('G:/master/python/PycharmProjects/ml/softmax/data/SoftInput.txt') # k = 4

    weights = gradientDescent(feature,label,k,10000,0.4)

    # 导入测试数据
    test_data = generate_test_data(4000, np.shape(weights)[0])
    print(test_data)

    # 预测
    result = predict(test_data,weights)
    print(result)






















