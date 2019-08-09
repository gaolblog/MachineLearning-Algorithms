# -*- coding: utf-8 -*-
# @Time     : 2018/1/30 15:44
# @Author   : gaol
# @Project  : ml
# @File     : Logistic.py

import random
import time
import numpy as np

# 数据加载
def load_data(file, train_flag):
    feature_data = []
    label_data = []

    fr = open(file, 'r')
    if train_flag is True:  # 加载训练集数据
        for line in fr:
            lineArr = line.strip().split()  # lineArr为一个字符串列表
            feature_data.append([1.0, float(lineArr[0]), float(lineArr[1])])
            label_data.append([int(lineArr[2])])
        
        return np.mat(feature_data),np.mat(label_data)
    else:                   # 加载测试集数据
        for line in fr:
            lineArr = line.strip().split()
            feature_data.append([1.0, float(lineArr[0]), float(lineArr[1])])
        return np.mat(feature_data)

# sigmoid函数
def sigmoid(z):
    return 1.0 / (1 + np.exp(-z))

# 损失函数值
def err_rate(label, h):
    m = np.shape(h)[0]

    sum_err = 0
    for i in range(m):
        if h[i,0]>0 and (1-h[i,0])>0:
            sum_err += (label[i,0]*np.log(h[i,0]) + (1-label[i,0])*np.log(1-h[i,0]))
        else:
            sum_err += 0

    return -sum_err / m

# 梯度下降法求解模型参数。BGD:每迭代一次，最小化所有训练样本的损失函数
def lr_train_bgd(dataMat,labelVect,maxCycle,alpha):
    n = np.shape(dataMat)[1]            # 特征数目
    w = np.mat(np.ones((n, 1)))         # 初始化权重

    start_time = time.time()
    # 权值更新是用矩阵运算的
    i = 0
    while i < maxCycle:
        i += 1
        h = sigmoid(dataMat * w)        # “dataMat * w”就是WX(i)+b（i为第i个样本），得到的是(m,1)的矩阵（m为样本数目）
        err = labelVect - h             # “err”就是y(i)-hW,b(X(i))，得到的也是(m,1)的矩阵
        # 每做一次迭代更新都需要全部的训练数据参与（是一种批处理算法）
        w +=  alpha * dataMat.T * err   #  权重更新

        # 每迭代100次，输出一次损失函数的值
        if i % 100 == 0:
            print(f'iter={i}，损失函数值={err_rate(labelVect, h)}')
    end_time = time.time()
    print(f'BGD训练耗时：{end_time - start_time} sec')
    return w

# 随机梯度下降法。SGD：每迭代一次，最小化每条训练样本的损失函数
def lr_train_sgd(dataMat,labelVect,maxCycle,alpha = 0.001):
    m, n = np.shape(dataMat)        # m为样本数目，n为特征数目
    w = np.mat(np.ones((n, 1)))     # 初始化权重

    alpha0 = 0.001          # 初始学习率
    decay_rate = 1          # 衰减率
    epoch_num = 0           # 代数
    epsilon = 1e-4          # 允许误差大小
    j = 0
    while j < maxCycle:
    # while True:
        j += 1
        dataIndex = list(range(m))
        loss = 0
        # alpha = alpha0 / (1+decay_rate*epoch_num)
        for i in range(m):
            randIndex = random.choice(dataIndex)

            h_ = sigmoid(dataMat[randIndex] * w)             # 误差
            err_ = labelVect[randIndex] - h_
            w += alpha * dataMat[randIndex].T * err_         # 随机的选择一条样本来更新权重

            loss += err_rate(labelVect[randIndex], h_)      # err_rate(labelVect, h)返回的是一条样本的损失函数值

            dataIndex.remove(randIndex)

        loss /= m
        # 每迭代100次，输出一次损失函数的值
        if j % 100 == 0:
            print(f'iter={j}，损失函数值={loss}')

        # epoch_num = j
        # if loss < epsilon:
        #     break

    return w

def lr_train_sgd2(dataMat,labelVect,maxCycle,alpha=0.001):
    m, n = np.shape(dataMat)        # m为样本数目，n为特征数目
    w = np.mat(np.ones((n, 1)))     # 初始化权重

    epsilon = 1

    dataIndex = [i for i in range(m)]
    loss_ = 0
    for i in range(m):
        # randIndex = int(random.uniform(0, m))
        # print(f'randIndex：{randIndex}')

        h = sigmoid(dataMat * w)
        err = labelVect - h
        # 随机取样本索引
        randIndex = random.choice(dataIndex)
        w += alpha * dataMat[randIndex].T * err[randIndex]      # 随机的选择一条样本来更新权重

        loss_ += err_rate(labelVect[randIndex], h[randIndex])   # err_rate(labelVect, h)返回的是一条样本的损失函数值
        # print(loss)

        # dataIndex.remove(randIndex)

        loss = loss_ / m
    # 每迭代100次，输出一次损失函数的值
    # if j % 100 == 0:
        print(f'iter={i}，损失函数值={loss}')

    # if loss < epsilon:
    #     break

    return w

# 预测新数据的类别
def predict(data, w):
    h = sigmoid(data * w)
    m = np.shape(h)[0]

    for i in range(m):
        if h[i, 0] < 0.5:
            h[i ,0] = 0
        else:
            h[i, 0] = 1
    return h

def plotBestFit(dataMat,labelVect,weights):
    import matplotlib.pyplot as plt

    m = np.shape(dataMat)[0]
    xcord1 = []; ycord1 = []    # 正样本
    xcord2 = []; ycord2 = []    # 负样本

    for i in range(m):
        if int(labelVect[i,0]) == 1:
            xcord1.append(dataMat[i,1]); ycord1.append(dataMat[i,2])
        else:
            xcord2.append(dataMat[i,1]); ycord2.append(dataMat[i,2])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('Logistic分离超平面', fontproperties="SimHei")
    train_label0 = ax.scatter(xcord2, ycord2, s=30, c='green')
    train_label1 = ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    plt.legend(handles=[train_label0, train_label1], labels=['0', '1'], loc='best')

    x = np.arange(0, 10.0, 0.1)
    y = (-weights[1,0]*x-weights[0,0])/weights[2,0]
    ax.plot(x,y)
    plt.xlabel('X1'); plt.ylabel("X2")
    # 训练数据图例说明
    plt.savefig('./../resources/Logistic分离超平面.png')
    plt.show()

# 主函数
if __name__ == '__main__':
    # 准备训练数据
    dataMat,labelVect = load_data('./../data/train_data', True)
    # dataMat,labelVect = load_data('G:/master/python/PycharmProjects/ml/logistic/data/testSet.txt', True)

    # 训练LR模型
    # w = lr_train_bgd(dataMat,labelVect,1000,0.001)
    start_time = time.time()
    w = lr_train_sgd(dataMat,labelVect,1000)
    end_time = time.time()
    print(f'SGD训练耗时：{end_time - start_time} sec')
    print(w)

    # plotDataSet(dataMat, labelVect)
    plotBestFit(dataMat, labelVect, w)

    # 准备测试数据
    # testData = load_data('G:/master/python/PycharmProjects/ml/logistic/data/test_data', False)

    # 对测试数据做分类预测
    # h = predict(testData, w)
    # print(h)