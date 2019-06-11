# -*- coding: utf-8 -*-
# @Time     : 2019/5/28 20:19
# @Author   : gaol
# @Project  : ml
# @File     : Perceptron.py

from random import choice
import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    # 初始化数据集和标签
    def __init__(self, data, labels):
        self.__data = np.array(data)
        self.__labels = np.array(labels).transpose()
        self.__eta = 1

    def train(self):
        m, n = np.shape(self.__data)    # 对当前数据集是(3, 2)

        weights = np.zeros(n)
        bias = 0

        numIter = 0
        while True:
            numIter += 1
            dataIndexes = []
            # 遍历样本，寻找误分类点
            for i in range(m):
                if self.__labels[i] * (np.dot(weights, self.__data[i]) + bias) <= 0:
                    dataIndexes.append(i)

            if len(dataIndexes) == 0:   # 如果没有误分类点了，则停止更新模型参数
                break
            else:
                randIndex = choice(dataIndexes) # 随机选择一个误分类点做SGD更新
                # SGD更新权重和偏置
                weights += self.__eta * self.__labels[randIndex] * self.__data[randIndex]
                bias += self.__eta * self.__labels[randIndex]
                print(f'迭代次数：{numIter}，误分类点：x{randIndex+1}，w：{weights}，b：{bias}')

        return weights,bias

    def plot(self, w, b):
        # 创建绘图窗口
        fig = plt.figure()
        # 将绘图窗口分成1行1列，选择第一块区域作子图
        ax1 = fig.add_subplot(1, 1, 1)
        # 设置二维坐标
        ax1.set_xlabel('x1')
        ax1.set_ylabel('x2')
        # 设置标题
        ax1.set_title('Perceptron分离超平面', fontproperties="SimHei")
        # 绘制散点图
        train_label_positive = ax1.scatter(self.__data[:2,0], self.__data[:2,1], marker='o', c='c')
        train_label_negative = ax1.scatter(self.__data[2:,0], self.__data[2:,1], marker='D', c='m')
        # 训练数据图例说明
        plt.legend(handles=[train_label_positive, train_label_negative], labels=['1', '-1'], loc='best')
        # 绘制直线图（分离超平面）
        if w[1] != 0:
            x_points = np.linspace(0, 5, 20)
            y_ = -(w[0] * x_points + b) / w[1]  # Ax+By+C=0直线方程的斜截式：y=-(Ax+C)/B (B≠0)
            ax1.plot(x_points, y_, c='y')
        else:
            x_point = -b / w[0] # B=0时：直线方程为x = -C/A
            ax1.vlines(x_point, -5, 5, colors = 'y')
        # 显示
        plt.savefig('./../resources/Perceptron分离超平面2.png')
        plt.show()


if __name__ == '__main__':
    # 准备数据
    data = [[3, 3],
            [4, 3],
            [1, 1]]
    labels = [1, 1, -1]

    # 创建一个感知机对象
    perceptron = Perceptron(data, labels)
    # 学习参数
    w, b = perceptron.train()
    # 绘图
    perceptron.plot(w, b)

    print(f'最终的分离超平面：{w[0]}x1+{w[1]}x2+{b}')

