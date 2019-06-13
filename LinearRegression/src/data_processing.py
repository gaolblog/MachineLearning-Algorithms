# -*- coding: utf-8 -*-
# @Time     : 2019/6/12 20:41
# @Author   : gaol
# @Project  : ml
# @File     : data_processing.py

import numpy as np
import matplotlib.pyplot as plt

def load_data(file, flag):
    feature = [];label = []
    train_data = [];test_data = []

    if flag == 'train':
        f = open(file,'r')
        for line in f:
            lineArr = line.strip().split()  # lineArr为一个字符串列表
            feature.append([1, float(lineArr[0])])
            label.append(float(lineArr[-1]))
            train_data.append([float(lineArr[0]),float(lineArr[1])])
        f.close()
        return np.mat(feature), np.mat(label).T, np.mat(train_data)
    else:
        f = open(file, 'r')
        for line in f:
            lineArr = line.strip().split()
            test_data.append([1, float(lineArr[0])])
        f.close()
        return np.mat(test_data)


def plot(train_data,w,test_data,predict,method_name):
    # 创建绘图窗口
    fig = plt.figure()
    # 将绘图窗口分成1行1列，选择第一块区域作子图
    ax1 = fig.add_subplot(1, 1, 1)
    # 设置二维坐标
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    # 设置图像标题
    ax1.set_title(f'线性回归-{method_name} 拟合曲线图', fontproperties="SimHei")

    # 绘制训练数据散点图
    train = ax1.scatter(train_data[:,0].getA().flatten(), train_data[:,1].getA().flatten(), marker='.', c='c')

    # 绘制拟合曲线图
    x_points = np.linspace(0, 1, 20)
    y_ = w[1,0] * x_points + w[0,0]
    ax1.plot(x_points, y_, c='y')

    # 绘制测试数据散点图
    test = ax1.scatter(test_data[:,1].getA().flatten(), predict[:,0].getA().flatten(), marker='.', c='r')

    # 数据图例说明
    plt.legend(handles=[train, test], labels=['train', 'test'], loc='best')

    # 显示
    plt.savefig(f'./../resources/线性回归-{method_name} 拟合曲线图.png')
    plt.show()

if __name__ == '__main__':
    feature, label, train_data = load_data('./../data/data.txt','train')
    print(train_data[:,0])