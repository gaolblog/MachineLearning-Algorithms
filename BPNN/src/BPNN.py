# -*- coding: utf-8 -*-
# @Time     : 2019/6/5 19:43
# @Author   : gaol
# @Project  : ml
# @File     : BPNN.py

import time
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt

"""
训练数据加载
"""
def load_data(file_name):
    feature_data = [];label_tmp = []
    # 获取特征
    f = open(file_name, 'r')
    for line in f:
        feature_tmp = []
        lineArr = line.strip().split('\t')
        for i in range(len(lineArr) - 1):
            feature_tmp.append(float(lineArr[i]))
        label_tmp.append(int(lineArr[-1]))
        feature_data.append(feature_tmp)
    f.close()

    # 获取标签
    m = len(label_tmp)
    n_class = len(set(label_tmp))   # 获取类别数目
    label_data = np.mat(np.zeros((m, n_class)))
    for i in range(m):
        label_data[i, label_tmp[i]] = 1

    return np.mat(feature_data),label_data,n_class

"""
生成测试数据
"""
def generate_testData():
    # 在[-4.5,4.5]之间随机生成20000组点
    testData = np.random.uniform(low=-4.5, high=4.5, size=(10000, 2))
    '''
    testData = np.mat(np.zeros((20000, 2)))
    m = np.shape(testData)[0]
    x = np.mat(np.random.rand(20000, 2))
    for i in range(m):
        testData[i, 0] = x[i, 0] * 9 - 4.5
        testData[i, 1] = x[i, 1] * 9 - 4.5
    '''
    return np.mat(testData)

"""
sigmoid函数
@param  x: float/mat
"""
def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))

"""
sigmoid函数的导数
@param  x: float/mat
"""
def partial_sigmoid(x):
    m, n = np.shape(x)
    out = np.mat(np.zeros((m, n)))
    for i in range(m):
        for j in range(n):
            out[i, j] = sigmoid(x[i, j]) * (1 - sigmoid(x[i, j]))

    return out

"""
计算隐含层输入
@param  feature: (400×20)维的样本
@param  w0: 输入层到隐含层之间的权重 
@param  b0: 输入层到隐含层之间的偏置
"""
def hidden_in(feature, w0, b0):
    m = np.shape(feature)[0]
    hidden_in = feature * w0    # (400×20)维。每一个样本的w_(1i)*x1+w_(2i)*x2 (i∈[1,20])，共400个样本针对20个神经元的输入加权和

    for i in range(m):
        hidden_in[i,] += b0     # 给每个样本的w_(1i)*x1+w_(2i)*x2 (i∈[1,20])加上偏置项b^{1}_(1i) (i∈[1,20])（“{1}表示第1层”）

    return hidden_in

"""
计算隐含层输出
@param  hidden_in: 隐含层输入
"""
def hidden_out(hidden_in):
    hidden_out = sigmoid(hidden_in)     # (400×20)维
    return hidden_out

"""
计算输出层输入
@param  hidden_out: (400×20)维的隐含层输出
@param  w1: 隐含层到输出层之间的权重 
@param  b1: 隐含层到输出层之间的偏置
"""
def predict_in(hidden_out, w1, b1):
    m = np.shape(hidden_out)[0]
    predict_in = hidden_out * w1        # (400×2)维

    for i in range(m):
        predict_in[i,] += b1

    return predict_in

"""
计算输出层输出
@param  hidden_in: 输出层输入
"""
def predict_out(predict_in):
    predict_out = sigmoid(predict_in)   # (400×2)维
    return predict_out

"""
@param      feature
@param      w0
@param      w1
@param      b0
@param      b1
@return     输出预测值 
"""
def predict(feature, w0, w1, b0, b1):
    return predict_out(predict_in(hidden_out(hidden_in(feature, w0, b0)), w1, b1))

"""
损失函数值
@param      err:预测值与标签值之间的差
"""
def cost(err):
    m, n = np.shape(err)

    cost_sum = 0
    for i in range(m):
        for j in range(n):
            cost_sum += err[i,j] ** 2
    return cost_sum / m


"""
@param  n_hidden: 隐含层节点数目   20
@param  n_output: 输出层节点数目   2
@return
"""
def bp_train(feature, label, n_hidden, maxCycle, alpha, n_output):
    m, n = np.shape(feature)  # (400×2)

    # 初始化

    fan_in = n;fan_out = n_hidden  # fan_in为i-1层节点数，fan_out为第i层节点数
    interval_low = -4.0 * sqrt(6) / sqrt(fan_in + fan_out)
    interval_high = 4.0 * sqrt(6) / sqrt(fan_in + fan_out)  # 区间上下界
    w0 = np.mat(np.random.uniform(low=interval_low, high=interval_high, size=(n, n_hidden)))
    b0 = np.mat(np.random.uniform(low=interval_low, high=interval_high, size=(1, n_hidden)))

    fan_in = n_hidden;fan_out = n_output  # fan_in为i-1层节点数，fan_out为第i层节点数
    interval_low = -4.0 * sqrt(6) / sqrt(fan_in + fan_out)
    interval_high = 4.0 * sqrt(6) / sqrt(fan_in + fan_out)  # 区间上下界
    w1 = np.mat(np.random.uniform(low=interval_low, high=interval_high, size=(n_hidden, n_output)))
    b1 = np.mat(np.random.uniform(low=interval_low, high=interval_high, size=(1, n_output)))
    '''
    w0 = np.mat(np.random.rand(n, n_hidden))
    w0 = w0 * (8.0 * sqrt(6) / sqrt(n + n_hidden)) - \
         np.mat(np.ones((n, n_hidden))) * \
         (4.0 * sqrt(6) / sqrt(n + n_hidden))
    b0 = np.mat(np.random.rand(1, n_hidden))
    b0 = b0 * (8.0 * sqrt(6) / sqrt(n + n_hidden)) - \
         np.mat(np.ones((1, n_hidden))) * \
         (4.0 * sqrt(6) / sqrt(n + n_hidden))
    w1 = np.mat(np.random.rand(n_hidden, n_output))
    w1 = w1 * (8.0 * sqrt(6) / sqrt(n_hidden + n_output)) - \
         np.mat(np.ones((n_hidden, n_output))) * \
         (4.0 * sqrt(6) / sqrt(n_hidden + n_output))
    b1 = np.mat(np.random.rand(1, n_output))
    b1 = b1 * (8.0 * sqrt(6) / sqrt(n_hidden + n_output)) - \
         np.mat(np.ones((1, n_output))) * \
         (4.0 * sqrt(6) / sqrt(n_hidden + n_output))
    '''

    # 训练
    i = 0
    while i <= maxCycle:
        # 正向传播
        ## 计算隐含层的输入
        hidden_input = hidden_in(feature, w0, b0)
        ## 计算隐含层的输出
        hidden_output = hidden_out(hidden_input)
        ## 计算输出层的输入
        output_input = predict_in(hidden_output, w1, b1)
        ## 计算输出层的输出
        output_output = predict_out(output_input)

        # 反向传播
        ## 隐含层到输出层之间的残差
        delta_output = -np.multiply((label - output_output), partial_sigmoid(
            output_input))  # (400×2)维。输出层n_l上第i个神经元的残差：δ^{nl}_i = -(y_i-a^{nl}_i)*(∂a^{nl}_i/∂z^{nl}_i)

        ## 输入层到隐含层之间的残差
        delta_hidden = np.multiply(delta_output * w1.T, partial_sigmoid(
            hidden_input))  # (400×20)维。非输出层l上第i个神经元的残差：δ^{l}_i = (∑(j=1→S_(l+1))δ^{l+1}_j * w^{l}_(ij))*(∂a^{l}_i/∂z^{l}_i)

        # 修正权重和偏置
        w1 -= alpha * (
                    hidden_output.T * delta_output)  # “hidden_output.T * delta_output”就是∑(i=1→m)∂J(W,b;X^i,y^i)/∂W^l_(ij)。更新完后的w1中的W^{2}_(ij)其实是400个样本的W^{2}_(ij)的累加和
        b1 -= alpha * np.sum(delta_output,
                             axis=0) / m  # “np.sum(delta_output, axis=0)”就是∑(i=1→m)∂J(W,b;X^i,y^i)/∂b^l_(i)。
        w0 -= alpha * (feature.T * delta_hidden)
        b0 -= alpha * np.sum(delta_hidden, axis=0) / m

        # 损失函数值
        if i % 100 == 0:
            print(f'iter:{i},cost:{(1 / 2) * cost(predict(feature, w0, w1, b0, b1) - label)}')
        i += 1

    return w0, w1, b0, b1

"""
@param      label: 训练样本标签
@param      pre: 训练样本的预测值
"""
def err_rate(label, pre):
    m = np.shape(label)[0]
    err = 0

    for i in range(m):
        if label[i,0] != pre[i,0]:
            err += 1
    rate = err / m

    return rate

"""
测试数据打上预测结果标签，并保存到文件“test_data.txt”
"""
def labeling_testData(test_data, test_labels):
    labeled0_testData = [];labeled1_testData = []
    m, n = np.shape(test_data)

    # 打标签
    for i in range(m):
        tmp = []
        for j in range(n):
            tmp.append(test_data[i,j])
        if test_labels[i,1] > 0.5:
            tmp.append(1)
            labeled1_testData.append(tmp)
        else:
            tmp.append(0)
            labeled0_testData.append(tmp)
    # ordered_testData = labeled_testData[np.argsort(labeled_testData[:, 2])]  # 对打好标签的测试数据按标签升序排序
    k = len(labeled0_testData)  # 统计测试数据中被打0标签的样本点数目
    ordered_testData = np.mat(labeled0_testData + labeled1_testData)

    # 写文件
    f = open('./../data/test_data.txt', 'w')
    m1, n1 = np.shape(ordered_testData)
    for i in range(m1):
        tmp = []
        for j in range(n1):
            tmp.append(str(ordered_testData[i, j]))
        f.write("\t".join(tmp) + "\n")
    f.close()
    return ordered_testData,k


def plot(train_data, test_data, k):
    # 如果是matrix，转换为array
    train_data = train_data.getA()
    test_data = test_data.getA()

    # 创建绘图窗口
    fig = plt.figure()
    # 将绘图窗口分成1行1列，选择第一块区域作子图
    ax1 = fig.add_subplot(1, 1, 1)
    # 设置二维坐标
    ax1.set_xlabel('x1')
    ax1.set_ylabel('x2')
    # 设置坐标轴范围
    plt.xticks(np.arange(-5.5, 5.6, 0.5))
    plt.yticks(np.arange(-5.5, 5.6, 0.5))
    # 设置图像标题
    ax1.set_title('BP神经网络分离超平面',fontproperties="SimHei")

    # 绘制打完标签后的测试数据的散点图
    test_label0 = ax1.scatter(test_data[:k][:,0],test_data[:k][:,1],marker='.',c='red')
    test_label1 = ax1.scatter(test_data[k:][:,0],test_data[k:][:,1],marker='.',c='black')
    # plt.legend(handles=[test_label0, test_label1], labels=['0', '1'], loc='lower center')

    # 绘制训练数据的散点图
    train_label0 = ax1.scatter(train_data[:200][:,0], train_data[:200][:,1], marker='o', c='c')
    train_label1 = ax1.scatter(train_data[200:][:,0], train_data[200:][:,1], marker='o', c='y')
    # 训练数据图例说明
    plt.legend(handles=[train_label0, train_label1], labels=['0', '1'], loc='lower right')

    # 添加注释
    ax1.text(-4.5, 5.0, '(红色点是测试数据中的0样本，黑色点是1样本)',fontproperties="SimSun")
    # 保存显示
    plt.savefig('./../resources/BP神经网络分离超平面.png')
    plt.show()

if __name__ == '__main__':
    s_time = time.time()
    # 加载训练数据
    feature, label, n_class = load_data('./../data/train_data.txt')
    # 导入测试数据
    testData = generate_testData()
    # 训练BP神经网络模型
    w0, w1, b0, b1 = bp_train(feature, label, 20, 1000, 0.1, n_class)
    # 得到最终的预测结果
    # res = predict(feature, w0, w1, b0, b1)
    # print(f'训练准确性：{1-err_rate(np.argmax(label,axis=1),np.argmax(res,axis=1))}')
    res = predict(testData, w0, w1, b0, b1)
    # 给测试数据打上预测结果标签
    labeled_testData,k = labeling_testData(testData, res)

    # 绘图
    plot(feature,labeled_testData,k)
    print(f'cost time:{time.time()-s_time} sec')