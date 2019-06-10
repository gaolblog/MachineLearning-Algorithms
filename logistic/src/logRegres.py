from numpy import *

def loadDataSet():
    dataMat = [];labelMat = []
    fr = open('G:/master/python/PycharmProjects/ml/logistic/data/testSet.txt','r')
    for line in fr:
        lineArr = line.strip().split() # lineArr为一个字符串列表
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat,labelMat

def sigmoid(z):
    return 1.0/(1+exp(-z))

def gradAscent(dataMatIn,classLabels):
    dataMatrix = mat(dataMatIn) # dataMatrix为一个100*3的二维矩阵
    labelMat = mat(classLabels).transpose() # labelMat为一个100*1的列向量
    m,n = shape(dataMatrix) # [X0,X1,X2]矩阵的大小100*3

    # 设置梯度上升算法所需的参数
    alpha = 0.001   # 步长
    maxCycles = 500 # 最大迭代次数
    weights = ones((n,1)) # 生成3*1的单位向量

    # 矩阵运算
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights) # (100*3) * (3*1)
        error = labelMat - h # 真实类别与预测类别之间的差值
        weights = weights + alpha * (dataMatrix.transpose() * error) # 按以上步骤得到的差值方向调整回归系数
    return weights

def plotDataSet():
    import matplotlib.pyplot as plt
    dataMat,labelMat = loadDataSet()
    dataArr = array(dataMat)
    n = shape(dataArr)[0]
    xcord1 = [];ycord1 = [] # 正样本
    xcord2 = [];ycord2 = [] # 负样本

    # 根据数据集标签进行分类
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i,1]); ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1]); ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    plt.title('DataSet')
    plt.xlabel('X1')
    plt.ylabel("X2")
    plt.show()

def plotBestFit(weights):
    import matplotlib.pyplot as plt
    dataMat,labelMat = loadDataSet()
    dataArr = array(dataMat)
    n = shape(dataArr)[0]
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []

    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i,1]); ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1]); ycord2.append(dataArr[i,2])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-4.0, 4.0, 0.1)
    y = (-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x,y)
    plt.xlabel('X1'); plt.ylabel("X2")
    plt.show()

if __name__ == '__main__':
    # 准备数据
    dataMat,labelMat = loadDataSet()
    dataArr = array(dataMat)

    # 通过梯度上升法求解参数
    weights = gradAscent(dataArr,labelMat)
    print(weights)
    # plotBestFit(weights.getA())
    # plotDataSet()