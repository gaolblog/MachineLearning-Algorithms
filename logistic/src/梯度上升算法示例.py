# 梯度上升算法示例
def gradAscentSample():
    def fDerrivative(oldX):
        return -2 * oldX + 4
    oldX = -1
    newX = 0 # 梯度上升算法的初始值，即从(0,0)开始
    alpha = 0.01 # 学习率，用于梯度控制更新的幅度
    presision = 0.00000001 # 精度

    while abs(newX - oldX) > presision:
        oldX = newX
        newX = oldX + alpha * fDerrivative(oldX)
        print(newX,oldX)

if __name__ == '__main__':
    gradAscentSample()