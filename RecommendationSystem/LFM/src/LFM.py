# -*- coding: utf-8 -*-
# @Time     : 2019/5/13 21:35
# @Author   : gaol
# @Project  : RecommenderSystem
# @File     : LFM.py

import random
import math
import numpy as np
from operator import itemgetter

# 将数据划分为训练集和测试集
def SplitData(data, M, k, seed):
    test = {};train = {}
    random.seed(seed)

    for user, item in data:
        if random.randint(0, M) == k:
            if user not in test:
                test[user] = dict()
            if item not in test[user]:
                test[user][item] = 0
            test[user][item] += 1
        else:
            if user not in train:
                train[user] = dict()  # train = {'1':{'1193':1,'661':1,'914':1,...},'2':{'1357':1,...}}
            if item not in train[user]:
                train[user][item] = 0
            train[user][item] += 1

    return train, test

# 随机采样，对每个用户而言包含较为平衡的正负样本
def RandomSelectNegativeSample(items, items_pool):
    ret = {}
    for i in items:
        ret[i] = 1

    while 0 not in ret.values():    # 如果没有采集到负样本就重新采样
        n = 0
        for i in range(0, len(items)*3):    # 范围上限设为“len(items)*3”，主要是为了保证正负样本平衡
            item = items_pool[random.randint(0, len(items_pool)-1)]
            # item = items_pool[np.random.uniform(0, len(items_pool))]

            if item in ret:
                continue
            ret[item] = 0
            n += 1
            if n > len(items):
                break

    # print(ret)
    return ret

def ImprovedRandomSelectNegativeSample(items, items_pool, ratio):
    ret = {}
    for i in items:
        ret[i] = 1

    posSize = len(ret)  # 正样本数目
    negSize = ratio * posSize  # 负样本数目

    while 0 not in ret.values():  # 如果没有采集到负样本就重新采样
        n = 0
        for i in range(0, len(items)*3):    # 范围上限设为“len(items)*3”，主要是为了保证正负样本平衡
            item = items_pool[random.randint(0, len(items_pool)-1)]

            if item in ret:
                continue
            ret[item] = 0
            n += 1
            if n == negSize:
                break

    # print(ret)
    return ret

# 采样热门物品为负样本，并保持较为平衡的正负样本
def SelectPositiveNegativeSample(items, ordereditems_pool, ratio):
    ret = {}

    for i in items.keys():
        ret[i] = 1

    posSize = len(ret)  # 正样本数目
    negSize = ratio * posSize   # 负样本数目
    n = 0
    for item, rates in ordereditems_pool:
        if item not in ret:
            ret[item] = 0
            n += 1
        if n == negSize:
            break
    # print(ret)
    return ret

# 训练集中每一个用户都有一个P向量；每个物品都有一个Q向量
def InitModel(users_pool, items_pool, F):
    P = dict()
    Q = dict()

    for u in users_pool:
        if u not in P:
            P[u] = {}
        P[u] = np.random.rand(F)

    for i in items_pool.keys():
        if i not in Q:
            Q[i] = {}
        Q[i] = np.random.rand(F)

    return [P, Q]

def sigmod(x):
    '''''
    单位阶跃函数,将兴趣度限定在[0,1]范围内
    :param x: 兴趣度
    :return: 兴趣度
    '''
    y = 1.0 / (1 + math.exp(-x))
    return y

def Predict(Puser, Qitem):
    return sigmod(np.dot(Puser, Qitem))

'''
F：隐特征个数
N：迭代次数
alpha：学习速率
lamda：正则化参数
'''
# LFM模型训练过程
'''
def LatentFactorModel(user_items, F, N, alpha, lamda):  # user_items = {'A': {'a':1,'b':1, 'd':1}, 'B': {'b':1, 'c':1, 'e':1}, 'C': {'c':1, 'd':1}, 'D': {'b':1, 'c':1, 'd':1}, 'E': {'a':1, 'd':1}, 'F': {'d':1}}
    users_pool = []
    items_pool = {}

    for user, items in user_items.items():
        users_pool.append(user)
        for item in items.keys():
            if item not in items_pool:
                items_pool[item] = 0
            items_pool[item] += 1
    # print(users_pool)   # users_pool = ['A', 'B', 'C', 'D', 'E', 'F']
    # print(items_pool)   # items_pool = {'a': 2, 'b': 3, 'd': 5, 'c': 3, 'e': 1}
    [P, Q] = InitModel(users_pool, items_pool, F)

    for step in range(0, N):    # 迭代N次取到最优参数
        for user,items in user_items.items():
            samples = RandomSelectNegativeSample(items, list(items_pool.keys()))     # 用户'A'的samples = {'a': 1, 'b': 1, 'd': 1, 'e': 0, 'c': 0}
            # err_sum = 0
            for item, rui in samples.items():
                eui = rui - Predict(P[user], Q[item])
                # err_sum += eui
                # 向量化更新P、Q矩阵
                P[user] += alpha * (eui * Q[item] - lamda * P[user])
                Q[item] += alpha * (eui * P[user] - lamda * Q[item])
            # print(f'{user}:{err_sum}')
        # 学习率调整
        alpha *= 0.9
    # print([P ,Q])
    return [P, Q]
'''
def LatentFactorModel(user_items, F, N, alpha, lamda, ratio):  # user_items = {'A': {'a':1,'b':1, 'd':1}, 'B': {'b':1, 'c':1, 'e':1}, 'C': {'c':1, 'd':1}, 'D': {'b':1, 'c':1, 'd':1}, 'E': {'a':1, 'd':1}, 'F': {'d':1}}
    users_pool = []
    items_pool = {}

    for user, items in user_items.items():
        users_pool.append(user)
        for item in items.keys():
            if item not in items_pool:
                items_pool[item] = 0
            items_pool[item] += 1
    # print(users_pool)   # users_pool = ['A', 'B', 'C', 'D', 'E', 'F']
    # print(items_pool)   # items_pool = {'a': 2, 'b': 3, 'd': 5, 'c': 3, 'e': 1}
    # ordereditems_pool = sorted(items_pool.items(), key = itemgetter(1), reverse = True)

    # 初始化P、Q矩阵
    [P, Q] = InitModel(users_pool, items_pool, F)

    for step in range(0, N):    # 迭代N次取到最优参数
        for user,items in user_items.items():
            samples = ImprovedRandomSelectNegativeSample(items, list(items_pool.keys()), ratio)     # 用户'A'的samples = {'a': 1, 'b': 1, 'd': 1, 'e': 0, 'c': 0}
            # print(f'{user}-{samples}')
            # err_sum = 0
            for item, rui in samples.items():
                eui = rui - Predict(P[user], Q[item])
                # err_sum += eui
                # 向量化更新P、Q矩阵
                P[user] += alpha * (eui * Q[item] - lamda * P[user])
                Q[item] += alpha * (eui * P[user] - lamda * Q[item])
            # print(f'{user}:{err_sum}')
        # 学习率调整
        alpha *= 0.9
    # print([P ,Q])
    return [P, Q]

# 预测
def LFMRecommend(train, user, P, Q):
    rank = {}

    puf = P[user]   # puf为向量
    for i,qfi in Q.items(): # qfi为向量
        if i not in train[user].keys():
            if i not in rank:
                rank[i] = 0
            rank[i] += sigmod((puf * qfi).sum())
    # print(rank)
    return rank

if __name__ == '__main__':
    # 数据集按均匀分布划分为M份，M-1份均为训练集，剩下1份为测试集
    M = 8
    k = 0
    seed = 42  # 随机数种子
    # data = {'A': {'a':1,'b':1, 'd':1}, 'B': {'b':1, 'c':1, 'e':1}, 'C': {'c':1, 'd':1}, 'D': {'b':1, 'c':1, 'd':1}, 'E': {'a':1, 'd':1}, 'F': {'d':1}}   # 1表示喜欢，0表示不感兴趣
    data = [('A','a'),('A','b'),('A','d'),('B','b'),('B','c'),('B','e'),('C','c'),('C','d'),('D','b'),('D','c'),('D','d'),('E','a'),('E','d'),('F','d')]
    # data = [{'A': ['a','b','d']}, {'B': ['b','c','e']}, {'C': ['c','d']}, {'D': ['b','c','d']}, {'E': ['a','d']}, {'F': ['d']}]
    train, test = SplitData(data, M, k, seed)
    print(f'train-{train}')
    print(f'test-{test}')
    # RandomSelectNegativeSample(list(data[0].values())[0])
    # InitModel(train[0], 4)
    [P, Q] = LatentFactorModel(train, 4, 1, 0.02, 0.01, 1)
    rank = LFMRecommend(train, 'A', P, Q)
    print(rank)
    # print(SelectPositiveNegativeSample({'a':1,'b':1, 'd':1}, [('d',5), ('b',3), ('c',3), ('a',2),('e',1)], 1))