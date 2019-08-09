# -*- coding: utf-8 -*-
# @Time     : 2018/10/15 09:23
# @Author   : gaol
# @Project  : RecommendationSystem
# @File     : UserCF.py


import math
from operator import itemgetter
# from numba import *
import concurrent.futures

# 低效的用户之间兴趣相似度计算方式
'''
def UserSimilarity(train):
    W = dict()

    for u in train.keys():
        for v in train.keys():
            if u == v:
                continue
            # W[u][v] = len(train[u] & train[v])
            W[u][v] = len(train[u] & train[v]) / math.sqrt(len(train[u]) * len(train[v]))
    return W
'''

'''
计算用户兴趣相似度矩阵W
'''
# @njit
def UserSimilarity(train):
    # 建立物品-用户的倒排表
    item_users = dict()

    for u, items in train.items():
        for i in items.keys():
            if i not in item_users:
                item_users[i] = set()
            item_users[i].add(u)

    # item_users = {'a': {'A', 'B'}, 'b': {'A', 'C'}, 'd': {'D', 'A'}, 'c': {'D', 'B'}, 'e': {'D', 'C'}}
    # item_users.items() = [('a', {'A', 'B'}), ('b', {'A', 'C'}), ('d', {'A', 'D'}), ('c', {'B', 'D'}), ('e', {'D', 'C'})]

    # 计算|N(u)∩N(v)|=C[u][v]
    C = dict()
    N = dict()
    for i, users in item_users.items():
        for u in users:
            if u not in N:
                N[u] = 0
            N[u] += 1
            for v in users:
                if u == v:
                    continue
                if u not in C:
                    C[u] = dict()
                if v not in C[u]:
                    C[u][v] = 0
                # C[u][v] += 1
                C[u][v] += 1 / math.log(1 + len(users))

    # N = {'A': 3, 'B': 2, 'C': 2, 'D': 3}
    # C = {'A': {'B': 1, 'C': 1, 'D': 1}, 'B': {'A': 1, 'D': 1}, 'C': {'A': 1, 'D': 1}, 'D': {'A': 1, 'B': 1, 'C': 1}}

    # C.items() = [('A', {'B': 1, 'C': 1, 'D': 1}), ('B', {'A': 1, 'D': 1}), ('C', {'A': 1, 'D': 1}), ('D', {'A': 1, 'B': 1, 'C': 1})]
    # 计算用户之间的余弦兴趣相似度|N(u)∩N(v)|/√|N(u)||N(v)|，返回兴趣相似度矩阵W
    W = dict()
    for u, related_users in C.items():
        if u not in W:
            W[u] = dict()
        # 例如：related_users.items() = [('B', 1), ('C', 1), ('D', 1)]
        for v, cuv in related_users.items():
            if v not in W[u]:
                W[u][v] = cuv / math.sqrt(N[u] * N[v])

    # W = {'B': {'A': 0.4082482904638631, 'D': 0.4082482904638631}, 'A': {'B': 0.4082482904638631, 'C': 0.4082482904638631, 'D': 0.3333333333333333}, 'C': {'A': 0.4082482904638631, 'D': 0.4082482904638631}, 'D': {'A': 0.3333333333333333, 'B': 0.4082482904638631, 'C': 0.4082482904638631}}
    return W

# @njit
# def UserCFRecommend(user, train, W, K):
def UserCF_IIFRecommend(user, train, W, K):  # W为用户相似度矩阵，K为和user兴趣相似的K个用户
    rank = dict()   # 推荐物品列表
    # interacted_items = {'a': 1, 'b': 1, 'd': 1}
    interacted_items = train[user]

    # W[user].items() = [('B', 0.4082482904638631), ('C', 0.4082482904638631), ('D', 0.3333333333333333)]
    # sorted(W[user].items(), key=itemgetter(1), reverse=True) = [('B', 0.4082482904638631), ('C', 0.4082482904638631), ('D', 0.3333333333333333)]
    # 先找到和用户user兴趣相似度最高的K个用户
    for v, wuv in sorted(W[user].items(), key=itemgetter(1), reverse=True)[0:K]:  # key()函数是作用于“W[user].items()”列表中的每个tuple元素的第二个元素的
        # train[v].items() = [('a', 1), ('c', 1)]、[('b', 1), ('e', 1)]、[('c', 1), ('d', 1), ('e', 1)]
        # 将这K个用户有过行为但用户user没有过行为的物品加入到user的推荐列表中
        for i, rvi in train[v].items():
            if i not in interacted_items:
                if i not in rank:
                    rank[i] = 0
                rank[i] += wuv * rvi
            # rank[i] = round(rank[i], 6)
    # 这个返回的rank是和user用户兴趣相似的K个用户有过行为但user没有过行为的，推荐给user的所有物品列表
    # print(rank)   # {'a': 0.37160360818355515, 'd': 0.7432072163671103, 'c': 0.37160360818355515}
    return rank


def main():
    train = {'A':{'a':1,'b':1,'d':1}, 'B':{'a':1,'c':1}, 'C':{'b':1,'e':1}, 'D':{'c':1,'d':1,'e':1}}
    # train = {'A': {'a': 1, 'b': 1, 'd': 1, 'f':1}, 'B': {'a': 1, 'c': 1}, 'C': {'a':1, 'b': 1, 'e': 1, 'f':1}, 'D': {'b': 1, 'c': 1, 'd': 1, 'e': 1}, 'E':{'g':1}}
    W = UserSimilarity(train)
    print(W)
    print(UserCF_IIFRecommend('A', train, W, 3))    # {'a': 0.37160360818355515, 'd': 0.7432072163671103, 'c': 0.37160360818355515}


if __name__ == '__main__':
    main()