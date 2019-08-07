# -*- coding: utf-8 -*-
# @Time     : 2019/5/21 10:43
# @Author   : gaol
# @Project  : RecommenderSystem
# @File     : PersonalRank.py

def PersonalRank(G, alpha, root, max_step):
    rank = {x:0 for x in G.keys()}
    rank[root] = 1

    for k in range(max_step):
        tmp = {x:0 for x in G.keys()}
        # 取出节点i和它的出边尾节点集合ri
        for i, ri in G.items():
            # 取节点i的出边尾节点j以及边e(i,j)的权重wij,边的权重都为1，归一化后就是1/len(ri)
            for j, wij in ri.items():
                tmp[j] += alpha * rank[i] / len(ri)
        tmp[root] = 1 - alpha
        rank = tmp
    print(rank)
    return rank

if __name__ == '__main__':
    # G = {'A': {'a':1,'b':1, 'd':1}, 'B': {'b':1, 'c':1, 'e':1}, 'C': {'c':1, 'd':1}, 'D': {'b':1, 'c':1, 'd':1}, 'E': {'a':1, 'd':1}, 'F': {'d':1}}
    G = {'A': {'a':1,'c':1}, 'B': {'a':1,'b':1, 'c':1, 'd':1}, 'C': {'c':1, 'd':1}, 'a' : {'A' : 1, 'B' : 1}, 'b' : {'B' : 1}, 'c' : {'A' : 1, 'B' : 1, 'C':1}, 'd' : {'B' : 1, 'C' : 1}}
    PersonalRank(G, 0.8, 'A', 50)