import math
from operator import itemgetter

'''
计算物品相似度矩阵
'''
def ItemSimilarity(train):	# train = {'A':{'a','b','d'}, 'B':{'b','c','e'}, 'C':{'c','d'}, 'D':{'b','c','d'}, 'E':{'a','d'}}
	N = {}
	C = {}	# 共现矩阵C
	W = {}	# 物品相似度矩阵W

	# 计算两两物品的共现矩阵C
	for u, items in train.items():
		for i in items:
			if i not in N:
				N[i] = 0
			N[i] += 1

			for j in items:
				if i == j:
					continue
				if i not in C:
					C[i] = {}
				if j not in C[i]:
					C[i][j] = 0
				# C[i][j] += 1
				C[i][j] += 1 / math.log(1 + len(items))
	'''
	N = {'a': 2, 'b': 3, 'd': 4, 'c': 3, 'e': 1}
	C = {'a': {'b': 1, 'd': 2}, 'b': {'a': 1, 'd': 2, 'c': 2, 'e': 1}, 'd': {'a': 2, 'b': 2, 'c': 2}, 'c': {'b': 2, 'e': 1, 'd': 2}, 'e': {'b': 1, 'c': 1}}
	'''
	
	for i, related_items in C.items():
		for j, cij in related_items.items():
			if i not in W:
				W[i] = {}
			if j not in W[i]:
				W[i][j] = 0
			W[i][j] = cij / math.sqrt(N[i] * N[j])

	# 返回物品相似度矩阵
	# W = {'b': {'a': 0.4082482904638631, 'd': 0.5773502691896258, 'c': 0.6666666666666666, 'e': 0.5773502691896258}, 'a': {'b': 0.4082482904638631, 'd': 0.7071067811865475}, 'd': {'b': 0.5773502691896258, 'a': 0.7071067811865475, 'c': 0.5773502691896258}, 'c': {'b': 0.6666666666666666, 'e': 0.5773502691896258, 'd': 0.5773502691896258}, 'e': {'b': 0.5773502691896258, 'c': 0.5773502691896258}}

	# 将ItemCF相似度矩阵按最大值归一化
	for i, j_wij in W.items():
		max_wij = max(j_wij.values())
		for j, wij in j_wij.items():
			W[i][j] = W[i][j] / max_wij

	# W = {'b': {'d': 0.8660254037844387, 'a': 0.6123724356957947, 'c': 1.0, 'e': 0.8660254037844387}, 'd': {'b': 0.721969316263228, 'a': 1.0, 'c': 0.8164965809277261}, 'a': {'b': 0.5105093993383438, 'd': 1.0}, 'c': {'b': 1.0, 'e': 0.8660254037844387, 'd': 0.9794138964885573}, 'e': {'b': 1.0, 'c': 1.0}}
	return W

# def ItemCFRecommend(train, user_id, W, K):	# K为和物品和i相似的K个物品数
def ItemCF_IUFNormRecommend(train, user_id, W, K):
	rank = {}

	# pi为用户u对自己的历史行为列表中物品的兴趣度（对于隐反馈数据集，可设置为1）
	pi = 1
	# ru为用户u的历史行为物品列表
	ru = train[user_id]

	for i in ru:
		for j, wj in sorted(W[i].items(), key=itemgetter(1), reverse=True)[0:K]:	# wj为物品相似度
			if j in ru:
				continue
			if j not in rank:
				rank[j] = 0
			rank[j] += pi * wj

			# 带解释的ItemCF
			'''
			rank[j].weight += pi * wj
			rank[j].reason[i] = pi * wj		# 推荐j是因为用户历史行为列表中有i，对j的兴趣度是pi*wj
			'''
	return rank  # rank = {'c': 1.2441, 'e': 0.5774}

def main():
	train = {'A':{'a','b','d'}, 'B':{'b','c','e'}, 'C':{'c','d'}, 'D':{'b','c','d'}, 'E':{'a','d'}}

	W = ItemSimilarity(train)
	
	rank = ItemCF_IUFNormRecommend(train, 'A', W, 3)
	print(rank)

if __name__ == '__main__':
	main()