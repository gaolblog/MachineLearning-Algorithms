'''
Created on Oct 18, 2017

@author: gaol
'''

from numpy import *

import sys
sys.path.append("G:\python\naiveBayes")

# 读取短信csv文件，创建短信文本列表和字符串标签列表
import csv
with open("G:/python/naiveBayes/smsSpam.csv",encoding='utf-8',errors='ignore') as csvfile:
	smsReader = csv.reader(csvfile)
	classSms = [row[0] for row in smsReader] # classSms为短信字符串标签列表（ham/spam）
##测试语句：print(classSms)

with open("G:/python/naiveBayes/smsSpam.csv",encoding='utf-8',errors='ignore') as csvfile:
	smsReader = csv.reader(csvfile)
	smsList = [row[1] for row in smsReader] # smsList为短信文本列表
##测试语句：print(smsList)

# 将短信字符串spam/ham列表数量化为短信0/1标签向量
def class2Vet(classSms):
	classVec = []
	for i in list(range(len(classSms))):
		if classSms[i] == 'spam':
			classVec.append(1)
		else:
			classVec.append(0)
	return classVec
##测试语句：print(classVec)

# 统计spam/ham短信数量
print("spam=%d,ham=%d" %(sum(class2Vet(classSms)),(len(classSms)-sum(class2Vet(classSms)))))

# 短信文本解析
def textParse(documentString):	# 输入参数为一条短信文本列表
    import re
    listOfTokens = re.split(r'\W*'r'\d*', documentString) # 返回列表,r表示原生字符串
    													  # 去除非字母字符及数字，去除长度<2的词汇，小写化
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]

# 构建初步的语料库
def createCorpus(documentSet):
	wordList = [];docList = [];corpusList = set()
	for i in list(range(len(documentSet))):
		wordList = textParse(documentSet[i]) # wordList为去除非字母字符及数字之后，并解析原短信文本为词汇集的列表
		docList.append(wordList) # docList为每条短信文本生成的wordList构成的列表集
		# corpusList.extend(docList)
	#return corpusList
	for documentWord in docList: 
		corpusList = corpusList | set(documentWord) # 将docList中的每条短信文本的词汇列表求并集，去除重复词汇
	return list(corpusList) # 将创建的初步语料库存入列表
# print(createCorpus(smsList)) # 打印初步创建的语料库
print("length(smsOriginCorpus)=%d" % len(createCorpus(smsList))) # 初步语料库的词汇长度为7432

# 构建停用词列表
def createStopWords():
	import csv
	with open("G:/python/naiveBayes/EnStopWords.csv",encoding='utf-8',errors='ignore') as stopWordsFile:
		stopWorsReader = csv.reader(stopWordsFile)
		stopWordsList = [row for row in stopWordsFile]
	for i in list(range(len(stopWordsList))):
		stopWordsList[i] = stopWordsList[i].strip()	# 去除停用词最后的换行符
	return stopWordsList
##测试打印停用词列表：print(stopWordsList)
##测试打印停用词列表长度：print("length(stopWordsList)=%d" %len(stopWordsList))

# 词形还原 
def lemmatizeCorpus(smsUpdateCorpus):
	import nltk
	from nltk.stem import WordNetLemmatizer # 词形还原函数
	lemmatizer=WordNetLemmatizer()

	lemmatizeCorpus = []
	for corpusWord in smsUpdateCorpus:
		lemmatizeCorpus.append(lemmatizer.lemmatize(corpusWord))
	return lemmatizeCorpus

# 语料库清理
def cleanCorpus(smsOriginCorpus): # 函数参数为初步构建的语料库列表
	myStopWordsList = createStopWords()
	smsUpdateCorpus = [] # 更新后的语料库
	smsFinalCorpus = [] # 清理后的最终语料库
	
	# 在删除初步语料库中的停用词时出现bug，此处代码主要用于测试初步语料库中的词汇确实是不重复的
	''' 
	count = 0
	smsOriginCorpusSet = set(smsOriginCorpus)
	for item in smsOriginCorpusSet:
		count += smsOriginCorpus.count(item)
	print(count)
	'''
	# 用此for循环删除初步语料库中的停用词时，出现bug：不同次运行“smsNaiveBayes.py”时统计得到的新语料库的长度不一致
	'''
	for corpusWord in smsOriginCorpus:
		if corpusWord in myStopWordsList:
			smsOriginCorpus.remove(corpusWord)
	smsUpdateCorpus = smsOriginCorpus
	'''
	smsUpdateCorpus = [item for item in smsOriginCorpus if item not in myStopWordsList] # 去除初步语料库中的停用词，
																						# 并将剩余词汇更新至新的语料库列表
	smsFinalCorpus = lemmatizeCorpus(smsUpdateCorpus)
	return smsFinalCorpus
# print(cleanCorpus(createCorpus(smsList))) # 打印去除停用词之后的语料库。和初步的语料库相比，去除了559个停用词
print("length(smsUpdateCorpus)=%d" % len(cleanCorpus(createCorpus(smsList))))

# 将FinalCorpus写入到csv文件以改进停用词列表
def FinalCorpus2Csv(finalCorpus):
	import csv
	with open("G:/python/naiveBayes/FinalCorpus2Csv.csv",'w') as stopWordsFile:
		stopWorsWriter = csv.writer(stopWordsFile)
		for word in finalCorpus:
			stopWorsWriter.writerow(word)
	#stopWordsFile.close()
#FinalCorpus2Csv(cleanCorpus(createCorpus(smsList)))

# 清理原始短信文本集，返回词汇集列表
def cleanSmsText(documentSet):
	wordList = [];docListSet = [];updateDocList = []
	myStopWordsList = createStopWords() # 生成停用词列表
	
	# 此for循环用于将短信文本集转成短信词汇集
	for i in list(range(len(documentSet))):
		wordList = textParse(documentSet[i]) # wordList为去除非字母字符及数字之后，并解析原短信文本为词汇集的列表
		docListSet.append(wordList) # docList为每条短信文本生成的wordList构成的列表集
	# print(docListSet)
	# 此for循环用于生成一个包含5559个空子列表的列表
	for i in range(len(docListSet)):
		updateDocList.append([]) # updateDocList = [[],[],[],...[]](其中的子列表共5559个)
	# 此for循环用于按照停用词列表更新上一步生成的空updateDocList
	for docList in range(len(docListSet)):
		for word in docListSet[docList]:
			if word not in myStopWordsList:
				updateDocList[docList].append(word)
	#测试代码：此for循环用于打印短信文本词汇集列表集中为空的列表序号：607、616、699、843、1426、1872、1897、1927、2085
	# 2109、2271、2348、2567、2746、2815、3821、4008、4280、4328、4569、4584、4724、5030、5142、5335、5429、5535、5559
	'''for i in list(range(len(updateDocList))):
		if len(docListSet[i]) == 0:
			print(i)'''
	return updateDocList
# print("\n")
print("length(cleanSmsText(smsList))=%d" %len(cleanSmsText(smsList))) # 打印清理后的短信文本的词集列表的长度
# print(cleanSmsText(smsList))

# 创建训练集和测试集
trainingSet = cleanSmsText(smsList)[0:4169] # 将短信数据集的75%作为训练集
# print(trainingSet)
print("length(trainingSet)=%d" %len(trainingSet))
testingSet = cleanSmsText(smsList)[4169:5560] # 将短信数据集的25%作为测试集
# print(testingSet)
print("length(testingSet)=%d" %len(testingSet))

# naive bayes词袋模型(朴素贝叶斯的多项式模型)
def bagOfWords2VecMN(finalCorpus,inputSet):
    returnVec = [0]*len(finalCorpus)
    
    for word in inputSet:
        if word in finalCorpus:
            returnVec[finalCorpus.index(word)] += 1
        else: 
        	pass
        	# print ("the word: %s is not in my Corpus!" % word)
    return returnVec

# 训练朴素贝叶斯分类模型
# 计算垃圾短信的先验概率pAb，垃圾短信p1Vect/非垃圾短信p0Vect的似然概率向量
def trainNB(trainMatrix,trainCategory): # 参数为trainMat矩阵以及每篇文档的类别标签构成的向量
    numTrainDocs = len(trainMatrix) #numTrainDocs=5559
    numWords = len(trainMatrix[0]) #numWords=32
    pAbusive = sum(trainCategory)/float(numTrainDocs) # 垃圾短信的先验概率为0.13528424082513793
    
    #p0Num = zeros(numWords); p1Num = zeros(numWords)    # 构造维数为6873的全0向量 
    p0Num = ones(numWords); p1Num = ones(numWords)    # 构造维数为6873的全1向量，即假定每个特征词至少出现一次 
    # p0Denom = 2.0; p1Denom = 2.0  
    p0Denom = 6873.0; p1Denom = 6873.0                                              
    
    for i in range(numTrainDocs):
        if trainCategory[i] == 1: # 在垃圾短信1标签下统计
            p1Num += trainMatrix[i] # 计算在标签1下所有特征词各自出现的频数
            p1Denom += sum(trainMatrix[i]) # 计算在标签1下所有特征词出现的总频数
        else:					  # 在非垃圾短信0标签下统计
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    
    # p1Vect = p1Num/p1Denom # 计算p(W|1),W表示特征向量
    # p0Vect = p0Num/p0Denom # 计算p(W|0)
    p1Vect = log(p1Num/p1Denom) # 取对数防止6873个接近0的小概率值相乘结果为0         
    p0Vect = log(p0Num/p0Denom)          
    # return p0Num,p0Denom,p0Vect,p1Vect,pAbusive
    return p0Vect,p1Vect,pAbusive # 返回模型参数
    # return pAbusive

def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)    
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    # print(p1,p0)
    if p1 > p0:
        return 1
    else: 
        return 0
# # # # # # # # # # # # # # # # # # # # # # # # # main # # # # # # # # # # # # # # # # # # # # # # # # #
# 用词向量填充trainMat矩阵
myFinalCorpus = cleanCorpus(createCorpus(smsList)) # 存储最终的语料库

trainMat = []
for postinDoc in cleanSmsText(smsList): 
	trainMat.append(bagOfWords2VecMN(myFinalCorpus,postinDoc)) # 循环结束后的trainMat可看成为5559*6873的矩阵
##测试语句：print(trainMat[0])
print("len(trainMat)=%d" %len(trainMat))
print("len(trainMat[0])=%d" %len(trainMat[0]))

# 计算训练集中垃圾短信的先验概率pAb，垃圾短信p1V/非垃圾短信p0V的似然概率向量
p0V,p1V,pAb = trainNB(trainMat[0:4169],class2Vet(classSms[0:4169]))
print("\n")
print("pAb=%f" %pAb)
print("p0V=")
print(p0V)
print("length(p0V)=%d" %len(p0V))
print("p1V=")
print(p1V)
print("length(p1V)=%d" %len(p1V))

# 朴素贝叶斯模型测试
testSmsIndex = [] # 元素为4169~5559
for indexNum in list(range(4169,5559,1)):
	testSmsIndex.append(indexNum)
##测试语句：print(testSmsIndex)
# print(trainMat[5559])
testSpamIndex = [] # 0/1稀疏矩阵
for index in testSmsIndex:
	# testEntry = trainMat[index] # 测试数据集中的一条短信
	testSpamIndex.append(classifyNB(trainMat[index],p0V,p1V,pAb))
print(testSpamIndex)

# 统计测试数据集中的垃圾短信数目
spamCount = 0
for result in testSpamIndex:	
	if result == 1:
		spamCount += result
print("spam=%d,ham=%d" %(spamCount,len(testingSet)-spamCount))


