import json
from sklearn import preprocessing
from numpy import *

file  = "data/Patio.json"

# clean the data
def clean_doc(file_path):
    pre = []
    for line in open(file_path, 'r'):
        tokens = json.loads(line)
        text = tokens['reviewText']
        # filter out the non-alpha character
        text = [word for word in text.rstrip().split(' ')if word.isalpha()]
        # filter character with length <= 1
        text = [word for word in text if len(word) > 1]
        pre.append(text)
    return pre

# read the dictionary
neg1 = []
pos1 = []
neg2 = []
pos2 = []
Pos = []
Neg = []
much = []
more = []
little = []
deny = []
infile = open('data/pos.csv', 'r')
for line in infile:
    data = line.rstrip().split(',')
    pos2.append(data[0])

infile = open('data/neg.csv', 'r')
for line in infile:
    data = line.rstrip().split(',')
    neg2.append(data[0])

infile = open('data/extreme dict.csv', 'r')
for line in infile:
    data = line.rstrip().split(',')
    much.append(data[0])

infile = open('data/more.csv', 'r')
for line in infile:
    data = line.rstrip().split(',')
    more.append(data[0])

infile = open('data/sih.csv', 'r')
for line in infile:
    data = line.rstrip().split(',')
    little.append(data[0])

infile = open('data/not.csv', 'r')
for line in infile:
    data = line.rstrip().split(',')
    deny.append(data[0])
Pos = pos1 + pos2
Neg = neg1 + neg2


# calculate the sentiment score
def GetScore(list):
    pos_s = 0  # 积极词的第一次分值
    poscount2 = 0  # 积极词反转后的分值
    poscount3 = 0  # 积极词的最后分值（包括叹号的分值）
    neg_s = 0  # 积极词的第一次分值
    negcount2 = 0  # 积极词反转后的分值
    negcount3 = 0  # 积极词的最后分值（包括叹号的分值）
    n = -1  #记录当前情感词的位置
    for w in list:
        n += 1
        if (w in Neg) == True:
            c = 0
            if (w in neg2) == True:
                neg_s = neg_s + 2
            elif (w in neg1) == True:
                neg_s = neg_s + 1
            for m in list[n-2:n]: #对于每一个情感词，遍历从当前情感词到前两个词
                if m in much:
                    neg_s *= 2.0
                elif m in more:
                    neg_s *= 1.5
                elif m in little:
                    neg_s *= 0.5
            for m in list[n-1:n]:
                if m in deny :
                     c += 1
            if  c > 0:
                #  扫描情感词前的否定词数
                # 如果出现反转，该词不加入消极分数，而加入积极分数
                poscount2 += neg_s
                neg_s = 0
            else:
                negcount2 += neg_s
                neg_s = 0

        elif (w in Pos) == True:
            d = 0
            if (w in pos2) == True:
                pos_s = pos_s + 2
            elif (w in pos1) == True:
                    pos_s = pos_s + 1
            for m in list[n-2:n]:
                if m in much:
                    pos_s *= 2.0
                elif m in more:
                    pos_s *= 1.5
                elif m in little:
                    pos_s *= 0.5
            for m in list[n - 1:n]:
                if m in deny:
                    d += 1
            if d > 0:
                # print('反转')
                # 扫描整句话的否定词数
                # 如果出现反转，该词不加入积极分数，而加入负面分数
                # poscount2 += pos_s
                negcount2 += pos_s
                # print('总体反转后分值', poscount2)
                pos_s = 0
            else:
                poscount2 += pos_s
                pos_s = 0
            # a = n  # 当前情感词变成前一个情感词
    score1 = poscount2 - negcount2
    return score1

def normsent():
    file_out = open("data/newpatio.json",'w')
    text = clean_doc(file)
    Score = []
    count = 0
    for i in range(len(text)):
        s = GetScore(text[i])
        Score.append(s)
    data = array(Score).reshape(-1, 1)
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0,5))
    score = min_max_scaler.fit_transform(data)
    for line in open(file):
        tokens = json.loads(line)
        tokens['reviewText'] = float(score[count])
        outStr = json.dumps(tokens,ensure_ascii=False )+ '\n'
        file_out.write(outStr)
        count += 1

a = normsent()


