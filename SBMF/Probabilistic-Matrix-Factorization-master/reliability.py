from user import consistency
from sklearn import preprocessing
from numpy import *
import json

def calweight():
    in_file = open("data/newpatio.json")
    a = consistency()
    itemsum = a[0]
    userconsist = a[1]
    weights = []
    for line in in_file:
        tokens = json.loads(line)
        uid = str(tokens['reviewerID'])
        mid = tokens['asin']
        rat = int(tokens['overall'])
        sen = float(tokens['reviewText'])
        help = int(tokens['helpful'][0])
        weight = (help / itemsum[mid]) / (userconsist[uid][0])
        weights.append(weight)
    data = array(weights).reshape(-1, 1)
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    score = min_max_scaler.fit_transform(data)
    return score




