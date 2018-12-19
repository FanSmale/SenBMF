from numpy import *
import random
import json

def load_rating_data(file_path):
    usersname, itemsname, userscount, itemscount = [], [], [], []
    num_words = 0
    num_ratings = 0
    u, i = 1, 1
    """
    load movie lens 100k ratings from original rating file.
    need to download and put rating data in /data folder first.
    Source: http://www.grouplens.org/
    """
    prefer = []
    for line in open(file_path, 'r'):  # 打开指定文件
        tokens = json.loads(line)
        uid = str(tokens['reviewerID'])
        mid = tokens['asin']
        rat = float(tokens['overall'])
        sen = float(tokens['reviewText'])
        wei = float(tokens['helpful'])
        # 将字符串id转换为数字型
        if uid not in usersname:
            usersname.append(uid)
            userscount.append(u)
            uid = u
            u += 1
        else:
            for k in range(len(usersname)):
                if usersname[k] == uid:
                    uid = userscount[k]
        if mid not in itemsname:
            itemsname.append(mid)
            itemscount.append(i)
            mid = i
            i += 1
        else:
            for k in range(len(itemsname)):
                if itemsname[k] == mid:
                    mid = itemscount[k]
        prefer.append([uid, mid, rat,sen,wei])
        num_ratings += 1
    data = array(prefer)
    print(num_words, num_ratings)
    return data


def spilt_rating_dat(data, size=0.9):
    train_data = []
    test_data = []
    count = 0
    for line in data:
        rand = random.random()
        # if count < 20000:
        if rand < size:
            test_data.append(line)
        else:
            train_data.append(line)
        count += 1
    test_data = array(test_data)
    train_data = array(train_data)
    return train_data, test_data
