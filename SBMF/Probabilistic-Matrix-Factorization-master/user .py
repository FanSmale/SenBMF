from scipy import spatial
import json

def consistency():
    file = open("data/newpatio.json")
    rating,sentiment,userconsist,itemsum = {},{},{},{}
    users,items = [],[]
    # csvFile2 = open('cred.csv', 'w', newline='')
    # writer = csv.writer(csvFile2)
    for line in file:
        tokens = json.loads(line)
        uid = str(tokens['reviewerID'])
        mid = tokens['asin']
        rat = int(tokens['overall'])
        sen = float(tokens['reviewText'])
        help = int(tokens['helpful'][0])
        # calculuate the total helpfulness of each item
        if mid not in items:
            itemsum[mid] = 1
            itemsum[mid] += help
            items.append(mid)
        else:
            itemsum[mid] += help
        # calculate the consistency of each user
        if uid not in users:
            sentiment[uid] = []
            rating[uid] = []
            sentiment[uid].append(sen)
            rating[uid].append(rat)
            users.append(uid)
        else:
            sentiment[uid].append(sen)
            rating[uid].append(rat)
    for user in rating:
        userconsist[user] = []
        z = spatial.distance.euclidean(rating[user],sentiment[user])
        userconsist[user].append(float(z))
    return itemsum,userconsist









