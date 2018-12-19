from reliability import calweight
import json

file = open("data/newpatio.json")
file_out = open("data/finalpatio.json", 'w')
weight = calweight()
count = 0
for line in file:
    tokens = json.loads(line)
    tokens['helpful'] = float(weight[count])
    outStr = json.dumps(tokens, ensure_ascii=False) + '\n'
    file_out.write(outStr)
    count += 1

