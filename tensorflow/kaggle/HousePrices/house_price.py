import pandas as pd

train = pd.read_csv('train.csv')
ret = train.isnull().sum()
tmp = []
for x in ret.keys():
    if ret[x]>0:
        tmp.append(x)

print(tmp)
