import pandas as pd
import numpy as np


#train = pd.read_csv('train.csv')
#ret = train.isnull().sum()
#tmp = []
#for x in train.keys():
#    if len(set(train[x].values))<10:
#        print(x,':',set(train[x].values))
#
def data_preprocess(data):
    train = pd.read_csv(data)
    corrmat = train.corr()
    train = train[corrmat.index[abs(corrmat["SalePrice"])>=0.3]] # selest values that of corr are over 0.3
    train = train.drop(['LotFrontage'],axis=1)
    train = train.drop(['GarageYrBlt'],axis=1)

    #mszoning_map = {x:idx for idx,x in enumerate(set(train['MSZoning'].values))}
    #train = train.fillna({"Embarked":"S"})
    train = train.fillna({"MasVnrArea":0.0})
    #train['MSZoning'] = train['MSZoning'].map(mszoning_map)
    #train['MSZoning'] = setofData(train,'MSZoning')
    #train['SaleType'] = setofData(train,'SaleType')
    for x in train.select_dtypes(include=['object']).keys():
        train[x] = setofData(train,x)

    #sex_map = {"male":0,"female":1}
    #train['Sex'] = train['Sex'].map(sex_map)

    ##seperate with ages in a new group
    #train['Age'] = train['Age'].fillna(-0.5)
    #bins = [0,3000,5000,10000,20000,np.inf]
    #labels = ['A','B','C','D','E']
    #train['AreaGroup'] = pd.cut(train['LotArea'], bins, labels = labels)
    #area_map = {'A':1, 'B':2, 'C':3, 'D':4, 'E':5}
    #train['AreaGroup'] = train['AreaGroup'].map(area_map)
    #train = train.drop(['LotArea'],axis=1)
    #train['MSSubClass'] = setofData(train,'MSSubClass')
    #train['YrSold'] = setofData(train,'YrSold')


    ##Name process
    #train['Title'] = train.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    #train['Title'] = train['Title'].replace(['Lady','Don','Jonkheer'],'Rare')
    #train['Title'] = train['Title'].replace(['Capt','Col','Dr','Major','Rev','Dona'],'Officer')
    #train['Title'] = train['Title'].replace(['Countess','Sir'],'Rare')
    #train['Title'] = train['Title'].replace('Ms','Miss')
    #train['Title'] = train['Title'].replace('Mlle','Miss')
    #train['Title'] = train['Title'].replace('Mme','Mrs')
    #title_map = {"Mr": 1, "Miss":2, "Mrs":3, "Master":4, "Rare":5,'Officer':6}
    #train['Title'] = train["Title"].map(title_map)
    #train = train.drop(['Name'],axis=1)
    #train['FareRange'] = pd.qcut(train['Fare'],4,labels = [1,2,3,4])
    #train = train.drop(['Fare'],axis=1)
    return train

#mapping str values to int ft
def setofData(train,col):
    keys = set(train[col].values)
    #if len(keys)>10:
    #    print(col,' check!')
    #    return train[col]
    map_dict = {x:idx for idx,x in enumerate(keys)}
    train[col] = train[col].map(map_dict)
    return train[col]

from sklearn.model_selection import KFold,cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier


train = data_preprocess('train.csv')
yData = train['SalePrice']
train = train.drop(['SalePrice'],axis=1)
xData = train
#print(set(xData['AreaGroup'].values))
k_fold = KFold(n_splits=10, shuffle=True, random_state=0)
clf = RandomForestClassifier(n_estimators=13)
clf.fit(xData,yData)
score = cross_val_score(clf, xData, yData, cv=k_fold, n_jobs=1, scoring= 'accuracy')
print(score)


#print(set(train['MasVnrArea'].values))
#a = train.isnull().sum()
#print(a[a>0])
