import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

ret = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
ret = ret.dropna(subset=['Age'])
test = test.dropna(subset=['Age'])

def data_preprocess(train):
    train = train.drop(['Cabin'],axis=1)
    train = train.drop(['Ticket'],axis=1)

    embarked_map = {"S":1,"C":2,"Q":3}
    train = train.fillna({"Embarked":"S"})
    train['Embarked'] = train['Embarked'].map(embarked_map)

    sex_map = {"male":0,"female":1}
    train['Sex'] = train['Sex'].map(sex_map)
    bins = [0,10,20,30,40,50,60,70,np.inf]
    labels = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Middle Adult', 'Adult']
    train['AgeGroup'] = pd.cut(train['Age'], bins, labels = labels)
    age_map = {'Baby':1, 'Child':2, 'Teenager':3, 'Student':4, 'Young Adult':5, 'Middle Adult':6,'Adult':7,'Unknown':0}
    train['AgeGroup'] = train['AgeGroup'].map(age_map)
    train = train.drop(['Age'],axis=1)

    #Name process
    train['Title'] = train.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    train['Title'] = train['Title'].replace(['Lady','Don','Jonkheer'],'Rare')
    train['Title'] = train['Title'].replace(['Capt','Col','Dr','Major','Rev','Dona'],'Officer')
    train['Title'] = train['Title'].replace(['Countess','Sir'],'Rare')
    train['Title'] = train['Title'].replace('Ms','Miss')
    train['Title'] = train['Title'].replace('Mlle','Miss')
    train['Title'] = train['Title'].replace('Mme','Mrs')
    title_map = {"Mr": 1, "Miss":2, "Mrs":3, "Master":4, "Rare":5,'Officer':6}
    train['Title'] = train["Title"].map(title_map)
    train = train.drop(['Name'],axis=1)
    train['FareRange'] = pd.qcut(train['Fare'],4,labels = [1,2,3,4])
    train = train.drop(['Fare'],axis=1)
    return train
ret = data_preprocess(ret)
yData = ret['AgeGroup']
ret = ret.drop(['AgeGroup','Survived'],axis=1)
xData = ret
clf = RandomForestClassifier(n_estimators=13)
clf.fit(xData,yData)

test = data_preprocess(test)
test = test.dropna(subset=['FareRange'])
test_label = test['AgeGroup']
test = test.drop(['AgeGroup'],axis=1)
prediction = clf.predict(test)
score = metrics.accuracy_score(test_label,prediction)
print("score:",score)
