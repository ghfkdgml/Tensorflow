import pandas as pd
import numpy as np
import tensorflow as tf

train = pd.read_csv('train.csv')
train = train.drop(['Cabin'],axis=1)
#train = train.drop(['Name'],axis=1)
#train = train.drop(['Fare'],axis=1)
#train = train.drop(['PassengerId'],axis=1)
train = train.drop(['Ticket'],axis=1)
embarked_map = {"S":1,"C":2,"Q":3}
train = train.fillna({"Embarked":"S"})
train['Embarked'] = train['Embarked'].map(embarked_map)
sex_map = {"male":0,"female":1}
train['Sex'] = train['Sex'].map(sex_map)
#seperate with ages in a new group
train['Age'] = train['Age'].fillna(-0.5)
bins = [-1,0,5,12,18,24,35,60,np.inf]
labels = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']
train['AgeGroup'] = pd.cut(train['Age'], bins, labels = labels)
age_map = {'Baby':1, 'Child':2, 'Teenager':3, 'Student':4, 'Young Adult':5, 'Adult':6, 'Senior':7,'Unknown':0}
train['AgeGroup'] = train['AgeGroup'].map(age_map)
train = train.drop(['Age'],axis=1)

#Name process
train['Title'] = train.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
train['Title'] = train['Title'].replace(['Lady','Capt','Col','Don','Dr','Major','Rev','Jonkheer','Dona'],'Rare')
train['Title'] = train['Title'].replace(['Countess','Sir'],'Master')
train['Title'] = train['Title'].replace('Ms','Miss')
train['Title'] = train['Title'].replace('Mlle','Miss')
train['Title'] = train['Title'].replace('Mme','Mrs')
title_map = {"Mr": 1, "Miss":2, "Mrs":3, "Master":4, "Rare":5}
train['Title'] = train["Title"].map(title_map)
train = train.drop(['Name'],axis=1)

train['FareRange'] = pd.qcut(train['Fare'],4, labels = [1,2,3,4])
train = train.drop(['Fare'],axis=1)
#xy = np.loadtxt('data4_zoo.csv', delimiter=',', dtype=np.float32)
#xData = xy[:,:-1]
#yData = xy[:,[-1]]
yData = train['Survived']
#yData = [[float(yData[n])] for n in range(len(yData))]
train = train.drop(['Survived'],axis=1)
xData = train

from sklearn.model_selection import KFold,cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier


k_fold = KFold(n_splits=10, shuffle=True, random_state=0)
clf = RandomForestClassifier(n_estimators=13)
score = cross_val_score(clf, xData, yData, cv=k_fold, n_jobs=1, scoring= 'accuracy')
print(score)

#prediction = clf.predict(test_data)
#result = pd.DataFram({"PassengerId":
