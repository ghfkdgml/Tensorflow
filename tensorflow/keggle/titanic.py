import pandas as pd
import numpy as np

train = pd.read_csv('train.csv')
train = train.drop(['Cabin'],axis=1)
#train = train.drop(['Name'],axis=1)
train = train.drop(['Fare'],axis=1)
train = train.drop(['PassengerId'],axis=1)
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

#Name process
train['Title'] = train.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
train['Title'] = train['Title'].replace(['Lady','Capt','Col','Don','Dr','Major','Rev','Jonkheer','Dona'],'Rare')
train['Title'] = train['Title'].replace(['Countess','Sir'],'Master')
train['Title'] = train['Title'].replace('Ms','Miss')
train['Title'] = train['Title'].replace('Mlle','Miss')
train['Title'] = train['Title'].replace('Mme','Mrs')
title_map = {"Mr": 1, "Miss":2, "Mrs":3, "Master":4, "Rare":5}
train['Title'] = train["Title"].map(title_map)
print(train[['Title','Survived']].groupby(['Title'], as_index=False).mean())

