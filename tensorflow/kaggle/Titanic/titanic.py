import pandas as pd
import numpy as np
import tensorflow as tf

def data_preprocess(data):
    train = pd.read_csv(data)
    train = train.drop(['Cabin'],axis=1)
    train = train.drop(['Ticket'],axis=1)

    embarked_map = {"S":1,"C":2,"Q":3}
    train = train.fillna({"Embarked":"S"})
    train['Embarked'] = train['Embarked'].map(embarked_map)

    sex_map = {"male":0,"female":1}
    train['Sex'] = train['Sex'].map(sex_map)

    #seperate with ages in a new group
    train['Age'] = train['Age'].fillna(-0.5)
    bins = [-1,0,5,12,18,24,35,45,55,65,np.inf]
    labels = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Middle Adult', 'Adult','Young Senior', 'Senior']
    train['AgeGroup'] = pd.cut(train['Age'], bins, labels = labels)
    age_map = {'Baby':1, 'Child':2, 'Teenager':3, 'Student':4, 'Young Adult':5, 'Middle Adult':6,'Adult':7, 'Young Senior':8,'Senior':9,'Unknown':0}
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
    
#Data get
#train = pd.read_csv('train.csv')
#train = train.drop(['Cabin'],axis=1)
#train = train.drop(['Ticket'],axis=1)
#embarked_map = {"S":1,"C":2,"Q":3}
#train = train.fillna({"Embarked":"S"})
#train['Embarked'] = train['Embarked'].map(embarked_map)
#sex_map = {"male":0,"female":1}
#train['Sex'] = train['Sex'].map(sex_map)
##seperate with ages in a new group
#train['Age'] = train['Age'].fillna(-0.5)
#bins = [-1,0,5,12,18,24,35,60,np.inf]
#labels = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']
#train['AgeGroup'] = pd.cut(train['Age'], bins, labels = labels)
#age_map = {'Baby':1, 'Child':2, 'Teenager':3, 'Student':4, 'Young Adult':5, 'Adult':6, 'Senior':7,'Unknown':0}
#train['AgeGroup'] = train['AgeGroup'].map(age_map)
#train = train.drop(['Age'],axis=1)
##
##Name process
#train['Title'] = train.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
#train['Title'] = train['Title'].replace(['Lady','Capt','Col','Don','Dr','Major','Rev','Jonkheer','Dona'],'Rare')
#train['Title'] = train['Title'].replace(['Countess','Sir'],'Master')
#train['Title'] = train['Title'].replace('Ms','Miss')
#train['Title'] = train['Title'].replace('Mlle','Miss')
#train['Title'] = train['Title'].replace('Mme','Mrs')
#title_map = {"Mr": 1, "Miss":2, "Mrs":3, "Master":4, "Rare":5}
#train['Title'] = train["Title"].map(title_map)
#train = train.drop(['Name'],axis=1)
#
#train['FareRange'] = pd.qcut(train['Fare'],4, labels = [1,2,3,4])
#train = train.drop(['Fare'],axis=1)
#xy = np.loadtxt('data4_zoo.csv', delimiter=',', dtype=np.float32)
#xData = xy[:,:-1]
#yData = xy[:,[-1]]
train = data_preprocess('train.csv')
yData = train['Survived']
#yData = [[float(yData[n])] for n in range(len(yData))]
train = train.drop(['Survived'],axis=1)
xData = train

from sklearn.model_selection import KFold,cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier


k_fold = KFold(n_splits=10, shuffle=True, random_state=0)
clf = RandomForestClassifier(n_estimators=13)
clf.fit(xData,yData)
score = cross_val_score(clf, xData, yData, cv=k_fold, n_jobs=1, scoring= 'accuracy')
print(score)

test_data = data_preprocess('test.csv')
test_data = test_data.fillna({"FareRange":1})
#test_data = test_data.drop('Survived', axis=1)
prediction = clf.predict(test_data)
submission = pd.DataFrame({
    "PassengerId": test_data["PassengerId"],
    "Survived": prediction})

submission.to_csv('submission.csv', index=False)

submission = pd.read_csv('submission.csv')
print(submission.head())
