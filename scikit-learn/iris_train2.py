from sklearn import svm, metrics
from sklearn.model_selection import train_test_split
import pandas as pd

csv = pd.read_csv('iris.csv')

csv_data = csv[["SepalLength","SepalWidth","PetalLength","PetalWidth"]]
csv_label = csv["Name"]

total_len = len(csv)
train_len = int(total_len *2 / 3)
train_data, test_data, train_label, test_label =\
        train_test_split(csv_data,csv_label)

clf = svm.SVC()
clf.fit(train_data, train_label)
predict = clf.predict(test_data)

score = metrics.accuracy_score(test_label, predict)
print("correct:",score)


