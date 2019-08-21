from sklearn import svm, metrics
import random, re

csv = []
with open('iris.csv','r') as f:
    for line in f:
        line = line.strip()
        cols = line.split(',')
        fn = lambda n : float(n) if re.match(r'^[0-9\.]+$',n) else n
        cols =list(map(fn,cols))
        csv.append(cols)

del csv[0]

random.shuffle(csv)

total_len = len(csv)
train_len = int(total_len *2 / 3)
train_data = []
train_label = []
test_data = []
test_label = []

for i in range(total_len):
    data = csv[i][0:4]
    label = csv[i][4]
    if i < train_len:
        train_data.append(data)
        train_label.append(label)
    else:
        test_data.append(data)
        test_label.append(label)

clf = svm.SVC()
clf.fit(train_data, train_label)
predict = clf.predict(test_data)

score = metrics.accuracy_score(test_label, predict)
print("correct:",score)


