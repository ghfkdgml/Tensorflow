from sklearn import svm,metrics
import pandas as pd

xor_input = [
        [0,0,0],
        [0,1,1],
        [1,0,1],
        [1,1,0]]

xor_df = pd.DataFrame(xor_input)
xor_data = xor_df.ix[:,0:1]
xor_label = xor_df.ix[:,2]

clf = svm.SVC()
clf.fit(xor_data,xor_label)
predict = clf.predict(xor_data)

score = metrics.accuracy_score(xor_label,predict)
print("correct:",score)
