from sklearn import svm
from sklearn.externals import joblib
import json

with open('./lang/freq.json',"r",encoding="utf-8") as f:
    d = json.load(f)
    data = d[0]

clf = svm.SVC()
clf.fit(data["freqs"],data["labels"])

joblib.dump(clf, "./lang/freq.pkl")

