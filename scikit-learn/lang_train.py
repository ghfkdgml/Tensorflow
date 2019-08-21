from sklearn import svm,metrics
import glob,os,re,json

def check_frequent(fname):
    name = os.path.basename(fname)
    lang = re.match(r'^[a-z{2,}]',name).group()
    with open(fname, "r",encoding="utf-8") as f:
        text = f.read()
    text = text.lower()
    cnt = [0 for n in range(0,26)]
    code_a = ord("a")#ascii code of char "a"
    code_z = ord("z")

    for ch in text:
        n = ord(ch)
        if code_a<= n and n <=code_z:
            cnt[n - code_a] += 1

    total = sum(cnt)
    freq = list(map(lambda n: n/total,cnt))
    return (freq, lang)

def load_files(path):
    freqs = []
    labels = []
    file_list = glob.glob(path)
    for fname in file_list:
        r = check_frequent(fname)
        freqs.append(r[0])
        labels.append(r[1])
    return {"freqs":freqs,"labels":labels}

def run():
    data = load_files("./lang/train/*.txt")
    test = load_files("./lang/test/*.txt")

    with open("./lang/freq.json","w") as fp:
        json.dump([data,test],fp)

    clf = svm.SVC()
    clf.fit(data["freqs"],data["labels"])

    predict = clf.predict(test["freqs"])

    score = metrics.accuracy_score(test["labels"],predict)
    print("correct:",score)
    report = metrics.classification_report(test["labels"],predict)
    print("report:",report)

if __name__ == "__main__":
    run()
