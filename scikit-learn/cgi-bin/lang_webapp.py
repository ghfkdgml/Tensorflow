import cgi,os
from sklearn.externals import joblib

pklfile = os.path.dirname(__file__)+"//freq.pkl"
clf = joblib.load(pklfile)

def show_form(text, msg=""):
    print("Content-Type: text/html; charset=utf-8")
    print("")
    print("""
        <html><body><form>
        <textarea name="text" rows="8" cols="40">{0}</textarea>
        <p><input type="submit" value="check"></p>
        <p>{1}</p>
        </form></body></html>
    """.format(cgi.escape(text),msg))

def detect_lang(text):
    text = text.lower()
    code_a, code_z = (ord("a"),ord("z"))
    cnt = [0 for i in range(26)]
    for ch in text:
        n = ord(ch) - code_a
        if 0 <= n and n < 26:cnt[n] += 1
    total = sum(cnt)
    if total == 0: return "no input"
    freq = list(map(lambda n: n/total,cnt))
    res = clf.predict([freq])

    lang_dic = {"e":"English", "f":"France", "i":"Indonesia", "t":"Tagalog"}
    if res[0]:
        return lang_dic[res[0]]
    else:
        return "none"

form = cgi.FieldStorage()
text = form.getvalue("text", default="")
msg = ""
if text != "":
    lang = detect_lang(text)
    msg = "Result:" +lang
show_form(text,msg)
