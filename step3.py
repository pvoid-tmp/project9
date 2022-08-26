import pandas as pd
from sklearn.metrics import roc_auc_score
from urllib import request
import urllib.request
import json

X_test = pd.read_csv("X_test.csv")
y_test = pd.read_csv("y_test.csv")


def get_prediction(x):
    text, = x
    body = {'text': text,
            }

    myurl = "http://127.0.0.1:8180/predict"
    req = urllib.request.Request(myurl)
    req.add_header('Content-Type', 'application/json; charset=utf-8')
    jsondata = json.dumps(body)
    jsondataasbytes = jsondata.encode('utf-8')   # needs to be bytes
    req.add_header('Content-Length', len(jsondataasbytes))
    response = urllib.request.urlopen(req, jsondataasbytes)
    return json.loads(response.read())['predictions']


predictions = X_test[['text']].apply(lambda x: get_prediction(x), 1)
print(roc_auc_score(y_score=predictions.values, y_true=y_test))
