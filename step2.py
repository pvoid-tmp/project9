import pandas as pd
from sklearn.metrics import roc_auc_score
import dill
dill._dill._reverse_typemap['ClassType'] = type

X_test = pd.read_csv("X_test.csv")
y_test = pd.read_csv("y_test.csv")

with open('./models/logreg_pipeline.dill', 'rb') as in_strm:
    pipeline = dill.load(in_strm)

predictions = pipeline.predict_proba(X_test)

print(roc_auc_score(y_score=predictions[:, 1][:], y_true=y_test))
