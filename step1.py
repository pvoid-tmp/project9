import pandas as pd
import dill
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion


class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.key]


class TextImputer(BaseEstimator, TransformerMixin):
    def __init__(self, key, value):
        self.key = key
        self.value = value

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X[self.key] = X[self.key].fillna(self.value)
        return X


df = pd.read_csv("./cleaned.csv")
X_train, X_test, y_train, y_test = train_test_split(df, df['label'], test_size=0.33, random_state=42)

X_test.to_csv("X_test.csv", index=None)
y_test.to_csv("y_test.csv", index=None)


features = ['text']
target = 'label'


text = Pipeline([
                ('imputer', TextImputer('text', '')),
                ('selector', ColumnSelector(key='text')),
                ('tfidf', TfidfVectorizer(max_df=0.9, min_df=10))
                ])

feats = FeatureUnion([('text', text)])


pipeline = Pipeline([
    ('features', feats),
    ('classifier', LogisticRegression()),
])

pipeline.fit(X_train, y_train)

with open("./models/logreg_pipeline.dill", "wb") as f:
    dill.dump(pipeline, f)
