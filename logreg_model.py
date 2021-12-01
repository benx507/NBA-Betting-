import json

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV,train_test_split
from sklearn import metrics
from tqdm import tqdm

from datetime import datetime

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold

import sys


data_path = "games.csv"

def gen_pipeline(use_pca=True, pca_components=10, scale_data=True):
    pipeline_components = []
    if scale_data:
            scaler = StandardScaler()
            pipeline_components.append(('Standard Scaler', scaler))
    if use_pca:
        pca = PCA(pca_components)
        pipeline_components.append(('PCA', pca))
    return Pipeline(pipeline_components)

def _load_data(data_csv, delete_first_ten, holdout_year):
        """
        Load the data from a csv file. Just return the X and y, splitting into
        train and test data will be handled elsewhere.
        """
        # Load the data
        data = pd.read_csv(data_csv, index_col=0)

        data['outcome'] = data.apply(lambda x: 0 if x['winner'] == 'Away' else 1, axis=1)

        y = data['outcome']
        #meta = data[['home_team', 'away_team']]
        X = data.drop(['winner', 'outcome','winning_abbr'], axis=1)
        # Take care of any NaN values
        X = X.fillna(0)
        return X.to_numpy(), y.to_numpy()

x,y= _load_data(data_path,False, None)
scaler = StandardScaler().fit(x)
#print("b4")
X = scaler.transform(x)
#print("after")
#print(X)
pca = PCA(n_components=4)
principalComponents = pca.fit_transform(X)
principalDf = pd.DataFrame(data = principalComponents)



X_train, X_test, y_train, y_test = train_test_split(principalDf, y, test_size=0.33, random_state=42)

clf = LogisticRegression(max_iter=1000)

clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)

print(y_pred)
print(y_test)

correct = 0

for i in range(len(y_pred)):
    y1 = y_pred[i]
    y2 = y_test[i]
    if y1==y2:
        correct = correct+1

print(f"accuracy: {correct/len(y_pred)}")