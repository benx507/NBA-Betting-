import json

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV,train_test_split
from sklearn import metrics
from tqdm import tqdm

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier

import sys

data_path = "2019_last10.csv"
data_path = "20182019last5.csv"

def load_data(data_csv):
        """
        Load the data from a csv file. Just return the X and y, splitting into
        train and test data will be handled elsewhere.
        """
        # Load the data
        data = pd.read_csv(data_csv, index_col=0)

        data['outcome'] = data.apply(lambda x: 0 if x['winner'] == 'Away' else 1, axis=1)

        y = data['outcome']
        X = data.drop(['winner', 'outcome','winning_abbr'], axis=1)
        # Take care of any NaN values
        X = X.fillna(0)
        return X.to_numpy(), y.to_numpy()

x,y= load_data(data_path)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

scaler = StandardScaler()
pca = PCA()
forest = RandomForestClassifier()
pipe_steps = [('scaler', scaler),('pca', pca),('forest',forest)]
pipe = Pipeline(pipe_steps)

param_grid = {
    "pca__n_components": [3,4,5,7,10],
    'forest__max_depth': [6, 8, None],
    #'forest__n_estimators': [50, 150, 250],
    #'forest__min_samples_split': [0.02, 0.05],
    #'forest__min_samples_leaf': [0.01, 0.05]
}
search = GridSearchCV(pipe, param_grid, n_jobs=-1)
search.fit(X_train,y_train)
print("Best parameter (CV score=%0.3f):" % search.best_score_)
print(search.best_params_)

best_model = search.best_estimator_
print(f"best model score on test set: {best_model.score(X_test,y_test)}")
#print(best_model.predict_proba(X_test[0:10]))
#print(best_model.predict(X_test[0:10]))
#print(y_test[0:10])

probs = best_model.predict_proba(X_test)
predicts = best_model.predict(X_test)
count = 0
sum = 0
indices = []
for i in range(len(predicts)):
    threshold = 0.1
    if probs[i][0] <= 0.5 - threshold or probs[i][0] >= 0.5 + threshold:
        sum = sum+1
        indices.append(i)
        if predicts[i] == y_test[i]:
            count = count+1

#print(probs[indices])
#print(predicts[indices])
#print(y_test[indices])
print(f"threshold accuracy: {count/sum}")