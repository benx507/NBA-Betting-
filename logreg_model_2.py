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

import sys

data_path = "2018_last5_fixed.csv"
data_path = "20182019last5.csv"

def gen_pipeline(use_pca=True, pca_components=10, scale_data=True):
    pipeline_components = []
    if scale_data:
            scaler = StandardScaler()
            pipeline_components.append(('Standard Scaler', scaler))
    if use_pca:
        pca = PCA(pca_components)
        pipeline_components.append(('PCA', pca))

    return Pipeline(pipeline_components)

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
logistic = LogisticRegression()
pipe_steps = [('scaler', scaler),('pca', pca),('logistic',logistic)]
pipe = Pipeline(pipe_steps)

param_grid = {
    "pca__n_components": [5, 10, 15, 20, 25],
    "logistic__C": np.logspace(-4, 4, 4),
}
search = GridSearchCV(pipe, param_grid, n_jobs=-1)
search.fit(X_train,y_train)
print("Best parameter (CV score=%0.3f):" % search.best_score_)
print(search.best_params_)

#testing pipeline-- all three do the same thing
best_model = search.best_estimator_
print(f"best model score on test set: {best_model.score(X_test,y_test)}")

test = Pipeline(pipe_steps).set_params(**search.best_params_)
test.fit(X_train,y_train)
print(test.score(X_test,y_test))

test = Pipeline(pipe_steps).fit(X_train,y_train)
test.set_params(**search.best_params_)
print(test.score(X_test,y_test))

# log_pipe = gen_pipeline()

# X_train = log_pipe.fit_transform(X_train_raw)
# X_test = log_pipe.transform(X_test_raw)

# clf = LogisticRegression()
# clf.fit(X_train,y_train)

# y_pred = clf.predict(X_test)
# print(clf.score(X_test, y_test))

# pipe.fit(X_train_raw,y_train)
# print(pipe.predict(X_test_raw))
# print(pipe.score(X_test_raw, y_test))