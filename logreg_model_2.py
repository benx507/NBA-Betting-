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

from joblib import dump,load

data_path = "20182019last5.csv"
#data_path = "20182019_last10_rt.csv"
#data_path = "2019_last5_v2.csv"

def plot_grid_search(cv_results, grid_param_1, grid_param_2, name_param_1, name_param_2):
    # Get Test Scores Mean and std for each grid search
    scores_mean = cv_results['mean_test_score']
    scores_mean = np.array(scores_mean).reshape(len(grid_param_2),len(grid_param_1))

    scores_sd = cv_results['std_test_score']
    scores_sd = np.array(scores_sd).reshape(len(grid_param_2),len(grid_param_1))

    # Plot Grid search scores
    _, ax = plt.subplots(1,1)

    # Param1 is the X-axis, Param 2 is represented as a different curve (color line)
    for idx, val in enumerate(grid_param_2):
        ax.plot(grid_param_1, scores_mean[idx,:], '-o', label= name_param_2 + ': ' + str(val))

    ax.set_title("Grid Search Scores", fontsize=20, fontweight='bold')
    ax.set_xlabel(name_param_1, fontsize=16)
    ax.set_ylabel('CV Average Score', fontsize=16)
    ax.legend(loc="best", fontsize=5)
    ax.grid('on')
    plt.show()

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

def load_data2(data_csv):
        """
        Load the data from a csv file. Just return the X and y, splitting into
        train and test data will be handled elsewhere.
        """
        # Load the data
        data = pd.read_csv(data_csv, index_col=0)

        y = data['outcome']
        X = data.drop(['game_id', 'outcome'], axis=1)
        # Take care of any NaN values
        X = X.fillna(0)
        return X.to_numpy(), y.to_numpy()

x,y= load_data(data_path)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25,random_state=42)
#random_state = 42

scaler = StandardScaler()
pca = PCA()
logistic = LogisticRegression()
pipe_steps = [('scaler', scaler),('pca', pca),('logistic',logistic)]
pipe = Pipeline(pipe_steps)

param_grid = {
    "pca__n_components": [5,7,10,15,20,25],
    "logistic__C": np.logspace(-4, 4, 4),
}
search = GridSearchCV(pipe, param_grid, n_jobs=-1)
search.fit(X_train,y_train)
print("Best parameter (CV score=%0.3f):" % search.best_score_)
print(search.best_params_)

#grid search results
# Calling Method 
plot_grid_search(search.cv_results_, param_grid["pca__n_components"], \
    param_grid['logistic__C'], 'n pca components', 'logistic c')

#testing pipeline-- all three do the same thing
best_model = search.best_estimator_
print(f"best model score on test set: {best_model.score(X_test,y_test)}")

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

print(f"threshold accuracy: {count/sum}")

modelPath = "logRegModel.joblib"
dump(best_model,modelPath)

# test = Pipeline(pipe_steps).set_params(**search.best_params_)
# test.fit(X_train,y_train)
# print(test.score(X_test,y_test))

# test = Pipeline(pipe_steps).fit(X_train,y_train)
# test.set_params(**search.best_params_)
# print(test.score(X_test,y_test))

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