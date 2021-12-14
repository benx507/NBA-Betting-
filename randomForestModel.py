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

#data_path = "2019_last5_v2.csv"
#data_path = "20182019last5.csv"
data_path = "20182019_last10_rt.csv"

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
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

scaler = StandardScaler()
pca = PCA()
forest = RandomForestClassifier()
pipe_steps = [('scaler', scaler),('pca', pca),('forest',forest)]
pipe = Pipeline(pipe_steps)

param_grid = {
    "pca__n_components": [5,7,10,15,20,25],
    'forest__max_depth': [4, 6, 8, None],
}
    #'forest__n_estimators': [50, 150, 250],
    #'forest__min_samples_split': [0.02, 0.05],
    #'forest__min_samples_leaf': [0.01, 0.05]

search = GridSearchCV(pipe, param_grid, n_jobs=-1)
search.fit(X_train,y_train)
print("Best parameter (CV score=%0.3f):" % search.best_score_)
print(search.best_params_)

best_model = search.best_estimator_
print(f"best model score on test set: {best_model.score(X_test,y_test)}")
#print(best_model.predict_proba(X_test[0:10]))
#print(best_model.predict(X_test[0:10]))
#print(y_test[0:10])

#grid search results
# Calling Method 
plot_grid_search(search.cv_results_, param_grid["pca__n_components"], \
    param_grid['forest__max_depth'], 'n pca components', 'forest__max_depth')

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

