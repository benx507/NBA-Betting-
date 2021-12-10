import json


import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
import sklearn.model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV,train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn import metrics
from tqdm import tqdm
from sklearn.metrics import classification_report,confusion_matrix
from sknn.mlp import Classifier, Layer, Regressor
from sklearn.model_selection import GridSearchCV
import warnings



from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics import r2_score
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier


from sklearn.datasets import load_iris
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
import sys
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")
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
with warnings.catch_warnings():
    warnings.simplefilter("ignore", ConvergenceWarning)
    x,y= load_data(data_path)

    pca = PCA(n_components=46)
    principalComponents = pca.fit_transform(x)
    principalDf = pd.DataFrame(data = principalComponents)
    X_train, X_test, y_train, y_test = train_test_split(principalDf, y, test_size=0.25, random_state=42)
    sc_X = StandardScaler()
    X_trainscaled=sc_X.fit_transform(X_train)
    X_testscaled=sc_X.transform(X_test)

    clf = MLPClassifier(max_iter=200)

    parameter_space = {
        'hidden_layer_sizes': [(100,), (50,100,), (50,75,100,), (200),(100,200),(100,150,200)],
        'activation': ['tanh', 'relu'],
        'solver': ['sgd', 'adam'],
        'alpha': [0.0001, 0.05],
        'learning_rate': ['constant','adaptive'],
    }
    clf = GridSearchCV(clf, parameter_space, n_jobs=-1, cv=5)

    clf.fit(X_trainscaled, y_train)

    print('Best parameters found:\n', clf.best_params_)
    y_test, y_pred = y_test , clf.predict(X_testscaled)
    print('Results on the test set:')
    print('Train Accuracy : %.3f'%clf.best_estimator_.score(X_trainscaled, y_train))
    print('Test Accuracy : %.3f'%clf.best_estimator_.score(X_testscaled, y_test))
    print('Best Accuracy Through Grid Search : %.3f'%clf.best_score_)
    print('Best Parameters : ',clf.best_params_)
