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

from joblib import dump, load
from todayGames import getInputVector


modelPath = 'logRegModel.joblib'
clf = load(modelPath)

#hometeam is 0 if team0 is home, 1 if team1 is home
def predictGame(team0, team1, date, homeTeam):
    print("getting input vector")
    test = getInputVector(team0,team1,date,homeTeam)
    test2d = np.reshape(test,(1,-1))
    predicts = clf.predict(test2d)
    winner = predicts[0]
    if homeTeam:
        if winner:
            winningTeam = team1
        else:
            winningTeam = team0
    else:
        if winner:
            winningTeam = team0
        else:
            winningTeam = team1

    print(f"predicted winner: {winningTeam}")
    probs = clf.predict_proba(test2d)
    print(f"predicted probabilities: {probs}")

#predictGame("GSW","IND","12/13/21",1)
#phx doesnt work, probably thinks asking for 2022 season ?
predictGame("CHO","DAL","12/13/21",1)