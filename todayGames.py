from sportsipy.nba.schedule import Schedule
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from tqdm import tqdm

def padHomeVector(v):
    #indices: home ft: should be at 13?  home 3pt: should be at 26?
    a = v[:13]
    ft = np.array([80.0])
    b = v[13:25]
    threept = np.array([35])
    c = v[25:len(v)-1]
    return np.concatenate([a,ft,b,threept,c])

def stripAwayVector(v):
    a = v[:13]
    b = v[14:26]
    c = v[27:]
    pace = [100.0]
    return np.concatenate([a,b,c,pace])

def getTeamData(team, date, home):
    game_id = []
    game_data = []
    today_date = pd.to_datetime(date)
    sched = Schedule(team, year = today_date.year+1)
    for game in tqdm(sched):
            game_date = pd.to_datetime(game.date)
            if game_date < today_date:
                box_score_index = game.boxscore_index
                if box_score_index in game_id:
                    continue
                game_id.append(box_score_index)
                box_score_df = game.boxscore.dataframe
                box_score_df = box_score_df.drop(['home_free_throw_percentage',\
                    'home_three_point_field_goal_percentage'], axis=1)

                game_data.append(box_score_df)
            else:
                print("breaking")
                break

    season_data = pd.concat(game_data, axis=0)
    print(f"season data length:{len(season_data)}")
    season_data['date'] = pd.to_datetime(season_data['date'], infer_datetime_format=True)
    year_data = season_data.sort_values('date', ascending=True)

    winner_cols= year_data[["winner", "winning_abbr"]]

    n_back = [10]
    features = None
    for n in n_back:
        last_n = year_data.drop(['date'], axis=1)\
                                        .rolling(n, min_periods=1).mean()
        data = last_n.iloc[len(last_n)-1]

        if features is None:
            features = data.to_numpy()
        else:
            features = np.append(features,data.to_numpy())

    if home:
        features = padHomeVector(features[37:])
    else:
        features = stripAwayVector(features[:37])
 
    return features

def getInputVector(team1,team2,date,home):
    feats1 = getTeamData(team1,date,(not home))
    feats2 = getTeamData(team2,date,home)
    if home:
        return np.concatenate([feats1,feats2])
    else:
        return np.concatenate([feats2,feats1])
    
# unsplit = getTeamData('GSW','10/27/21',1)
# print(f"lenght unsplit:{len(unsplit)}")

# homesplit = unsplit[37:]
# awaysplit = unsplit[:37]
# print(homesplit)
# print(len(homesplit))
# print(awaysplit)
# print(len(awaysplit))

# homesplit1 = padHomeVector(homesplit)
# awaysplit1 = stripAwayVector(awaysplit)
# print(homesplit1)
# print(len(homesplit1))
# print(awaysplit1)
# print(len(awaysplit1))

test = getInputVector("GSW","PHI","10/25/21",1)
print(test)
