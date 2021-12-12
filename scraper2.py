import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from tqdm import tqdm

import sportsipy.nba.schedule as schedule

TEAM_ABBREVIATIONS = ['ATL',
                     'BOS',
                     'BRK',
                     'CHI',
                     'CHO',
                     'CLE',
                     'DAL',
                     'DEN',
                     'DET',
                     'GSW',
                     'HOU',
                     'IND',
                     'LAC',
                     'LAL',
                     'MEM',
                     'MIA',
                     'MIL',
                     'MIN',
                     'NOP',
                     'NYK',
                     'OKC',
                     'ORL',
                     'PHI',
                     'PHO',
                     'POR',
                     'SAC',
                     'SAS',
                     'TOR',
                     'UTA',
                     'WAS']

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

def stripVector(v,home):
    if home:
        v = v[37:]
        return v[:len(v)-1]
    else:
        v = v[:37]
        a = v[:13]
        b = v[14:26]
        c = v[27:]
        return np.concatenate([a,b,c])

def create_dict(year):
    team_dict = {}
    for team in tqdm(TEAM_ABBREVIATIONS):
            path = str(year)+"_"+team+".csv"
            data = pd.read_csv(path, index_col=0)
            data = data.sort_values('date', ascending=False)
            team_dict[team] = data
    return team_dict

def averageLastN(n,team,dict,date):
    date = pd.to_datetime(date)
    teamData = dict[team]
    #flag is 1 if invalid
    flag = 0
    idx = -1
    count = 0
    for index, row in teamData.iterrows():
        if pd.to_datetime(row['date']) < date:
            idx = count
            break
        count = count+1
    end = idx+n
    numRows = len(teamData.index)
    if end > numRows-1:
        end = numRows-1
    #for first two games with no data
    if idx == -1 or end == idx:
        flag = 1
    gameData = []
    for i in range(idx,end):
        linedf = teamData.iloc[i]
        home = teamData.iloc[i]['location']
        
        numeric = linedf.drop(['date','location',\
                    'losing_abbr','losing_name','winner','winning_abbr',\
                        'winning_name','opponent'])
        stripped = stripVector(numeric.to_numpy(),home)
        gameData.append(stripped)
    
    #gameData=np.stack(gameData)
    #print("Game data:")
    #print(gameData)
    return np.mean(gameData,axis=0),flag      

def season_data(year,dict,path):
    game_id = []
    game_data = []
    outcomes = []
    ids = []

    for team in tqdm(TEAM_ABBREVIATIONS):
        sched = schedule.Schedule(team, year = year)
        for game in tqdm(sched):
            box_score_index = game.boxscore_index
            if box_score_index in game_id:
                continue
            game_id.append(box_score_index)
            opponent = game.opponent_abbr
            home = (game.location == "Home")
            date = pd.to_datetime(game.date)
            winner = game.result
            
            if home:
                homeName = team
                awayName = opponent
            else:
                homeName = opponent
                awayName = team

            print(homeName)
            print(date)
            homeData,flag1 = averageLastN(5,homeName,dict,date)
            awayData,flag2 = averageLastN(5,awayName,dict,date)
            if flag1 or flag2:
                continue
            if home:
                if winner:
                    outcomes.append(1)
                else:
                    outcomes.append(0)
            else:
                if winner:
                    outcomes.append(0)
                else:
                    outcomes.append(1)
            ids.append(box_score_index)
            game_data.append(np.append(homeData,awayData))

        teamDf = pd.DataFrame(game_data)
        teamDf['outcome'] = outcomes
        teamDf['game_id'] = ids
        teamDf.set_index('game_id')

        csv = Path(path)
        if not csv.exists():
            teamDf.to_csv(str(csv), mode='w+', header=True)
        else:
            teamDf.to_csv(str(csv), mode='a', header=False)
        game_data = []
        outcomes = []
        ids = []


d = create_dict(2019)
#print(averageLastN(5,'MEM',d,'2018-10-25'))
season_data(2019,d,"2019_last5_v2.csv")
