import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from tqdm import tqdm

import sportsipy.nba.schedule as schedule

TEAM_ABBREVIATIONS = ['ATL',
                     'BOS',
                     'CHI',
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

#filling csv with team-level data
def create_df(year, csv='games.csv'):
    game_id = []
    game_data = []
    nets_abbr = 'NJN' if year < 2013 else 'BRK'
    cha_abbr = 'CHA' if year < 2015 else 'CHO'
    pels_abbr = 'NOH' if year < 2014 else 'NOP'
    #we can run this loop when we want to fill out all the teams, right now, I am just doing Utah 2018 season as a test
    for team in tqdm(TEAM_ABBREVIATIONS + [nets_abbr, cha_abbr, pels_abbr]):
        sched = schedule.Schedule(team, year = year)
        for game in tqdm(sched):
            
            box_score_index = game.boxscore_index
            if box_score_index in game_id:
                continue
            game_id.append(box_score_index)
            box_score_df = game.boxscore.dataframe
            box_score_df = box_score_df.drop(['home_free_throw_percentage',\
                'home_three_point_field_goal_percentage'], axis=1)

            game_data.append(box_score_df)

        year_data = pd.concat(game_data, axis=0)
        year_data['date'] = pd.to_datetime(year_data['date'], infer_datetime_format=True)
        year_data = year_data.sort_values('date', ascending=True)

        winner_cols= year_data[["winner", "winning_abbr"]]

        n_back = [10]
        last_n_dfs = []
        for n in n_back:
            last_n = year_data.drop(['date'], axis=1)\
                                        .rolling(n, min_periods=1).mean()
            last_n_dfs.append(last_n)
        
        frames = [last_n_dfs[0], winner_cols]
        res = pd.concat(frames, axis= 1)

        csv = Path(csv)
        if not csv.exists():
            res.to_csv(str(csv), mode='w+', header=True)
        else:
            res.to_csv(str(csv), mode='a', header=False)
        game_data = []


create_df(2019,csv='2019_last10.csv')