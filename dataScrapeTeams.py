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

#filling csv with team-level data
def create_df(year,team, csv='games.csv'):
    game_id = []
    game_data = []
    #we can run this loop when we want to fill out all the teams, right now, I am just doing Utah 2018 season as a test
    for team1 in [team]:
        sched = schedule.Schedule(team1, year = year)
        for game in tqdm(sched):
            #location is 1 for home
            location = (game.location == 'Home')
            opponent = game.opponent_abbr
            box_score_index = game.boxscore_index
            if box_score_index in game_id:
                continue
            game_id.append(box_score_index)
            box_score_df = game.boxscore.dataframe
            box_score_df = box_score_df.drop(['home_free_throw_percentage',\
                'home_three_point_field_goal_percentage'], axis=1)
            box_score_df['location'] = [location]
            box_score_df['opponent'] = [opponent]

            game_data.append(box_score_df)

        team_data = pd.concat(game_data, axis=0)
        team_data['date'] = pd.to_datetime(team_data['date'], infer_datetime_format=True)
        team_data = team_data.sort_values('date', ascending=False)

        winner_cols= team_data[["winner", "winning_abbr"]]

        csv = Path(csv)
        if not csv.exists():
            team_data.to_csv(str(csv), mode='w+', header=True)
        else:
            team_data.to_csv(str(csv), mode='a', header=False)
        game_data = []

def create_df_all(year):
    for team in tqdm(TEAM_ABBREVIATIONS):
        path = str(year)+"_"+team+".csv"
        create_df(year,team,csv=path)

create_df_all(2019)