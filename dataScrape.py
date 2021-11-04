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
    # for team in TEAM_ABBREVIATIONS + [nets_abbr, cha_abbr, pels_abbr]:
    team = 'UTA'
    sched = schedule.Schedule(team, year = year)
    for game in tqdm(sched):
        
        box_score_index = game.boxscore_index
        if box_score_index in game_id:
            continue
        game_id.append(box_score_index)
        box_score_df = game.boxscore.dataframe
        game_data.append(box_score_df)

    year_data = pd.concat(game_data, axis=0)
    csv = Path(csv)
    if not csv.exists():
        year_data.to_csv(str(csv), mode='w+', header=True)
    else:
        year_data.to_csv(str(csv), mode='a', header=False)
    game_data = []


create_df(2018)