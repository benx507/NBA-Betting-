import Scraper
import numpy as np
import pandas as pd

teamID = 1610612739
testUrl = f"https://stats.nba.com/stats/teamplayerdashboard?DateFrom=&DateTo=&GameSegment=&LastNGames=0&LeagueID=&Location=&MeasureType=Base&Month=0&OpponentTeamID=0&Outcome=&PORound=&PaceAdjust=N&PerMode=Totals&Period=0&PlusMinus=N&Rank=N&Season=2020-21&SeasonSegment=&SeasonType=Regular+Season&ShotClockRange=&TeamID={teamID}&VsConference=&VsDivision="

scrape = Scraper.Scraper()

r = scrape.retrieve_json_api_from_url(testUrl)
playersRaw = r["resultSets"][1]["rowSet"]
playerNameIDs = [sublist[1:3] for sublist in playersRaw]
print(playerNameIDs)