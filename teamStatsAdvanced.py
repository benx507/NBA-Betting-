import Scraper
import numpy as np
import pandas as pd

testUrl = "https://stats.nba.com/stats/teamestimatedmetrics?LeagueID=00&Season=2020-21&SeasonType=Regular+Season"


scrape = Scraper.Scraper()

r = scrape.retrieve_json_api_from_url(testUrl)
teamsRaw = r["resultSet"]["rowSet"]
dfHeaders = r["resultSet"]["headers"]
df = pd.DataFrame(teamsRaw,columns=dfHeaders)
print(df)