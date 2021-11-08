import Scraper
import numpy as np
import pandas as pd

testUrl = "https://stats.nba.com/stats/playerestimatedmetrics?LeagueID=00&Season=2019-20&SeasonType=Regular+Season"

scrape = Scraper.Scraper()

r = scrape.retrieve_json_api_from_url(testUrl)
playersRaw = r["resultSet"]["rowSet"]
dfHeaders = r["resultSet"]["headers"]
df = pd.DataFrame(playersRaw,columns=dfHeaders)
print(df)