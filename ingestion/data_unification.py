import os
import pandas as pd
import numpy as np
import LanusStats  as ls
import soccerdata as sd

# fbref = ls.Fbref()

# print(ls.get_available_leagues('Fbref'))
# jugadores= fbref.get_all_player_season_stats("La Liga", "2024-2025", save_csv=False)
# #print(jugadores.head())
# print(jugadores.dtype())

fbref = sd.FBref('ENG-Premier League', '2021')

# Fetch data
games = fbref.read_schedule()
team_season_stats = fbref.read_team_season_stats(stat_type="standard")
player_season_stats = fbref.read_player_season_stats(stat_type="standard")
print(player_season_stats.head())