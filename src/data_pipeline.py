import pandas as pd
import nfl_data_py as nfl
import os

def create_training_data(seasons=[2023, 2024, 2025]):
    print(f"Fetching schedules and data for {seasons}...")
    games = nfl.import_schedules(seasons)
    pbp = nfl.import_pbp_data(seasons)

    #filter completed regular season games (no ties)
    games = games[(games['game_type'] == 'REG') & (games['result'].notna()) & (games['result'] != 0)].copy()
    games['home_win'] = (games['result'] > 0).astype(int)

    #calculate game-level offensive EPA for each team
    game_epa = pbp.groupby(['game_id', 'posteam'])['epa'].mean().reset_index()
    game_epa.rename(columns={'posteam': 'team', 'epa': 'off_epa'}, inplace=True)

    #make games to pair team appearance with game date
    team_games = games[['game_id', 'gameday', 'home_team', 'away_team']].melt(
        id_vars=['game_id', 'gameday'],
        value_vars=['home_team', 'away_team'],
        var_name='is_home',
        value_name='team'
    )

    #combine stats and sort by date
    team_stats=pd.merge(team_games, game_epa, on=['game_id', 'team'], how='left')
    teams_stats = team_stats.sort_values(by=['team', 'gameday'])

    #rolling 4-game window, shifted by 1
    team_stats['rolling_off_epa'] = team_stats.groupby('team')['off_epa'].transform(
        lambda x: x.rolling(window=4, min_periods=1).mean().shift(1)
    )

    #split into home and away views
    home_stats = team_stats[team_stats['is_home'] == 'home_team'][['game_id', 'rolling_off_epa']]
    away_stats = team_stats[team_stats['is_home'] == 'away_team'][['game_id', 'rolling_off_epa']]

    #merge back to games
    games = pd.merge(games, home_stats.rename(columns={'rolling_off_epa': 'home_rolling_off_epa'}), on='game_id', how='left')
    games = pd.merge(games, away_stats.rename(columns={'rolling_off_epa': 'away_rolling_off_epa'}), on='game_id', how='left')

    #differential feature: home minus away games
    games['diff_rolling_off_epa'] = games['home_rolling_off_epa'] - games['away_rolling_off_epa']

    #remove rows before rolling windows populated
    final_df = games.dropna(subset=['diff_rolling_off_epa']).copy()
    
    os.makedirs('../data/processed', exist_ok=True)
    final_df.to_csv('../data/processed/model_ready_data.csv', index=False)
    print("Data processed and saved to data/processed/model_ready_data.csv")

if __name__ == "__main__":
    create_training_data()
