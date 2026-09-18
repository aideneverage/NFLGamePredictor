import pandas as pd
import nfl_data_py as nfl
import os

def create_training_data(seasons=[2023, 2024, 2025]):
    print(f"Fetching schedules and play-by-play data for {seasons}...")
    games = nfl.import_schedules(seasons)
    pbp = nfl.import_pbp_data(seasons)

    # filter to completed regular season games without ties
    games = games[(games['game_type'] == 'REG') & (games['result'].notna()) & (games['result'] != 0)].copy()
    games['home_win'] = (games['result'] > 0).astype(int)

    # combine interceptions and lost fumbles into turnovers
    pbp['turnover'] = pbp['interception'].fillna(0) + pbp['fumble_lost'].fillna(0)

    # --- OFFENSIVE METRICS ---
    game_off = pbp.groupby(['game_id', 'posteam']).agg(
        off_epa=('epa', 'mean'),
        turnovers_lost=('turnover', 'sum')
    ).reset_index().rename(columns={'posteam': 'team'})

    # --- DEFENSIVE METRICS ---
    # epa is relative to the posteam, so for the defense, we flip the sign (or use it as points allowed)
    game_def = pbp.groupby(['game_id', 'defteam']).agg(
        def_epa=('epa', 'mean'), # EPA allowed per play (lower is better for defense)
        turnovers_forced=('turnover', 'sum')
    ).reset_index().rename(columns={'defteam': 'team'})

    # merge Offense and Defense game-level stats
    game_stats = pd.merge(game_off, game_def, on=['game_id', 'team'], how='outer')
    game_stats['net_turnovers'] = game_stats['turnovers_forced'] - game_stats['turnovers_lost']

    # reshape games to pair every team appearance with the game date
    team_games = games[['game_id', 'gameday', 'home_team', 'away_team']].melt(
        id_vars=['game_id', 'gameday'], 
        value_vars=['home_team', 'away_team'], 
        var_name='is_home', 
        value_name='team'
    )
    
    # combine stats and sort chronologically
    team_stats = pd.merge(team_games, game_stats, on=['game_id', 'team'], how='left')
    team_stats = team_stats.sort_values(by=['team', 'gameday'])

    # rolling 4-game window, shifted by 1 to prevent lookahead bias
    for col in ['off_epa', 'def_epa', 'net_turnovers']:
        team_stats[f'rolling_{col}'] = team_stats.groupby('team')[col].transform(
            lambda x: x.rolling(window=4, min_periods=1).mean().shift(1)
        )

    # split into home and away views
    home_stats = team_stats[team_stats['is_home'] == 'home_team'][
        ['game_id', 'rolling_off_epa', 'rolling_def_epa', 'rolling_net_turnovers']
    ]
    home_stats = home_stats.rename(columns=lambda x: f"home_{x}" if x != 'game_id' else x)

    away_stats = team_stats[team_stats['is_home'] == 'away_team'][
        ['game_id', 'rolling_off_epa', 'rolling_def_epa', 'rolling_net_turnovers']
    ]
    away_stats = away_stats.rename(columns=lambda x: f"away_{x}" if x != 'game_id' else x)

    # merge back to games
    games = pd.merge(games, home_stats, on='game_id', how='left')
    games = pd.merge(games, away_stats, on='game_id', how='left')

    # differential features: Home minus Away
    games['diff_rolling_off_epa'] = games['home_rolling_off_epa'] - games['away_rolling_off_epa']
    games['diff_rolling_def_epa'] = games['home_rolling_def_epa'] - games['away_rolling_def_epa']
    games['diff_rolling_turnovers'] = games['home_rolling_net_turnovers'] - games['away_rolling_net_turnovers']

    # remove rows before rolling windows populated
    final_df = games.dropna(subset=['diff_rolling_off_epa', 'diff_rolling_def_epa', 'diff_rolling_turnovers']).copy()
    
    os.makedirs('../data/processed', exist_ok=True)
    final_df.to_csv('../data/processed/model_ready_data.csv', index=False)
    print("Data processed and saved to data/processed/model_ready_data.csv")

if __name__ == "__main__":
    create_training_data()
    