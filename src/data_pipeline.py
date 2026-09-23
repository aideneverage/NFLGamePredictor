import pandas as pd
import numpy as np
import nfl_data_py as nfl
import os

def create_training_data(seasons=[2023, 2024, 2025, 2026]):
    print(f"Fetching schedules and play-by-play data for {seasons}...")
    games = nfl.import_schedules(seasons)
    
    try:
        pbp = nfl.import_pbp_data(seasons)
    except Exception:
        print("Note: 2026 play-by-play not available yet. Using prior years for rolling stats.")
        pbp = nfl.import_pbp_data([s for s in seasons if s < 2026])

    # KEEP unplayed games (result is NaN), drop ties
    games = games[(games['game_type'] == 'REG') & (games['result'] != 0)].copy()
    
    # 1 if home won, 0 if away won, NaN if game hasn't happened yet
    games['home_win'] = np.where(games['result'].notna(), (games['result'] > 0).astype(float), np.nan)

    # De-fragment pbp to remove the pandas warning
    pbp = pbp.copy()
    pbp['turnover'] = pbp['interception'].fillna(0) + pbp['fumble_lost'].fillna(0)

    # --- TEAM METRICS ---
    game_off = pbp.groupby(['game_id', 'posteam']).agg(
        off_epa=('epa', 'mean'),
        turnovers_lost=('turnover', 'sum')
    ).reset_index().rename(columns={'posteam': 'team'})

    game_def = pbp.groupby(['game_id', 'defteam']).agg(
        def_epa=('epa', 'mean'),
        turnovers_forced=('turnover', 'sum')
    ).reset_index().rename(columns={'defteam': 'team'})
    
    game_stats = pd.merge(game_off, game_def, on=['game_id', 'team'], how='outer')
    game_stats['net_turnovers'] = game_stats['turnovers_forced'] - game_stats['turnovers_lost']

    # --- QB METRICS ---
    qb_stats = pbp[(pbp['play_type'].isin(['pass', 'run']))]
    team_qb_game = qb_stats.groupby(['game_id', 'posteam']).agg(
        qb_epa=('epa', 'mean'),
        qb_cpoe=('cpoe', 'mean')
    ).reset_index().rename(columns={'posteam': 'team'})
    
    game_stats = pd.merge(game_stats, team_qb_game, on=['game_id', 'team'], how='left')

    # Reshape games to chronologically track teams
    team_games = games[['game_id', 'gameday', 'home_team', 'away_team']].melt(
        id_vars=['game_id', 'gameday'], 
        value_vars=['home_team', 'away_team'], 
        var_name='is_home', 
        value_name='team'
    )
    
    team_stats = pd.merge(team_games, game_stats, on=['game_id', 'team'], how='left')
    team_stats = team_stats.sort_values(by=['team', 'gameday'])

    # --- THE FIX: Forward-fill projected rolling stats for future games ---
    metrics = ['off_epa', 'def_epa', 'net_turnovers', 'qb_epa', 'qb_cpoe']
    
    def calculate_projected_rolling(x):
        # 1. Calculate rolling mean only on completed games
        valid_means = x.dropna().rolling(window=4, min_periods=1).mean()
        # 2. Align back to the full schedule, shift by 1, and forward-fill for future games
        return valid_means.reindex(x.index).shift(1).ffill()

    for col in metrics:
        team_stats[f'rolling_{col}'] = team_stats.groupby('team')[col].transform(calculate_projected_rolling)

    # Split back to home and away
    home_stats = team_stats[team_stats['is_home'] == 'home_team'].drop(columns=['gameday', 'is_home', 'team'])
    home_stats = home_stats.rename(columns=lambda x: f"home_{x}" if x != 'game_id' else x)

    away_stats = team_stats[team_stats['is_home'] == 'away_team'].drop(columns=['gameday', 'is_home', 'team'])
    away_stats = away_stats.rename(columns=lambda x: f"away_{x}" if x != 'game_id' else x)

    games = pd.merge(games, home_stats, on='game_id', how='left')
    games = pd.merge(games, away_stats, on='game_id', how='left')

    # Fill NAs in QB stats for rookies/early season
    games.fillna({'home_rolling_qb_epa': 0, 'away_rolling_qb_epa': 0, 
                  'home_rolling_qb_cpoe': 0, 'away_rolling_qb_cpoe': 0}, inplace=True)

    games['diff_rolling_off_epa'] = games['home_rolling_off_epa'] - games['away_rolling_off_epa']
    games['diff_rolling_def_epa'] = games['home_rolling_def_epa'] - games['away_rolling_def_epa']
    games['diff_rolling_turnovers'] = games['home_rolling_net_turnovers'] - games['away_rolling_net_turnovers']
    games['diff_rolling_qb_epa'] = games['home_rolling_qb_epa'] - games['away_rolling_qb_epa']
    games['diff_rolling_qb_cpoe'] = games['home_rolling_qb_cpoe'] - games['away_rolling_qb_cpoe']

    # Drop games where the rolling averages haven't populated yet (Weeks 1-3 of 2023)
    final_df = games.dropna(subset=['diff_rolling_off_epa', 'diff_rolling_def_epa']).copy()
    
    # Save 
    out_dir = '../data/processed' if os.path.basename(os.getcwd()) == 'src' else 'data/processed'
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'model_ready_data.csv')
    final_df.to_csv(out_path, index=False)
    print(f"Data processed and saved to {out_path}")

if __name__ == "__main__":
    create_training_data()