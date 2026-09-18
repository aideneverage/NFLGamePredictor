import pandas as pd
import numpy as np
import nfl_data_py as nfl
import os

def create_training_data(seasons=[2023, 2024, 2025, 2026]):
    print(f"Fetching schedules for {seasons}...")
    games = nfl.import_schedules(seasons)

    print("Fetching play-by-play data year-by-year...")
    pbp_frames = []
    for s in seasons:
        try:
            # We catch the error ourselves to bypass the nfl_data_py bug
            pbp_frames.append(nfl.import_pbp_data([s]))
            print(f"{s} play-by-play loaded.")
        except Exception as e:
            print(f"⚠️ {s} play-by-play not available yet (HTTP 404). Skipping...")
            
    if not pbp_frames:
        raise ValueError("No play-by-play data could be fetched.")
        
    pbp = pd.concat(pbp_frames, ignore_index=True)

    # Filter to regular season games (keep both completed and upcoming)
    games = games[games['game_type'] == 'REG'].copy()
    
    # Target: 1 for Home Win, 0 for Away Win, NaN for upcoming or ties
    games['home_win'] = np.where(
        games['result'].isna() | (games['result'] == 0),
        np.nan,
        (games['result'] > 0).astype(float)
    )

    # Calculate Turnovers
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

    team_games = games[['game_id', 'gameday', 'home_team', 'away_team', 'season', 'week']].melt(
        id_vars=['game_id', 'gameday', 'season', 'week'], 
        value_vars=['home_team', 'away_team'], 
        var_name='is_home', 
        value_name='team'
    )
    
    team_stats = pd.merge(team_games, game_stats, on=['game_id', 'team'], how='left')
    team_stats = team_stats.sort_values(by=['team', 'gameday'])

    # Rolling 4-game window; ffill() carries latest form to upcoming games seamlessly
    for col in ['off_epa', 'def_epa', 'net_turnovers']:
        team_stats[f'rolling_{col}'] = team_stats.groupby('team')[col].transform(
            lambda x: x.rolling(window=4, min_periods=1).mean().shift(1)
        )
        team_stats[f'rolling_{col}'] = team_stats.groupby('team')[f'rolling_{col}'].ffill()

    # --- QB METRICS ---
    passes = pbp[(pbp['play_type'] == 'pass') & (pbp['passer_player_name'].notna())].copy()
    qb_game_stats = passes.groupby(['game_id', 'posteam', 'passer_player_name']).agg(
        pass_attempts=('play_id', 'count'),
        qb_epa=('epa', 'mean'),
        qb_cpoe=('cpoe', 'mean')
    ).reset_index().rename(columns={'posteam': 'team'})

    starters = qb_game_stats.loc[qb_game_stats.groupby(['game_id', 'team'])['pass_attempts'].idxmax()].copy()
    starters = pd.merge(starters, games[['game_id', 'gameday']], on='game_id', how='left')
    starters = starters.sort_values(by=['passer_player_name', 'gameday'])

    starters['rolling_qb_epa'] = starters.groupby('passer_player_name')['qb_epa'].transform(
        lambda x: x.rolling(window=4, min_periods=1).mean().shift(1)
    )
    starters['rolling_qb_cpoe'] = starters.groupby('passer_player_name')['qb_cpoe'].transform(
        lambda x: x.rolling(window=4, min_periods=1).mean().shift(1)
    )

    # Fill rookie/backup blanks with league median 0
    starters['rolling_qb_epa'] = starters['rolling_qb_epa'].fillna(0)
    starters['rolling_qb_cpoe'] = starters['rolling_qb_cpoe'].fillna(0)

    # Map the last known starting QB for every team to cover upcoming games
    last_qb_per_team = starters.groupby('team').last().reset_index()
    qb_map = last_qb_per_team.set_index('team')['passer_player_name'].to_dict()
    qb_epa_map = last_qb_per_team.set_index('team')['rolling_qb_epa'].to_dict()
    qb_cpoe_map = last_qb_per_team.set_index('team')['rolling_qb_cpoe'].to_dict()

    games = games.drop(columns=['home_qb_name', 'away_qb_name'], errors='ignore')

    home_team_stats = team_stats[team_stats['is_home'] == 'home_team'][['game_id', 'rolling_off_epa', 'rolling_def_epa', 'rolling_net_turnovers']]
    away_team_stats = team_stats[team_stats['is_home'] == 'away_team'][['game_id', 'rolling_off_epa', 'rolling_def_epa', 'rolling_net_turnovers']]

    home_qbs = starters.rename(columns={
        'team': 'home_team', 
        'passer_player_name': 'home_qb_name', 
        'rolling_qb_epa': 'home_rolling_qb_epa', 
        'rolling_qb_cpoe': 'home_rolling_qb_cpoe'
    })[['game_id', 'home_team', 'home_qb_name', 'home_rolling_qb_epa', 'home_rolling_qb_cpoe']]

    away_qbs = starters.rename(columns={
        'team': 'away_team', 
        'passer_player_name': 'away_qb_name', 
        'rolling_qb_epa': 'away_rolling_qb_epa', 
        'rolling_qb_cpoe': 'away_rolling_qb_cpoe'
    })[['game_id', 'away_team', 'away_qb_name', 'away_rolling_qb_epa', 'away_rolling_qb_cpoe']]

    games = pd.merge(games, home_team_stats.rename(columns=lambda x: f"home_{x}" if x != 'game_id' else x), on='game_id', how='left')
    games = pd.merge(games, away_team_stats.rename(columns=lambda x: f"away_{x}" if x != 'game_id' else x), on='game_id', how='left')
    games = pd.merge(games, home_qbs, on=['game_id', 'home_team'], how='left')
    games = pd.merge(games, away_qbs, on=['game_id', 'away_team'], how='left')

    # Fill upcoming blank games with the last known starter for that team
    games['home_qb_name'] = games['home_qb_name'].fillna(games['home_team'].map(qb_map))
    games['home_rolling_qb_epa'] = games['home_rolling_qb_epa'].fillna(games['home_team'].map(qb_epa_map)).fillna(0)
    games['home_rolling_qb_cpoe'] = games['home_rolling_qb_cpoe'].fillna(games['home_team'].map(qb_cpoe_map)).fillna(0)

    games['away_qb_name'] = games['away_qb_name'].fillna(games['away_team'].map(qb_map))
    games['away_rolling_qb_epa'] = games['away_rolling_qb_epa'].fillna(games['away_team'].map(qb_epa_map)).fillna(0)
    games['away_rolling_qb_cpoe'] = games['away_rolling_qb_cpoe'].fillna(games['away_team'].map(qb_cpoe_map)).fillna(0)

    # Compute Differentials
    games['diff_rolling_off_epa'] = games['home_rolling_off_epa'] - games['away_rolling_off_epa']
    games['diff_rolling_def_epa'] = games['home_rolling_def_epa'] - games['away_rolling_def_epa']
    games['diff_rolling_turnovers'] = games['home_rolling_net_turnovers'] - games['away_rolling_net_turnovers']
    games['diff_rolling_qb_epa'] = games['home_rolling_qb_epa'] - games['away_rolling_qb_epa']
    games['diff_rolling_qb_cpoe'] = games['home_rolling_qb_cpoe'] - games['away_rolling_qb_cpoe']

    features = ['diff_rolling_off_epa', 'diff_rolling_def_epa', 'diff_rolling_turnovers', 'diff_rolling_qb_epa', 'diff_rolling_qb_cpoe']
    final_df = games.dropna(subset=features).copy()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(base_dir, 'data', 'processed')
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, 'model_ready_data.csv')
    final_df.to_csv(output_file, index=False)
    print(f"Data processed and saved to {output_file}")

if __name__ == "__main__":
    create_training_data()