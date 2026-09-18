import pandas as pd
import joblib
import shap
import os

def load_resources():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'data', 'processed', 'model_ready_data.csv')
    model_path = os.path.join(base_dir, 'models', 'nfl_gb_model.pkl')

    df = pd.read_csv(data_path)
    model = joblib.load(model_path)
    return df, model

def run_interactive_predictor():
    df, model = load_resources()
    test_df = df[df['season'] == 2025].copy()
    
    features = [
        'diff_rolling_off_epa', 'diff_rolling_def_epa', 'diff_rolling_turnovers', 
        'diff_rolling_qb_epa', 'diff_rolling_qb_cpoe'
    ]

    print("\n" + "="*50)
    print("        NFL GAME OUTCOME PREDICTOR (2025)")
    print("="*50)

    available_weeks = sorted(test_df['week'].unique())
    print(f"Available Weeks: {', '.join(str(w) for w in available_weeks)}")
    
    while True:
        try:
            selected_week = int(input("\nEnter Week number to inspect: "))
            if selected_week in available_weeks:
                break
            print(f"Please pick a week from: {available_weeks}")
        except ValueError:
            print("Invalid input. Enter an integer.")

    week_games = test_df[test_df['week'] == selected_week].reset_index(drop=True)
    
    print(f"\n--- Games for Week {selected_week} ---")
    print(f"{'#':<3} | {'Away Team':<15} @ {'Home Team':<15} | {'Status'}")
    print("-" * 55)
    for idx, row in week_games.iterrows():
        status = "Played" if pd.notna(row['home_win']) else "Upcoming"
        away_str = f"{row['away_team']} ({row['away_qb_name']})"
        home_str = f"{row['home_team']} ({row['home_qb_name']})"
        print(f"[{idx + 1:<2}] | {away_str:<15} @ {home_str:<15} | {status}")

    while True:
        try:
            choice = int(input(f"\nSelect game number (1-{len(week_games)}): "))
            if 1 <= choice <= len(week_games):
                selected_idx = choice - 1
                break
            print(f"Please enter a number between 1 and {len(week_games)}.")
        except ValueError:
            print("Invalid input.")

    game = week_games.iloc[selected_idx]
    x_game = week_games.loc[[selected_idx], features]

    home_prob = model.predict_proba(x_game)[0][1]
    away_prob = 1.0 - home_prob

    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(x_game)
    impact = shap_vals[1][0] if isinstance(shap_vals, list) else shap_vals[0]

    print("\n" + "="*55)
    print(f"  PREDICTION: {game['away_team']} ({game['away_qb_name']}) @ {game['home_team']} ({game['home_qb_name']})")
    print("="*55)
    print(f"  {game['home_team']:<4} Win Probability -> {home_prob:.1%}")
    print(f"  {game['away_team']:<4} Win Probability -> {away_prob:.1%}")
    
    favored = game['home_team'] if home_prob >= 0.5 else game['away_team']
    est_spread = abs(home_prob - 0.5) * 28
    print(f"  Favored : {favored} (Estimated Spread: -{est_spread:.1f})")
    
    if pd.notna(game['home_win']):
        actual = game['home_team'] if game['home_win'] == 1 else game['away_team']
        print(f"  Outcome : {actual} Won")
    print("-" * 55)

    print("Factor Contributions (SHAP):")
    for feat, val, imp in zip(features, x_game.iloc[0], impact):
        direction = f"favored {game['home_team']}" if imp > 0 else f"favored {game['away_team']}"
        print(f"  • {feat} = {val:+.4f} (Shift: {imp:+.4f}, {direction})")
    print("="*55 + "\n")

if __name__ == "__main__":
    run_interactive_predictor()
