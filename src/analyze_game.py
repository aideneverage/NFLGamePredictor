import pandas as pd
import joblib
import shap
import os

def load_resources():
    data_path = '../data/processed/model_ready_data.csv'
    model_path = '../models/nfl_gb_model.pkl'

    df = pd.read_csv(data_path)
    model = joblib.load(model_path)
    return df, model

def run_interactive_predictor():
    df, model = load_resources()

    #isolate 2025 test season
    test_df = df[df['season'] == 2025].copy()
    features = ['diff_rolling_off_epa', 'diff_rolling_def_epa', 'diff_rolling_turnovers']

    print("\n" + "="*50)
    print("        NFL GAME OUTCOME PREDICTOR (2025)")
    print("="*50)

    # 1. Week selection
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

    # 2. Filter and display games for that week
    week_games = test_df[test_df['week'] == selected_week].reset_index(drop=True)
    
    print(f"\n--- Games for Week {selected_week} ---")
    print(f"{'#':<4} | {'Away Team':<10} @ {'Home Team':<10} | {'Status'}")
    print("-" * 45)
    for idx, row in week_games.iterrows():
        status = "Played" if pd.notna(row['home_win']) else "Upcoming"
        print(f"[{idx + 1:<2}] | {row['away_team']:<10} @ {row['home_team']:<10} | {status}")

    # 3. Game selection
    while True:
        try:
            choice = int(input(f"\nSelect game number (1-{len(week_games)}): "))
            if 1 <= choice <= len(week_games):
                selected_idx = choice - 1
                break
            print(f"Please enter a number between 1 and {len(week_games)}.")
        except ValueError:
            print("Invalid input. Enter an integer.")

    # 4. Predict & Explain
    game = week_games.iloc[selected_idx]
    x_game = week_games.loc[[selected_idx], features]

    home_prob = model.predict_proba(x_game)[0][1]
    away_prob = 1.0 - home_prob

    # SHAP Explanations
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(x_game)
    impact = shap_vals[1][0] if isinstance(shap_vals, list) else shap_vals[0]

    # display breakdown card
    print("\n" + "="*50)
    print(f"  PREDICTION: {game['away_team']} @ {game['home_team']}")
    print("="*50)
    print(f"  Home Team: {game['home_team']:<4} Win Probability -> {home_prob:.1%}")
    print(f"  Away Team: {game['away_team']:<4} Win Probability -> {away_prob:.1%}")
    
    favored = game['home_team'] if home_prob >= 0.5 else game['away_team']
    est_spread = abs(home_prob - 0.5) * 28
    print(f"  Favored  : {favored} by {est_spread:.1f} pts spread margin estimate")
    
    if pd.notna(game['home_win']):
        actual = game['home_team'] if game['home_win'] == 1 else game['away_team']
        print(f"  Outcome  : {actual} Won")
    print("-" * 50)

    print("Factor Contributions (SHAP):")
    for feat, val, imp in zip(features, x_game.iloc[0], impact):
        direction = "favored Home" if imp > 0 else "favored Away"
        print(f"  • {feat} = {val:+.4f} (Shifted odds: {imp:+.4f}, {direction})")
    print("="*50 + "\n")

if __name__ == "__main__":
    run_interactive_predictor()
