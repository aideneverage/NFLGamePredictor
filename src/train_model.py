import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, brier_score_loss
import joblib
import os

def train_and_evaluate():
    data_path = '../data/processed/model_ready_data.csv' if os.path.basename(os.getcwd()) == 'src' else 'data/processed/model_ready_data.csv'
    df = pd.read_csv(data_path)

    features = [
        'diff_rolling_off_epa', 
        'diff_rolling_def_epa', 
        'diff_rolling_turnovers',
        'diff_rolling_qb_epa',
        'diff_rolling_qb_cpoe'
    ]
    target = 'home_win'

    # CRITICAL: Only train/test on games that have actually happened!
    df_played = df[df[target].notna()].copy()

    train_df = df_played[df_played['season'] < 2026]
    test_df = df_played[df_played['season'] == 2026]

    X_train, y_train = train_df[features], train_df[target]
    X_test, y_test = test_df[features], test_df[target]

    print(f"Training samples (2023-2025): {len(X_train)}")
    print(f"Testing samples (2026): {len(X_test)}")

    model = GradientBoostingClassifier(
        n_estimators=150, 
        learning_rate=0.05, 
        max_depth=4, 
        random_state=42
    )
    model.fit(X_train, y_train)

    # Evaluate accuracy only on completed 2026 games
    if len(X_test) > 0:
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]
        print(f"2026 Test Accuracy: {accuracy_score(y_test, preds):.3f}")
        print(f"2026 Brier Score: {brier_score_loss(y_test, probs):.3f}")
    else:
        print("No completed 2026 games yet to evaluate accuracy.")

    model_dir = '../models' if os.path.basename(os.getcwd()) == 'src' else 'models'
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'nfl_gb_model.pkl')
    
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    train_and_evaluate()