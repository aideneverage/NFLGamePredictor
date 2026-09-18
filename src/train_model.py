import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, brier_score_loss
import joblib
import os

def train_and_evaluate():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'data', 'processed', 'model_ready_data.csv')
    model_dir = os.path.join(base_dir, 'models')
    model_path = os.path.join(model_dir, 'nfl_gb_model.pkl')

    df = pd.read_csv(data_path)

    features = [
        'diff_rolling_off_epa', 
        'diff_rolling_def_epa', 
        'diff_rolling_turnovers',
        'diff_rolling_qb_epa',
        'diff_rolling_qb_cpoe'
    ]
    target = 'home_win'

    # Filter only completed games for training & testing
    completed_games = df[df[target].notna()].copy()

    # Train on 2023 & 2024; test benchmark on 2025
    train_df = completed_games[completed_games['season'] < 2025]
    test_df = completed_games[completed_games['season'] == 2025]

    X_train, y_train = train_df[features], train_df[target]
    X_test, y_test = test_df[features], test_df[target]

    print(f"Training samples (2023-2024): {len(X_train)}")
    print(f"Testing samples (2025): {len(X_test)}")

    model = GradientBoostingClassifier(n_estimators=150, learning_rate=0.05, max_depth=3, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    print(f"2025 Test Accuracy: {accuracy_score(y_test, preds):.3f}")
    print(f"2025 Brier Score: {brier_score_loss(y_test, probs):.3f}")

    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    train_and_evaluate()