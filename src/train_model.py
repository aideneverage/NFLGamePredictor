import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, brier_score_loss
import joblib
import os

def train_and_evaluate():
    df = pd.read_csv('../data/processed/model_ready_data.csv')

    features = ['diff_rolling_off_epa']
    target = 'home_win'

    #chronological split: train on 2023 & 2024, test on 2025
    train_df = df[df['season'] < 2025]
    test_df = df[df['season'] == 2025]

    X_train, y_train = train_df[features], train_df[target]
    X_test, y_test = test_df[features], test_df[target]

    print(f"Training samples (2023-2024): {len(X_train)}")
    print(f"Testing samples (2025): {len(X_test)}")

    model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42)
    model.fit(X_train, y_train)

    #evaluate
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    print(f"2025 Test Accuracy: {accuracy_score(y_test, preds):.3f}")
    print(f"2025 Brier Score: {brier_score_loss(y_test, probs):.3f}")

    #save model
    os.makedirs('../models', exist_ok=True)
    joblib.dump(model, '../models/nfl_gb_model.pkl')
    print("Model saved to models/nfl_gb_model.pkl")

if __name__ == "__main__":
    train_and_evaluate()

