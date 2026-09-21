# NFL Game Outcome Predictor

**Live Dashboard:** [https://nflaiprediction.streamlit.app/](https://nflaiprediction.streamlit.app/)

A machine learning pipeline and interactive web dashboard that predicts NFL game outcomes using historical play-by-play data, rolling team efficiencies, and quarterback metrics. 

This project goes beyond simple win/loss classification by returning well-calibrated **win probabilities**, estimating equivalent **point spreads**, and using **SHAP** (SHapley Additive exPlanations) to explicitly quantify *why* the model made its prediction.

---

## Features

- **Live Streamlit Dashboard:** Browse the 2025 schedule, view upcoming game probabilities, and review past game accuracy.
- **Custom "What-If" Simulator:** Manually pair any two teams and starting quarterbacks to simulate a hypothetical matchup on the fly.
- **Dynamic Feature Attribution (SHAP):** Visualizes exactly how much the starting Quarterback, Turnover Margin, or Defensive efficiency shifted the odds for a specific game.
- **No Target Leakage:** Features are strictly engineered using rolling averages that are shifted chronologically. The model only knows what was available *before kickoff*.

---

## The Machine Learning Model

The predictor is built using a **Gradient Boosting Classifier** (`scikit-learn`) trained on play-by-play data from the `nflverse` (via `nfl_data_py`) spanning the 2023–2025 seasons.

### Feature Engineering
Instead of using basic box-score stats, the model calculates 4-game rolling differentials for the following advanced metrics:
1. **Offensive EPA/Play (Expected Points Added):** The team's rolling offensive efficiency.
2. **Defensive EPA/Play Allowed:** The team's rolling defensive efficiency.
3. **Net Turnover Margin:** Turnovers forced minus turnovers lost.
4. **Starting QB EPA/Play:** The starting Quarterback's individual rolling efficiency. 
5. **Starting QB CPOE:** The starting Quarterback's Completion Percentage Over Expected.

*Note: Quarterback metrics are calculated per-player, not per-team. This allows the model to immediately adjust win probabilities when a backup quarterback is forced to start.*

---

## Running the Project Locally

If you want to clone this repository and run the data pipeline or dashboard on your own machine:

### 1. Installation
Ensure you have Python 3.9+ installed. Clone the repo and install the required dependencies:

```bash
git clone [https://github.com/aideneverage/nfl-predictor.git](https://github.com/aideneverage/nfl-predictor.git)
cd nfl-predictor
pip install -r requirements.txt
```

### 2. Generate the Data & Train the Model
The repository does not store the raw play-by-play data due to file size constraints. You must generate the features and train the .pkl file locally:

```bash
# Downloads nflverse data and calculates rolling features
python src/data_pipeline.py

# Trains the Gradient Boosting model and exports nfl_gb_model.pkl
python src/train_model.py
```

### 3. Launch the Dashboard
Run the Streamlit app locally:

```bash
python -m streamlit run app.py
```

## Author

Created by Aiden Everage.
