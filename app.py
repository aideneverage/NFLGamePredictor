import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="NFL Game Predictor",
    page_icon="🏈",
    layout="wide"
)

@st.cache_resource
def load_artifacts():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, 'data', 'processed', 'model_ready_data.csv')
    model_path = os.path.join(base_dir, 'models', 'nfl_gb_model.pkl')

    if not os.path.exists(data_path) or not os.path.exists(model_path):
        st.error("Missing model or data. Please run data_pipeline.py and train_model.py first.")
        st.stop()

    df = pd.read_csv(data_path)
    model = joblib.load(model_path)
    explainer = shap.TreeExplainer(model)
    return df, model, explainer

df, model, explainer = load_artifacts()

feature_cols = [
    'diff_rolling_off_epa', 
    'diff_rolling_def_epa', 
    'diff_rolling_turnovers', 
    'diff_rolling_qb_epa', 
    'diff_rolling_qb_cpoe'
]

friendly_names = {
    'diff_rolling_off_epa': 'Offensive EPA Diff',
    'diff_rolling_def_epa': 'Defensive EPA Allowed Diff',
    'diff_rolling_turnovers': 'Net Turnover Margin Diff',
    'diff_rolling_qb_epa': 'Starting QB EPA Diff',
    'diff_rolling_qb_cpoe': 'Starting QB CPOE Diff'
}

# --- Sidebar Controls ---
st.sidebar.title("🏈 NFL Game Predictor")
st.sidebar.markdown("Built by **[Aiden Everage](https://github.com/aideneverage)**")
st.sidebar.markdown("---")

# Season Selector (all seasons descending: 2026, 2025, 2024, 2023)
available_seasons = sorted([int(s) for s in df['season'].unique()], reverse=True)
selected_season = st.sidebar.selectbox("Season", available_seasons, index=0)

season_df = df[df['season'] == selected_season].copy()
available_weeks = sorted([int(w) for w in season_df['week'].unique()])

if not available_weeks:
    st.warning("No games found for this season.")
    st.stop()
    
selected_week = st.sidebar.selectbox("Week", available_weeks, index=0)

week_games = season_df[season_df['week'] == selected_week].reset_index(drop=True)

if week_games.empty:
    st.warning("No games found for this week.")
    st.stop()

# Clean, simplified matchup list: "AWAY @ HOME" (with a clean indicator if final)
matchup_options = []
for _, row in week_games.iterrows():
    status_suffix = " (Final)" if pd.notna(row['home_win']) else ""
    matchup_options.append(f"{row['away_team']} @ {row['home_team']}{status_suffix}")

selected_matchup_idx = st.sidebar.selectbox(
    "Select Matchup", 
    range(len(matchup_options)), 
    format_func=lambda i: matchup_options[i]
)

game = week_games.iloc[selected_matchup_idx]
x_game = week_games.loc[[selected_matchup_idx], feature_cols]

# Header section
is_completed = pd.notna(game['home_win'])
status_badge = "✅ FINAL" if is_completed else "⏳ UPCOMING"

st.title(f"{game['away_team']} at {game['home_team']}  `{status_badge}`")
gameday_str = f" • Gameday: {game['gameday']}" if pd.notna(game.get('gameday')) else ""
st.caption(f"{selected_season} Regular Season • Week {selected_week}{gameday_str}")

# Predictions
home_prob = model.predict_proba(x_game)[0][1]
away_prob = 1.0 - home_prob
favored_team = game['home_team'] if home_prob >= 0.5 else game['away_team']
est_spread = abs(home_prob - 0.5) * 28

col_left, col_mid, col_right = st.columns([1, 1, 1])

with col_left:
    st.subheader(f"Visiting: {game['away_team']}")
    st.write(f"**Starting QB:** {game.get('away_qb_name', 'TBD')}")
    st.metric("Win Probability", f"{away_prob:.1%}")

with col_mid:
    st.subheader("Matchup Outlook")
    st.metric("Model Favorite", favored_team, delta=f"Spread: -{est_spread:.1f} pts")
    
    if is_completed:
        actual_winner = game['home_team'] if game['home_win'] == 1 else game['away_team']
        is_correct = (actual_winner == favored_team)
        if is_correct:
            st.success(f"**Result:** {actual_winner} Won ✅ (Pick Correct)")
        else:
            st.error(f"**Result:** {actual_winner} Won ❌ (Upset)")
    else:
        st.info("Game has not been played yet. Prediction based on current form.")

with col_right:
    st.subheader(f"Home: {game['home_team']}")
    st.write(f"**Starting QB:** {game.get('home_qb_name', 'TBD')}")
    st.metric("Win Probability", f"{home_prob:.1%}")

st.progress(float(home_prob), text=f"Home Field Win Probability ({home_prob:.1%})")

st.markdown("---")

# --- SHAP Feature Attribution Chart ---
st.subheader("🔍 Prediction Breakdown (SHAP Values)")
shap_vals = explainer.shap_values(x_game)
impacts = shap_vals[1][0] if isinstance(shap_vals, list) else shap_vals[0]

chart_df = pd.DataFrame({
    'Feature': [friendly_names[f] for f in feature_cols],
    'Impact': impacts,
    'Raw Value': [f"{x_game.iloc[0][f]:+.3f}" for f in feature_cols]
}).sort_values(by='Impact', ascending=True)

fig, ax = plt.subplots(figsize=(8, 3.2))
bar_colors = ['#2E7D32' if val > 0 else '#C62828' for val in chart_df['Impact']]

ax.barh(chart_df['Feature'], chart_df['Impact'], color=bar_colors, height=0.5)
ax.axvline(0, color='gray', linestyle='--', linewidth=0.8)
ax.set_xlabel(f"Impact on Log-Odds (← Favors {game['away_team']} | Favors {game['home_team']} →)")
ax.grid(axis='x', linestyle=':', alpha=0.4)

for i, (val, raw) in enumerate(zip(chart_df['Impact'], chart_df['Raw Value'])):
    offset = 0.02 if val >= 0 else -0.02
    ha = 'left' if val >= 0 else 'right'
    ax.text(val + offset, i, f"diff: {raw}", va='center', ha=ha, fontsize=8, color='#333333')

plt.tight_layout()
st.pyplot(fig)

st.markdown("---")

# --- Interactive What-If Sandbox ---
st.subheader("🧪 'What-If' Scenario Sandbox")
st.caption("Adjust inputs to see how in-game shifts or QB changes would recalculate the win probability live.")

s1, s2, s3 = st.columns(3)
with s1:
    qb_epa = st.slider("QB EPA Diff", -0.50, 0.50, float(x_game.iloc[0]['diff_rolling_qb_epa']), 0.02)
with s2:
    to_diff = st.slider("Turnover Margin Diff", -3.0, 3.0, float(x_game.iloc[0]['diff_rolling_turnovers']), 0.25)
with s3:
    cpoe_diff = st.slider("QB CPOE Diff (%)", -20.0, 20.0, float(x_game.iloc[0]['diff_rolling_qb_cpoe']), 0.5)

x_sim = x_game.copy()
x_sim['diff_rolling_qb_epa'] = qb_epa
x_sim['diff_rolling_turnovers'] = to_diff
x_sim['diff_rolling_qb_cpoe'] = cpoe_diff

sim_prob = model.predict_proba(x_sim)[0][1]
st.write(f"**Adjusted Home Win Probability:** `{sim_prob:.1%}` (Shift: `{sim_prob - home_prob:+.1%}`)")