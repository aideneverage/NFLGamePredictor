import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="NFL Outcome Predictor",
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

# Helper to filter QBs by team history
def get_team_qbs(team_abbr):
    home_qbs = df[df['home_team'] == team_abbr]['home_qb_name'].dropna()
    away_qbs = df[df['away_team'] == team_abbr]['away_qb_name'].dropna()
    team_qbs = sorted(pd.concat([home_qbs, away_qbs]).unique())
    return team_qbs if len(team_qbs) > 0 else ["Unknown"]

# --- Sidebar Controls ---
st.sidebar.title("NFL Outcome Predictor")
st.sidebar.markdown("Built by **[Aiden Everage](https://github.com/chingu-dev)**")
st.sidebar.markdown("---")

app_mode = st.sidebar.radio(
    "Select Mode",
    ["📅 Past Schedule Explorer", "🧪 Custom Game Simulator"]
)

st.sidebar.markdown("---")

if app_mode == "📅 Past Schedule Explorer":
    
    # Only allow seasons up to 2025 since 2026 has no data yet
    available_seasons = sorted([int(s) for s in df['season'].unique() if 2023 <= s <= 2025], reverse=True)
    selected_season = st.sidebar.selectbox("Season", available_seasons, index=0)

    season_df = df[df['season'] == selected_season].copy()
    available_weeks = sorted([int(w) for w in season_df['week'].unique()])
    selected_week = st.sidebar.selectbox("Week", available_weeks, index=0)

    week_games = season_df[season_df['week'] == selected_week].reset_index(drop=True)

    if week_games.empty:
        st.warning("No games found for this week.")
        st.stop()

    matchup_options = []
    for idx, row in week_games.iterrows():
        status = "Final" if pd.notna(row['home_win']) else "Upcoming"
        away = f"{row['away_team']} ({row.get('away_qb_name', 'TBD')})"
        home = f"{row['home_team']} ({row.get('home_qb_name', 'TBD')})"
        matchup_options.append(f"[{status}] {away} @ {home}")

    selected_matchup_idx = st.sidebar.selectbox(
        "Game", 
        range(len(matchup_options)), 
        format_func=lambda i: matchup_options[i]
    )

    game = week_games.iloc[selected_matchup_idx]
    x_game = week_games.loc[[selected_matchup_idx], feature_cols]

    is_completed = pd.notna(game['home_win'])
    status_badge = "✅ FINAL" if is_completed else "⏳ UPCOMING"

    st.title(f"🏈 {game['away_team']} at {game['home_team']}  `{status_badge}`")
    st.caption(f"{selected_season} Regular Season • Week {selected_week} • Gameday: {game.get('gameday', 'TBD')}")

else:
    teams = sorted(df['home_team'].dropna().unique())

    st.sidebar.subheader("Away Team")
    away_team_input = st.sidebar.selectbox("Select Away Team", teams, index=teams.index("KC") if "KC" in teams else 0)
    # Dynamically filter QBs to only those who have played for the Away Team
    away_team_qbs = get_team_qbs(away_team_input)
    away_qb_input = st.sidebar.selectbox("Select Away QB", away_team_qbs, index=0, key='away_qb')

    st.sidebar.subheader("Home Team")
    home_team_input = st.sidebar.selectbox("Select Home Team", teams, index=teams.index("BAL") if "BAL" in teams else 1)
    # Dynamically filter QBs to only those who have played for the Home Team
    home_team_qbs = get_team_qbs(home_team_input)
    home_qb_input = st.sidebar.selectbox("Select Home QB", home_team_qbs, index=0, key='home_qb')

    # Look up the most recent team-level rolling stats for both teams
    latest_away_team_stats = df[(df['home_team'] == away_team_input) | (df['away_team'] == away_team_input)].sort_values('gameday').iloc[-1]
    away_off_epa = latest_away_team_stats['home_rolling_off_epa'] if latest_away_team_stats['home_team'] == away_team_input else latest_away_team_stats['away_rolling_off_epa']
    away_def_epa = latest_away_team_stats['home_rolling_def_epa'] if latest_away_team_stats['home_team'] == away_team_input else latest_away_team_stats['away_rolling_def_epa']
    away_turnovers = latest_away_team_stats['home_rolling_net_turnovers'] if latest_away_team_stats['home_team'] == away_team_input else latest_away_team_stats['away_rolling_net_turnovers']

    latest_home_team_stats = df[(df['home_team'] == home_team_input) | (df['away_team'] == home_team_input)].sort_values('gameday').iloc[-1]
    home_off_epa = latest_home_team_stats['home_rolling_off_epa'] if latest_home_team_stats['home_team'] == home_team_input else latest_home_team_stats['away_rolling_off_epa']
    home_def_epa = latest_home_team_stats['home_rolling_def_epa'] if latest_home_team_stats['home_team'] == home_team_input else latest_home_team_stats['away_rolling_def_epa']
    home_turnovers = latest_home_team_stats['home_rolling_net_turnovers'] if latest_home_team_stats['home_team'] == home_team_input else latest_home_team_stats['away_rolling_net_turnovers']

    # Look up the most recent QB-level rolling stats for both QBs
    try:
        latest_away_qb_stats = df[(df['home_qb_name'] == away_qb_input) | (df['away_qb_name'] == away_qb_input)].sort_values('gameday').iloc[-1]
        away_qb_epa = latest_away_qb_stats['home_rolling_qb_epa'] if latest_away_qb_stats['home_qb_name'] == away_qb_input else latest_away_qb_stats['away_rolling_qb_epa']
        away_qb_cpoe = latest_away_qb_stats['home_rolling_qb_cpoe'] if latest_away_qb_stats['home_qb_name'] == away_qb_input else latest_away_qb_stats['away_rolling_qb_cpoe']
    except IndexError:
        away_qb_epa, away_qb_cpoe = 0.0, 0.0 

    try:
        latest_home_qb_stats = df[(df['home_qb_name'] == home_qb_input) | (df['away_qb_name'] == home_qb_input)].sort_values('gameday').iloc[-1]
        home_qb_epa = latest_home_qb_stats['home_rolling_qb_epa'] if latest_home_qb_stats['home_qb_name'] == home_qb_input else latest_home_qb_stats['away_rolling_qb_epa']
        home_qb_cpoe = latest_home_qb_stats['home_rolling_qb_cpoe'] if latest_home_qb_stats['home_qb_name'] == home_qb_input else latest_home_qb_stats['away_rolling_qb_cpoe']
    except IndexError:
        home_qb_epa, home_qb_cpoe = 0.0, 0.0 

    # Construct the synthetic game row
    synthetic_row = pd.DataFrame([{
        'diff_rolling_off_epa': home_off_epa - away_off_epa,
        'diff_rolling_def_epa': home_def_epa - away_def_epa,
        'diff_rolling_turnovers': home_turnovers - away_turnovers,
        'diff_rolling_qb_epa': home_qb_epa - away_qb_epa,
        'diff_rolling_qb_cpoe': home_qb_cpoe - away_qb_cpoe
    }])
    
    # Overwrite the variables for rendering
    game = {
        'away_team': away_team_input,
        'home_team': home_team_input,
        'away_qb_name': away_qb_input,
        'home_qb_name': home_qb_input,
        'home_win': np.nan 
    }
    x_game = synthetic_row

    st.title(f"🧪 Custom Game: {game['away_team']} at {game['home_team']}")
    st.caption("Hypothetical game based on the most recent form of both teams and selected QBs.")

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
    
    if app_mode == "📅 Past Schedule Explorer" and pd.notna(game['home_win']):
        actual_winner = game['home_team'] if game['home_win'] == 1 else game['away_team']
        is_correct = (actual_winner == favored_team)
        if is_correct:
            st.success(f"**Result:** {actual_winner} Won ✅ (Pick Correct)")
        else:
            st.error(f"**Result:** {actual_winner} Won ❌ (Upset)")
    elif app_mode == "📅 Past Schedule Explorer":
        st.info("Game has not been played yet. Prediction is based on current form.")

with col_right:
    st.subheader(f"Home: {game['home_team']}")
    st.write(f"**Starting QB:** {game.get('home_qb_name', 'TBD')}")
    st.metric("Win Probability", f"{home_prob:.1%}")

st.progress(float(home_prob), text=f"Home Field Win Probability ({home_prob:.1%})")

st.markdown("---")

# --- SHAP Chart ---
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

# --- What-If Sandbox ---
st.subheader("🧪 Custom Game Simulation Sandbox")
st.caption("Test scenarios (e.g. QB changes or turnover surges) to recalculate win probabilities live.")

s1, s2, s3 = st.columns(3)
with s1:
    qb_epa = st.slider("QB EPA Differential", -0.50, 0.50, float(x_game.iloc[0]['diff_rolling_qb_epa']), 0.02)
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