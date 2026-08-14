import streamlit as st
import pandas as pd
import xgboost as xgb

# 1. Page Configuration
st.set_page_config(page_title="Live Win Probability", page_icon="🏀", layout="centered")

# 2. Load the Model
# We use st.cache_resource so the model only loads once, keeping the app lightning fast
@st.cache_resource
def load_model():
    model = xgb.XGBClassifier()
    model.load_model('ncaa_xgb_model.json')
    return model

model = load_model()

# 3. Application Header
st.title("🏀 NCAA Live Win Probability Predictor")
st.markdown("Enter the current game state to calculate the home team's exact probability of securing the victory.")
st.divider()

# 4. User Inputs (The Sidebar & Main Layout)
# We add a list of example teams for the dropdowns
ncaa_teams = ["Duke", "North Carolina", "Kansas", "Kentucky", "UConn", "Purdue", "Houston", "Villanova"]

col1, col2 = st.columns(2)

with col1:
    home_team = st.selectbox("Home Team", options=ncaa_teams, index=0)
    home_score = st.number_input("Home Team Score", min_value=0, max_value=150, value=75)
    period = st.selectbox("Current Period", options=[1, 2, 3, 4], index=1)

with col2:
    away_team = st.selectbox("Away Team", options=ncaa_teams, index=1)
    away_score = st.number_input("Away Team Score", min_value=0, max_value=150, value=70)
    minutes_remaining = st.slider("Minutes Remaining in Period", min_value=0, max_value=20, value=5)
    seconds_remaining = st.slider("Seconds Remaining", min_value=0, max_value=59, value=0)

# 5. Data Engineering (Replicating our SQL logic)
home_score_differential = home_score - away_score

if period == 1:
    total_seconds_remaining = (20 * 60) + (minutes_remaining * 60) + seconds_remaining
else:
    total_seconds_remaining = (minutes_remaining * 60) + seconds_remaining

# 6. Prediction Logic
if st.button("Calculate Probability", type="primary"):
    
    # Notice we still only pass the numeric variables to the model!
    input_data = pd.DataFrame({
        'period': [period],
        'home_score': [home_score],
        'away_score': [away_score],
        'home_score_differential': [home_score_differential],
        'total_seconds_remaining': [total_seconds_remaining]
    })
    
    win_prob = model.predict_proba(input_data)[0][1]
    
    # 7. Dynamic Output Display
    st.divider()
    
    # We use the selected team names dynamically in the header
    st.subheader(f"Matchup Prediction: {home_team} vs {away_team}")
    
    st.metric(label=f"{home_team} Win Probability", value=f"{(win_prob * 100):.1f}%")
    st.progress(float(win_prob))
    
    if win_prob > 0.5:
        st.success(f"{home_team} is currently favored to win the game.")
    else:
        st.error(f"{away_team} is currently favored to win the game.")