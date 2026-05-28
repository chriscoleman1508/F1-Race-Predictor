import os
import streamlit as st
import fastf1
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Advanced F1 Strategy AI", page_icon="📊", layout="centered")

st.title("📊 Multi-Factor F1 Race Predictor")
st.write("This advanced AI model uses **FastF1 live data** to calculate weights based on car performance averages, driver baseline skills, and qualifying variations.")

if not os.path.exists('f1_cache'):
    os.makedirs('f1_cache')

@st.cache_resource
def train_advanced_model():
    fastf1.Cache.enable_cache('f1_cache')
    historical_data = []
    
    for year in [2024, 2025, 2026]:
        try:
            schedule = fastf1.get_event_schedule(year)
            schedule = schedule[schedule['EventFormat'] != 'testing']
            for _, row in schedule.iterrows():
                try:
                    session = fastf1.get_session(year, row['EventName'], 'R')
                    session.load(laps=False, telemetry=False, weather=False)
                    if session.results is not None and not session.results.empty:
                        historical_data.append(session.results[['Position', 'GridPosition', 'Abbreviation', 'TeamName']])
                except Exception:
                    continue
        except Exception:
            continue
            
    df = pd.concat(historical_data, ignore_index=True)
    df['Position'] = pd.to_numeric(df['Position'], errors='coerce')
    df['GridPosition'] = pd.to_numeric(df['GridPosition'], errors='coerce')
    df = df.dropna().dropna()
    df = df[(df['GridPosition'] > 0) & (df['Position'] > 0)]
    
    # Calculate performance dictionaries
    team_stats_map = df.groupby('TeamName')['Position'].mean().to_dict()
    driver_stats_map = df.groupby('Abbreviation')['Position'].mean().to_dict()
    
    df['Team_Avg_Pos'] = df['TeamName'].map(team_stats_map)
    df['Driver_Avg_Pos'] = df['Abbreviation'].map(driver_stats_map)
    df['Driver_vs_Car_Diff'] = df['Driver_Avg_Pos'] - df['Team_Avg_Pos']
    df['Grid_vs_Car_Diff'] = df['GridPosition'] - df['Team_Avg_Pos']
    df['Did_Win'] = df['Position'].apply(lambda x: 1 if x == 1 else 0)
    
    feature_cols = ['GridPosition', 'Team_Avg_Pos', 'Driver_Avg_Pos', 'Driver_vs_Car_Diff', 'Grid_vs_Car_Diff']
    X = df[feature_cols]
    y = df['Did_Win']
    
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=150, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    
    return model, team_stats_map, driver_stats_map, feature_cols

with st.spinner('Compiling multi-season data and mapping car metrics...'):
    model, team_stats, driver_stats, feature_cols = train_advanced_model()

st.success("Advanced Realism Engine Deployed!")

st.header("🔮 Setup Simulation Matrix")

unique_drivers = sorted(list(driver_stats.keys()))
unique_teams = sorted(list(team_stats.keys()))

input_driver = st.selectbox("Select Driver Abbreviation:", unique_drivers, index=unique_drivers.index('VER') if 'VER' in unique_drivers else 0)
input_team = st.selectbox("Select Team Car Construction:", unique_teams, index=unique_teams.index('Red Bull Racing') if 'Red Bull Racing' in unique_teams else 0)
input_grid = st.slider("Starting Grid Placement:", min_value=1, max_value=20, value=1)

# Display real-time telemetry lookups to the user
t_rating = team_stats[input_team]
d_rating = driver_stats[input_driver]

col1, col2 = st.columns(2)
col1.metric(label=f"🏎️ {input_team} Car Rating (Avg Finish)", value=f"{t_rating:.2f}")
col2.metric(label=f"🏁 {input_driver} Driver Rating (Avg Finish)", value=f"{d_rating:.2f}")

if st.button("Run Simulation", type="primary"):
    # Generate the calculated factors dynamically
    d_v_c = d_rating - t_rating
    g_v_c = input_grid - t_rating
    
    scenario = pd.DataFrame([[input_grid, t_rating, d_rating, d_v_c, g_v_c]], columns=feature_cols)
    
    prediction = model.predict(scenario)[0]
    probabilities = model.predict_proba(scenario)[0]
    
    st.markdown("---")
    st.subheader("📊 AI Multi-Factor Matrix Output")
    
    if prediction == 1:
        st.balloons()
        st.success(f"🏆 **Prediction: WINNER!** The model accounts for the high car performance rating and starting advantage, calculating a **{probabilities[1]*100:.1f}%** win probability.")
    else:
        st.error(f"❌ **Prediction: Loss.** The math confirms that this combination of car capability ({t_rating:.1f}) and starting position doesn't pass the historical threshold for a win. (Confidence: {probabilities[0]*100:.1f}%)")