import os
import fastf1
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

print("🏎️ Initializing FastF1 and setting up cache...")
if not os.path.exists('f1_cache'):
    os.makedirs('f1_cache')
fastf1.Cache.enable_cache('f1_cache')

print("📡 Gathering historical race records (2024 - 2026) via FastF1 API...")
historical_data = []

# Fetch data up to the current 2026 season layout
for year in [2024, 2025, 2026]:
    try:
        schedule = fastf1.get_event_schedule(year)
        schedule = schedule[schedule['EventFormat'] != 'testing']
        
        for _, row in schedule.iterrows():
            try:
                session = fastf1.get_session(year, row['EventName'], 'R')
                session.load(laps=False, telemetry=False, weather=False)
                
                if session.results is not None and not session.results.empty:
                    df_res = session.results[['Position', 'GridPosition', 'Abbreviation', 'TeamName']].copy()
                    historical_data.append(df_res)
            except Exception:
                continue
    except Exception:
        continue

# Combine into one master data framework
df = pd.concat(historical_data, ignore_index=True)

print("🧹 Processing & Cleaning Live API Data...")
# Ensure positions and grids are strictly numbers
df['Position'] = pd.to_numeric(df['Position'], errors='coerce')
df['GridPosition'] = pd.to_numeric(df['GridPosition'], errors='coerce')
df = df.dropna(subset=['Position', 'GridPosition', 'Abbreviation', 'TeamName'])
df = df[(df['GridPosition'] > 0) & (df['Position'] > 0)]

print("🧠 Engineering Advanced Realistic AI Factors...")
# Factor 1 & 2: Calculate Global Baseline Averages across historical timelines
team_stats_map = df.groupby('TeamName')['Position'].mean().to_dict()
driver_stats_map = df.groupby('Abbreviation')['Position'].mean().to_dict()

df['Team_Avg_Pos'] = df['TeamName'].map(team_stats_map)
df['Driver_Avg_Pos'] = df['Abbreviation'].map(driver_stats_map)

# Factor 3: Machinery Delta (Does the driver beat their own car's limitations?)
df['Driver_vs_Car_Diff'] = df['Driver_Avg_Pos'] - df['Team_Avg_Pos']

# Factor 4: Grid Position (Directly in df['GridPosition'])

# Factor 5: Qualifying Overperformance Delta
df['Grid_vs_Car_Diff'] = df['GridPosition'] - df['Team_Avg_Pos']

# Target Vector: 1 if Won (P1), 0 otherwise
df['Did_Win'] = df['Position'].apply(lambda x: 1 if x == 1 else 0)

# Define our highly realistic factor checklist
feature_cols = ['GridPosition', 'Team_Avg_Pos', 'Driver_Avg_Pos', 'Driver_vs_Car_Diff', 'Grid_vs_Car_Diff']
X = df[feature_cols]
y = df['Did_Win']

print("🏋️ Training the Realism-Weighted Classifier...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=150, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("\n📊 --- REALISTIC MODEL EVALUATION ---")
print(f"Model Accuracy based on Real-World Factors: {accuracy_score(y_test, y_pred) * 100:.2f}%")

print("\n🔮 --- SIMULATING REALISTIC SCENARIOS ---")

def predict_realistic_winner(driver_abb, team_name, grid_pos):
    if driver_abb not in driver_stats_map or team_name not in team_stats_map:
        print(f"⚠️ Missing historical map data for {driver_abb} or {team_name}.")
        return
        
    # Dynamically extract factors based on historical mappings
    t_avg = team_stats_map[team_name]
    d_avg = driver_stats_map[driver_abb]
    d_v_c = d_avg - t_avg
    g_v_c = grid_pos - t_avg
    
    scenario = pd.DataFrame([[grid_pos, t_avg, d_avg, d_v_c, g_v_c]], columns=feature_cols)
    
    prediction = model.predict(scenario)[0]
    probabilities = model.predict_proba(scenario)[0]
    
    print(f"Evaluating: {driver_abb} in a {team_name} starting P{grid_pos}")
    print(f" -> Car Strength Rating (Avg Finish): {t_avg:.2f}")
    print(f" -> Driver Skill Rating (Avg Finish): {d_avg:.2f}")
    if prediction == 1:
        print(f"🏆 AI Prediction: WINNER! (Confidence: {probabilities[1]*100:.1f}%)")
    else:
        print(f"❌ AI Prediction: Will NOT win. (Confidence: {probabilities[0]*100:.1f}%)")
    print("-" * 50)

# Test cases proving "Good driver in a bad car can't easily win"
print("Test 1: Elite Driver in an Elite Car from Pole Position")
predict_realistic_winner('VER', 'Red Bull Racing', 1)

print("Test 2: Elite Driver in a historically struggling car from P5")
# Supposing Hamilton or Verstappen was placed in a lower midfield car layout
predict_realistic_winner('VER', 'Kick Sauber', 5)