import pandas as pd
import joblib
import numpy as np
import math
import random
from collections import Counter

# ==============================
# Load model + scaler
# ==============================

model = joblib.load("march_madness_model.pkl")
scaler = joblib.load("scaler.pkl")

# ==============================
# Load 2025 data
# ==============================

df = pd.read_csv("cbb26_prepared.csv")

# Feature columns (same as training)
features = [
    "WP",
    "ADJOE",
    "ADJDE",
    "NET_EFF",
    "BARTHAG",
    "TOR",
    "TORD",
    "ORB",
    "DRB",
    "FTR",
    "FTRD",
    "2P_O",
    "2P_D",
    "3P_O",
    "3P_D",
    "RK"
]

# ==============================
# Compute team strength
# ==============================

X_scaled = scaler.transform(df[features])

probs = model.predict_proba(X_scaled)[:, 1]
strength = np.log(probs / (1 - probs))

df["TEAM_STRENGTH"] = strength

# ==============================
# Sort teams
# ==============================

ranked = df.sort_values("TEAM_STRENGTH", ascending=False)

print("\nTop Teams (2025):\n")
print(ranked[["TEAM", "TEAM_STRENGTH"]].head(20))

# ==============================
# Game simulation
# ==============================

def predict_winner(team1, team2, df):

    t1 = df.loc[df["TEAM"] == team1]
    t2 = df.loc[df["TEAM"] == team2]

    if t1.empty or t2.empty:
        return None

    s1 = t1["TEAM_STRENGTH"].values[0]
    s2 = t2["TEAM_STRENGTH"].values[0]

    diff = s1 - s2

    prob = 1 / (1 + math.exp(-0.5 * diff))

    return team1 if random.random() < prob else team2

# ==============================
# Simulate tournament
# ==============================

def simulate_tournament(df, round1):

    winners = []

    for _, row in round1.iterrows():
        winners.append(predict_winner(row["TEAM1"], row["TEAM2"], df))

    current = winners

    while len(current) > 1:
        next_round = []
        for i in range(0, len(current), 2):
            next_round.append(predict_winner(current[i], current[i+1], df))
        current = next_round

    return current[0]

# ==============================
# LOAD 2025 BRACKET (YOU NEED THIS)
# ==============================

# Take top 64 teams
top_64 = ranked["TEAM"].head(64).tolist()

# Create matchups (1 vs 64, 2 vs 63, etc.)
round1_pairs = []

for i in range(32):
    round1_pairs.append((top_64[i], top_64[63 - i]))

# Convert to DataFrame
round1 = pd.DataFrame(round1_pairs, columns=["TEAM1", "TEAM2"])

# ==============================
# Monte Carlo
# ==============================

num_simulations = 1000
counts = Counter()

for _ in range(num_simulations):
    champ = simulate_tournament(df, round1)
    counts[champ] += 1

print("\nCHAMPION PROBABILITIES (2025):\n")

for team, count in counts.most_common(10):
    print(f"{team}: {count/num_simulations:.2%}")