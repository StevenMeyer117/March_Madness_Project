import pandas as pd
import joblib
import numpy as np
import math
import random
from collections import Counter

# ==============================
# Load model + scaler ONCE
# ==============================

model = joblib.load("march_madness_model.pkl")
scaler = joblib.load("scaler.pkl")

# ==============================
# Load + prepare data
# ==============================

def load_data():
    df = pd.read_csv("cbb26_prepared.csv")
    df["NET_EFF"] = df["ADJOE"] - df["ADJDE"]
    return df


# ==============================
# Compute strengths
# ==============================

def compute_strengths(df):

    features = [
        "WP","ADJOE","ADJDE","NET_EFF","BARTHAG",
        "TOR","TORD","ORB","DRB","FTR","FTRD",
        "2P_O","2P_D","3P_O","3P_D","RK"
    ]

    X_scaled = scaler.transform(df[features])

    probs = model.predict_proba(X_scaled)[:, 1]
    strength = np.log(probs / (1 - probs))

    df["TEAM_STRENGTH"] = strength

    return df.sort_values("TEAM_STRENGTH", ascending=False)


# ==============================
# Create bracket (top 64 teams)
# ==============================

def create_bracket(df):

    top_64 = df["TEAM"].head(64).tolist()

    matchups = []

    for i in range(32):
        matchups.append((top_64[i], top_64[63 - i]))

    return pd.DataFrame(matchups, columns=["TEAM1", "TEAM2"])


# ==============================
# Predict winner
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

def simulate_tournament(df, bracket):

    winners = []

    for _, row in bracket.iterrows():
        winners.append(predict_winner(row["TEAM1"], row["TEAM2"], df))

    current = winners

    while len(current) > 1:
        next_round = []
        for i in range(0, len(current), 2):
            next_round.append(predict_winner(current[i], current[i+1], df))
        current = next_round

    return current[0]


# ==============================
# MAIN FUNCTION (this is key)
# ==============================

def run_simulation(num_simulations=1000):

    df = load_data()
    df = compute_strengths(df)
    bracket = create_bracket(df)

    counts = Counter()

    for _ in range(num_simulations):
        champ = simulate_tournament(df, bracket)
        counts[champ] += 1

    results = {
        team: count / num_simulations
        for team, count in counts.most_common(10)
    }

    return results