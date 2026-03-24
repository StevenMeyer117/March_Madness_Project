import math
import pandas as pd
import joblib
scaler = joblib.load("scaler.pkl")

# ==============================
# Load Model
# ==============================

model = joblib.load("march_madness_model.pkl")


# ==============================
# Load Data
# ==============================

df = pd.read_csv("cbb2_prepared.csv")
df["NET_EFF"] = df["ADJOE"] - df["ADJDE"]

teams_2024 = df[df["YEAR"] == 2024].copy()


# ==============================
# Feature Columns
# ==============================

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
# Compute Team Strength
# ==============================

X_scaled = scaler.transform(teams_2024[features])

import numpy as np

# Convert probabilities to log-odds (fix compression issue)
probs = model.predict_proba(X_scaled)[:,1]
strength = np.log(probs / (1 - probs))

teams_2024["TEAM_STRENGTH"] = strength


# ==============================
# Show Top Teams
# ==============================

ranked = teams_2024.sort_values("TEAM_STRENGTH", ascending=False)

print("\nTop Teams According to Model:\n")

print(ranked[["TEAM","TEAM_STRENGTH"]].head(20))


# ==============================
# Predict Winner of Game
# ==============================

import random

def predict_winner(team1, team2, df):

    team1_row = df.loc[df["TEAM"] == team1]
    team2_row = df.loc[df["TEAM"] == team2]

    if team1_row.empty or team2_row.empty:
        print(f"Team not found: {team1} or {team2}")
        return None

    s1 = team1_row["TEAM_STRENGTH"].values[0]
    s2 = team2_row["TEAM_STRENGTH"].values[0]

    # Compute probability
    import math

    # Convert to log-odds style difference
    diff = s1 - s2

    # Sigmoid function (better separation)
    prob_team1 = 1 / (1 + math.exp(-0.3 * diff))

    # Random outcome
    if random.random() < prob_team1:
        return team1
    else:
        return team2


# ==============================
# Simulate One Round
# ==============================

def simulate_round(team_list, df):

    winners = []

    for i in range(0, len(team_list), 2):

        team1 = team_list[i]
        team2 = team_list[i+1]

        winner = predict_winner(team1, team2, df)

        winners.append(winner)

    return winners


# ==============================
# Load Round of 64
# ==============================

round1 = pd.read_csv("bracket_2024_round1.csv")


round64 = []

for _, row in round1.iterrows():

    winner = predict_winner(row["TEAM1"], row["TEAM2"], teams_2024)

    round64.append(winner)


print("\nROUND OF 64 WINNERS\n")

for team in round64:
    print(team)


# ==============================
# Run Tournament Until Champion
# ==============================

current_round = round64
round_number = 32

while len(current_round) > 1:

    print(f"\nROUND OF {round_number}\n")

    current_round = simulate_round(current_round, teams_2024)

    for team in current_round:
        print(team)

    round_number = round_number // 2


print("\nPREDICTED CHAMPION:\n")

print(current_round[0])


from collections import Counter

def simulate_tournament(df, round1):
    winners = []

    # Round of 64
    for _, row in round1.iterrows():
        winner = predict_winner(row["TEAM1"], row["TEAM2"], df)
        winners.append(winner)

    current_round = winners

    # Continue until champion
    while len(current_round) > 1:
        next_round = []

        for i in range(0, len(current_round), 2):
            team1 = current_round[i]
            team2 = current_round[i + 1]

            winner = predict_winner(team1, team2, df)
            next_round.append(winner)

        current_round = next_round

    return current_round[0]


# =========================
# MONTE CARLO SIMULATION
# =========================

num_simulations = 1000
champion_counts = Counter()

for _ in range(num_simulations):
    champ = simulate_tournament(teams_2024, round1)
    champion_counts[champ] += 1


print("\n\nCHAMPION PROBABILITIES:\n")

for team, count in champion_counts.most_common(10):
    prob = count / num_simulations
    print(f"{team}: {prob:.2%}")