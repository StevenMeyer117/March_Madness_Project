import pandas as pd
import joblib


# ==============================
# Load Model
# ==============================

model = joblib.load("march_madness_model.pkl")


# ==============================
# Load Data
# ==============================

df = pd.read_csv("cbb2_prepared.csv")

teams_2024 = df[df["YEAR"] == 2024].copy()


# ==============================
# Feature Columns
# ==============================

features = [
    "WP",
    "ADJOE",
    "ADJDE",
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

strength = model.predict_proba(teams_2024[features])[:,1]

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

def predict_winner(team1, team2, df):

    team1_row = df.loc[df["TEAM"] == team1]
    team2_row = df.loc[df["TEAM"] == team2]

    if team1_row.empty or team2_row.empty:
        print(f"Team not found: {team1} or {team2}")
        return None

    s1 = team1_row["TEAM_STRENGTH"].values[0]
    s2 = team2_row["TEAM_STRENGTH"].values[0]

    if s1 > s2:
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