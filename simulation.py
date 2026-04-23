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
    available_teams = set(df["TEAM"].tolist())

    missing = []
    for _, team1, team2 in REAL_FIRST_ROUND_MATCHUPS:
        if team1 not in available_teams:
            missing.append(team1)
        if team2 not in available_teams:
            missing.append(team2)

    if missing:
        raise ValueError(
            "These bracket teams were not found in cbb26_prepared.csv: "
            + ", ".join(sorted(set(missing)))
        )

    bracket = pd.DataFrame(
        REAL_FIRST_ROUND_MATCHUPS,
        columns=["REGION", "TEAM1", "TEAM2"]
    )

    return bracket


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

REAL_FIRST_ROUND_MATCHUPS = [
    # East
    ("East", "Duke", "Siena"),
    ("East", "Ohio St.", "TCU"),
    ("East", "Louisville", "South Florida"),
    ("East", "Michigan St.", "North Dakota St."),
    ("East", "St. John's", "Northern Iowa"),
    ("East", "Kansas", "Cal Baptist"),
    ("East", "UCLA", "UCF"),
    ("East", "Connecticut", "Furman"),

    # West
    ("West", "Arizona", "LIU"),
    ("West", "Villanova", "Utah St."),
    ("West", "Wisconsin", "High Point"),
    ("West", "Arkansas", "Hawaii"),
    ("West", "BYU", "Texas"),
    ("West", "Gonzaga", "Kennesaw St."),
    ("West", "Purdue", "Queens"),
    ("West", "Miami FL", "Missouri"),

    # Midwest
    ("Midwest", "Michigan", "Howard"),
    ("Midwest", "Georgia", "Saint Louis"),
    ("Midwest", "Kentucky", "Santa Clara"),
    ("Midwest", "Iowa St.", "Tennessee St."),
    ("Midwest", "Texas Tech", "Akron"),
    ("Midwest", "Alabama", "Hofstra"),
    ("Midwest", "Virginia", "Wright St."),
    ("Midwest", "Tennessee", "Miami OH"),

    # South
    ("South", "Florida", "Prairie View A&M"),
    ("South", "Clemson", "Iowa"),
    ("South", "Nebraska", "Troy"),
    ("South", "Vanderbilt", "McNeese St."),
    ("South", "North Carolina", "VCU"),
    ("South", "Illinois", "Penn"),
    ("South", "Saint Mary's", "Texas A&M"),
    ("South", "Houston", "Idaho"),
]

def simulate_tournament(df, bracket, return_path=False):
    current_games = bracket[["TEAM1", "TEAM2"]].values.tolist()

    round_names = [
        "Round of 64",
        "Round of 32",
        "Sweet 16",
        "Elite 8",
        "Final 4",
        "Championship"
    ]

    bracket_path = {}
    round_idx = 0

    while len(current_games) > 0:
        winners = []
        game_results = []

        for team1, team2 in current_games:
            winner = predict_winner(team1, team2, df)
            winners.append(winner)

            game_results.append({
                "team1": team1,
                "team2": team2,
                "winner": winner
            })

        round_name = round_names[round_idx]
        bracket_path[round_name] = game_results

        if len(winners) == 1:
            if return_path:
                return winners[0], bracket_path
            return winners[0]

        current_games = [
            [winners[i], winners[i + 1]]
            for i in range(0, len(winners), 2)
        ]
        round_idx += 1


def run_simulation(num_simulations=1000):
    df = load_data()
    df = compute_strengths(df)
    bracket = create_bracket(df)

    counts = Counter()
    sample_bracket = None

    for sim in range(num_simulations):
        champ, bracket_path = simulate_tournament(df, bracket, return_path=True)
        counts[champ] += 1

        if sim == 0:
            sample_bracket = bracket_path

    results = {
        team: count / num_simulations
        for team, count in counts.most_common(10)
    }

    return results, sample_bracket, bracket


if __name__ == "__main__":
    num_simulations = 100

    results, sample_bracket, first_round_bracket = run_simulation(num_simulations)

    print("\nFIRST ROUND MATCHUPS")
    print(first_round_bracket.to_string(index=False))

    print("\nTOP CHAMPION PROBABILITIES")
    for team, prob in results.items():
        print(f"{team}: {prob:.2%}")

    print("\nSAMPLE BRACKET PATH")
    for round_name, games in sample_bracket.items():
        print(f"\n{round_name}")
        for game in games:
            print(f"{game['team1']} vs {game['team2']} -> {game['winner']}")