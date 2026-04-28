import pandas as pd
import joblib
import numpy as np
import random
from collections import Counter

# ==============================
# LOAD MODEL + SCALER
# ==============================

model = joblib.load("march_madness_model.pkl")
scaler = joblib.load("scaler.pkl")


# ==============================
# NORMALIZE NAMES
# ==============================

def normalize_name(name):
    if name is None:
        return None
    return (
        str(name)
        .strip()
        .lower()
        .replace(".", "")
        .replace("'", "")
        .replace("&", "and")
    )


# ==============================
# LOAD DATA
# ==============================

def load_data():
    df = pd.read_csv("cbb26_prepared.csv")
    df["NET_EFF"] = df["ADJOE"] - df["ADJDE"]
    df["TEAM_CLEAN"] = df["TEAM"].apply(normalize_name)
    return df


# ==============================
# COMPUTE TEAM STRENGTH
# ==============================

def compute_strengths(df):
    features = [
        "WP", "ADJOE", "ADJDE", "NET_EFF", "BARTHAG",
        "TOR", "TORD", "ORB", "DRB", "FTR", "FTRD",
        "2P_O", "2P_D", "3P_O", "3P_D", "RK"
    ]

    X_scaled = scaler.transform(df[features])
    probs = model.predict_proba(X_scaled)[:, 1]

    probs = np.clip(probs, 1e-6, 1 - 1e-6)
    df["TEAM_STRENGTH"] = np.log(probs / (1 - probs))

    return df


# ==============================
# FIND TEAM ROW
# ==============================

def find_team(team, df):
    team_clean = normalize_name(team)

    exact = df.loc[df["TEAM_CLEAN"] == team_clean]
    if not exact.empty:
        return exact.iloc[0]

    fallback = df[df["TEAM_CLEAN"].str.contains(team_clean.split()[0], na=False)]
    if not fallback.empty:
        return fallback.iloc[0]

    return None


# ==============================
# PREDICT GAME
# ==============================

def predict_winner(team1, team2, df):
    if team1 is None or team2 is None:
        return None

    t1 = find_team(team1, df)
    t2 = find_team(team2, df)

    if t1 is None or t2 is None:
        return None

    s1 = t1["TEAM_STRENGTH"]
    s2 = t2["TEAM_STRENGTH"]

    prob_team1 = 1 / (1 + np.exp(-(s1 - s2)))

    return t1["TEAM"] if random.random() < prob_team1 else t2["TEAM"]


# ==============================
# PLAY ROUND
# ==============================

def play_round(prev_winners, df):
    matchups = []
    winners = []

    for i in range(0, len(prev_winners), 2):
        t1 = prev_winners[i]
        t2 = prev_winners[i + 1]

        matchups.append((t1, t2))

        winner = predict_winner(t1, t2, df)
        if winner is None:
            return None, None

        winners.append(winner)

    return matchups, winners


# ==============================
# SIMULATE ONE REGION
# ==============================

def simulate_region(df, region_name, region_games):
    rounds = {}

    r64_matchups = []
    r64_winners = []

    for _, row in region_games.iterrows():
        t1 = row["TEAM1"]
        t2 = row["TEAM2"]

        r64_matchups.append((t1, t2))

        winner = predict_winner(t1, t2, df)
        if winner is None:
            return None

        r64_winners.append(winner)

    r32_matchups, r32_winners = play_round(r64_winners, df)
    s16_matchups, s16_winners = play_round(r32_winners, df)
    e8_matchups, e8_winners = play_round(s16_winners, df)

    if None in [r32_matchups, s16_matchups, e8_matchups]:
        return None

    rounds["REGION"] = region_name
    rounds["R64_matchups"] = r64_matchups
    rounds["R32_matchups"] = r32_matchups
    rounds["S16_matchups"] = s16_matchups
    rounds["E8_matchups"] = e8_matchups
    rounds["REGION_CHAMP"] = e8_winners[0]

    return rounds


# ==============================
# SIMULATE FULL TOURNAMENT
# ==============================

def simulate_tournament(df, bracket):
    required_cols = {"REGION", "TEAM1", "TEAM2"}
    if not required_cols.issubset(bracket.columns):
        raise ValueError("Bracket CSV must contain REGION, TEAM1, and TEAM2 columns.")

    region_order = ["East", "West", "South", "Midwest"]
    tournament = {
        "regions": {},
        "Final Four_matchups": [],
        "Championship_matchup": [],
        "CHAMP": None,
    }

    region_champs = []

    for region in region_order:
        region_games = bracket[bracket["REGION"] == region]

        if len(region_games) != 8:
            raise ValueError(f"{region} must have exactly 8 Round of 64 games.")

        region_result = simulate_region(df, region, region_games)

        if region_result is None:
            return None

        tournament["regions"][region] = region_result
        region_champs.append(region_result["REGION_CHAMP"])

    # Final Four: East vs West, South vs Midwest
    final_four_matchups, final_four_winners = play_round(region_champs, df)

    if final_four_matchups is None:
        return None

    championship_matchup, champion = play_round(final_four_winners, df)

    if championship_matchup is None:
        return None

    tournament["Final Four_matchups"] = final_four_matchups
    tournament["Championship_matchup"] = championship_matchup
    tournament["CHAMP"] = champion[0]

    return tournament


# ==============================
# MONTE CARLO SIMULATION
# ==============================

def run_simulation(bracket_file, num_simulations=1000):
    df = load_data()
    df = compute_strengths(df)

    bracket = pd.read_csv(bracket_file)

    champion_counts = Counter()
    sample_bracket = None
    completed_sims = 0

    for i in range(num_simulations):
        results = simulate_tournament(df, bracket)

        if results is None:
            continue

        champion_counts[results["CHAMP"]] += 1
        sample_bracket = results
        completed_sims += 1

    probs = {
        team: count / completed_sims
        for team, count in champion_counts.most_common(10)
    } if completed_sims > 0 else {}

    return probs, sample_bracket