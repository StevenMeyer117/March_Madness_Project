import pandas as pd
import joblib
import numpy as np
import random

# ==============================
# LOAD MODEL + SCALER
# ==============================

model = joblib.load("march_madness_model.pkl")
scaler = joblib.load("scaler.pkl")


# ==============================
# LOAD DATA
# ==============================

def load_data():
    df = pd.read_csv("cbb26_prepared.csv")
    df["NET_EFF"] = df["ADJOE"] - df["ADJDE"]

    # Normalize team names once
    df["TEAM_CLEAN"] = df["TEAM"].apply(normalize_name)

    return df


# ==============================
# NORMALIZE NAMES
# ==============================

def normalize_name(name):
    if name is None:
        return None
    return (
        name.strip()
        .lower()
        .replace(".", "")
        .replace("'", "")
        .replace("&", "and")
    )


# ==============================
# COMPUTE TEAM STRENGTH
# ==============================

def compute_strengths(df):

    features = [
        "WP","ADJOE","ADJDE","NET_EFF","BARTHAG",
        "TOR","TORD","ORB","DRB","FTR","FTRD",
        "2P_O","2P_D","3P_O","3P_D","RK"
    ]

    X_scaled = scaler.transform(df[features])

    probs = model.predict_proba(X_scaled)[:, 1]

    # log-odds for better separation
    strength = np.log(probs / (1 - probs))

    df["TEAM_STRENGTH"] = strength

    return df


# ==============================
# PREDICT GAME
# ==============================

def predict_winner(team1, team2, df):

    if team1 is None or team2 is None:
        return None

    t1 = df.loc[df["TEAM_CLEAN"] == normalize_name(team1)]
    t2 = df.loc[df["TEAM_CLEAN"] == normalize_name(team2)]

    # fallback fuzzy match
    if t1.empty:
        t1 = df[df["TEAM_CLEAN"].str.contains(normalize_name(team1).split()[0])]
        if t1.empty:
            return None
        t1 = t1.iloc[[0]]

    if t2.empty:
        t2 = df[df["TEAM_CLEAN"].str.contains(normalize_name(team2).split()[0])]
        if t2.empty:
            return None
        t2 = t2.iloc[[0]]

    p1 = t1["TEAM_STRENGTH"].values[0]
    p2 = t2["TEAM_STRENGTH"].values[0]

    prob_team1 = p1 / (p1 + p2)

    # RETURN CLEAN DATASET NAME (important)
    return t1["TEAM"].values[0] if random.random() < prob_team1 else t2["TEAM"].values[0]


# ==============================
# SIMULATE FULL TOURNAMENT
# ==============================

def simulate_tournament(df, bracket):

    rounds = {}

    # ==============================
    # ROUND OF 64 (REAL MATCHUPS)
    # ==============================

    r64_matchups = []
    r64_winners = []

    for _, row in bracket.iterrows():
        t1 = row["TEAM1"]
        t2 = row["TEAM2"]

        r64_matchups.append((t1, t2))

        w = predict_winner(t1, t2, df)
        if w is None:
            return None

        r64_winners.append(w)

    rounds["R64_matchups"] = r64_matchups

    # ==============================
    # GENERIC ROUND BUILDER
    # ==============================

    def play_round(prev):
        matchups = []
        winners = []

        for i in range(0, len(prev), 2):
            t1 = prev[i]
            t2 = prev[i + 1]

            matchups.append((t1, t2))

            w = predict_winner(t1, t2, df)
            if w is None:
                return None, None

            winners.append(w)

        return matchups, winners

    # ==============================
    # BUILD TOURNAMENT
    # ==============================

    r32_matchups, r32_winners = play_round(r64_winners)
    if r32_matchups is None:
        return None
    rounds["R32_matchups"] = r32_matchups

    s16_matchups, s16_winners = play_round(r32_winners)
    if s16_matchups is None:
        return None
    rounds["S16_matchups"] = s16_matchups

    e8_matchups, e8_winners = play_round(s16_winners)
    if e8_matchups is None:
        return None
    rounds["E8_matchups"] = e8_matchups

    f4_matchups, f4_winners = play_round(e8_winners)
    if f4_matchups is None:
        return None
    rounds["F4_matchups"] = f4_matchups

    champ_matchup, champ = play_round(f4_winners)
    if champ_matchup is None:
        return None

    rounds["CHAMP_matchup"] = champ_matchup
    rounds["CHAMP"] = champ[0]

    return rounds


# ==============================
# MONTE CARLO SIMULATION
# ==============================

def run_simulation(bracket_file, num_simulations=1000):

    df = load_data()
    df = compute_strengths(df)

    bracket = pd.read_csv(bracket_file)

    from collections import Counter
    champion_counts = Counter()

    sample_bracket = None

    for i in range(num_simulations):

        results = simulate_tournament(df, bracket)
        if results is None:
            continue

        champion_counts[results["CHAMP"]] += 1

        # Save LAST simulation for display
        if i == num_simulations - 1:
            sample_bracket = results

    probs = {
        team: count / num_simulations
        for team, count in champion_counts.most_common(10)
    }

    return probs, sample_bracket