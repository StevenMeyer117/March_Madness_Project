import pandas as pd
import numpy as np
<<<<<<< HEAD
import joblib
=======
import random
>>>>>>> origin/Steven

# ==============================
# LOAD MODEL + SCALER
# ==============================

model = joblib.load("march_madness_model.pkl")
scaler = joblib.load("scaler.pkl")


# ==============================
<<<<<<< HEAD
# LOAD + PREP DATA
=======
# LOAD DATA
>>>>>>> origin/Steven
# ==============================

def normalize_name(name):
    if name is None:
        return ""
    return (
        str(name)
        .strip()
        .lower()
        .replace(".", "")
        .replace("'", "")
        .replace("&", "and")
    )


def load_data():
    df = pd.read_csv("cbb26_prepared.csv")

    df["NET_EFF"] = df["ADJOE"] - df["ADJDE"]
<<<<<<< HEAD
    df["TEAM_CLEAN"] = df["TEAM"].apply(normalize_name)

    df = df.dropna(subset=["BARTHAG", "TEAM"])
=======

    # Normalize team names once
    df["TEAM_CLEAN"] = df["TEAM"].apply(normalize_name)

>>>>>>> origin/Steven
    return df


# ==============================
<<<<<<< HEAD
# REAL NCAA SEEDING SYSTEM
=======
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
>>>>>>> origin/Steven
# ==============================

def create_seeds(df):

    df = df.copy()

    df = df.sort_values("BARTHAG", ascending=False).reset_index(drop=True)

    df["SEED_GLOBAL"] = range(1, len(df) + 1)

    return df


def build_regions(df):

    df = create_seeds(df)

    chunk = len(df) // 4

    return {
        "East": df.iloc[0:chunk],
        "West": df.iloc[chunk:2*chunk],
        "South": df.iloc[2*chunk:3*chunk],
        "Midwest": df.iloc[3*chunk:]
    }


def create_region_matchups(region_df):

    region_df = region_df.sort_values("SEED_GLOBAL").reset_index(drop=True)

    teams = [
        {
            "seed": row["SEED_GLOBAL"],
            "team": row["TEAM"]
        }
        for _, row in region_df.iterrows()
    ]

<<<<<<< HEAD
    n = len(teams)

    matchups = []

    for i in range(n // 2):
        matchups.append((teams[i], teams[n - 1 - i]))

    return matchups


# ==============================
# SAFE PREDICTION
=======
    X_scaled = scaler.transform(df[features])

    probs = model.predict_proba(X_scaled)[:, 1]

    # log-odds for better separation
    strength = np.log(probs / (1 - probs))

    df["TEAM_STRENGTH"] = strength

    return df


# ==============================
# PREDICT GAME
>>>>>>> origin/Steven
# ==============================

def predict_game(teamA, teamB, df):

<<<<<<< HEAD
    matchA = df[df["TEAM_CLEAN"] == normalize_name(teamA["team"])]
    matchB = df[df["TEAM_CLEAN"] == normalize_name(teamB["team"])]

    if matchA.empty or matchB.empty:
        return teamA if np.random.rand() > 0.5 else teamB

    rowA = matchA.iloc[0]
    rowB = matchB.iloc[0]

    featuresA = np.array([[rowA["WP"], rowA["ADJOE"], rowA["ADJDE"],
                           rowA["NET_EFF"], rowA["BARTHAG"], rowA["TOR"],
                           rowA["TORD"], rowA["ORB"], rowA["DRB"],
                           rowA["FTR"], rowA["FTRD"], rowA["2P_O"],
                           rowA["2P_D"], rowA["3P_O"], rowA["3P_D"],
                           rowA["RK"]]])

    featuresB = np.array([[rowB["WP"], rowB["ADJOE"], rowB["ADJDE"],
                           rowB["NET_EFF"], rowB["BARTHAG"], rowB["TOR"],
                           rowB["TORD"], rowB["ORB"], rowB["DRB"],
                           rowB["FTR"], rowB["FTRD"], rowB["2P_O"],
                           rowB["2P_D"], rowB["3P_O"], rowB["3P_D"],
                           rowB["RK"]]])

    diff = featuresA - featuresB
    prob = model.predict_proba(scaler.transform(diff))[0][1]

    return teamA if np.random.rand() < prob else teamB


# ==============================
# SAFE PAIRING
=======
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
>>>>>>> origin/Steven
# ==============================

def pair_teams(teams):

<<<<<<< HEAD
    if len(teams) % 2 != 0:
        teams = teams[:-1]

    return [(teams[i], teams[i+1]) for i in range(0, len(teams), 2)]


# ==============================
# REGION SIMULATION
# ==============================

def simulate_region(matchups, df, region_name):

    rounds = ["Round of 64", "Round of 32", "Sweet 16", "Elite 8"]
    bracket = {}

    current = matchups
    winner = None

    for round_name in rounds:

        next_round = []
        games = []

        for teamA, teamB in current:

            win = predict_game(teamA, teamB, df)

            games.append(f"{teamA['seed']} {teamA['team']} vs {teamB['seed']} {teamB['team']} → {win['team']}")

            next_round.append(win)

        bracket[f"{region_name} - {round_name}"] = games

        if len(next_round) == 1:
            winner = next_round[0]
            break

        current = pair_teams(next_round)

    return bracket, winner


# ==============================
# FULL BRACKET SIMULATION
# ==============================

def simulate_single_bracket(df):

    regions = build_regions(df)

    bracket = {}
    final_four = []

    for region_name, region_df in regions.items():

        matchups = create_region_matchups(region_df)

        region_bracket, winner = simulate_region(matchups, df, region_name)

        bracket.update(region_bracket)
        final_four.append(winner)

    ff_games = []
    winners = []

    for i in range(0, len(final_four), 2):

        if i + 1 >= len(final_four):
            break

        teamA = final_four[i]
        teamB = final_four[i + 1]

        win = predict_game(teamA, teamB, df)

        ff_games.append(f"{teamA['team']} vs {teamB['team']} → {win['team']}")
        winners.append(win)

    bracket["Final Four"] = ff_games

    if len(winners) >= 2:

        champ = predict_game(winners[0], winners[1], df)

        bracket["Championship"] = [
            f"{winners[0]['team']} vs {winners[1]['team']} → {champ['team']}"
        ]

        bracket["Champion"] = [champ["team"]]

    return bracket


# ==============================
# MONTE CARLO
=======
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
>>>>>>> origin/Steven
# ==============================

def run_simulation(bracket_file, num_simulations=1000):

    df = load_data()
<<<<<<< HEAD

    counts = {}
    last = None

    for _ in range(num_simulations):

        bracket = simulate_single_bracket(df)

        champ = bracket["Champion"][0]
        counts[champ] = counts.get(champ, 0) + 1

        last = bracket

    probs = {k: v / num_simulations for k, v in counts.items()}

    return probs, last
=======
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
>>>>>>> origin/Steven
