import pandas as pd
import numpy as np
import joblib

# ==============================
# LOAD MODEL + SCALER
# ==============================

model = joblib.load("march_madness_model.pkl")
scaler = joblib.load("scaler.pkl")

# ==============================
# LOAD + PREP DATA
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
    df["TEAM_CLEAN"] = df["TEAM"].apply(normalize_name)

    df = df.dropna(subset=["BARTHAG", "TEAM"])
    return df


# ==============================
# REAL NCAA SEEDING SYSTEM
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

    n = len(teams)

    matchups = []

    for i in range(n // 2):
        matchups.append((teams[i], teams[n - 1 - i]))

    return matchups


# ==============================
# SAFE PREDICTION
# ==============================

def predict_game(teamA, teamB, df):

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
# ==============================

def pair_teams(teams):

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
# ==============================

def run_simulation(num_simulations=1000):

    df = load_data()

    counts = {}
    last = None

    for _ in range(num_simulations):

        bracket = simulate_single_bracket(df)

        champ = bracket["Champion"][0]
        counts[champ] = counts.get(champ, 0) + 1

        last = bracket

    probs = {k: v / num_simulations for k, v in counts.items()}

    return probs, last