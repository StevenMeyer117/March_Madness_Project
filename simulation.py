import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

# ==============================
# LOAD + TRAIN MODEL
# ==============================

df = pd.read_csv("cbb2_prepared.csv")

df["RK"] = df["RK"].fillna(df["RK"].max() + 1)
df = df[df["POSTSEASON_NUM"] > 0]

features = [
    "ADJOE",
    "ADJDE",
    "BARTHAG",
    "TOR",
    "ORB",
    "DRB",
    "FTR",
    "RK"
]

df["TARGET"] = (df["BARTHAG"] > df["BARTHAG"].median()).astype(int)

model = LogisticRegression(max_iter=1000)
model.fit(df[features], df["TARGET"])

# ==============================
# GAME SIMULATION
# ==============================

def simulate_game(teamA, teamB):

    diff = np.array([
        teamA["ADJOE"] - teamB["ADJOE"],
        teamA["ADJDE"] - teamB["ADJDE"],
        teamA["BARTHAG"] - teamB["BARTHAG"],
        teamA["TOR"] - teamB["TOR"],
        teamA["ORB"] - teamB["ORB"],
        teamA["DRB"] - teamB["DRB"],
        teamA["FTR"] - teamB["FTR"],
        teamA["RK"] - teamB["RK"]
    ]).reshape(1, -1)

    prob = model.predict_proba(diff)[0][1]

    return teamA if np.random.rand() < prob else teamB

# ==============================
# CREATE SEEDED REGION
# ==============================

def create_region(teams):

    # Use SEED if available, else RK
    if "SEED" in teams.columns:
        teams = teams.sort_values("SEED")
    else:
        teams = teams.sort_values("RK")

    teams = teams.reset_index(drop=True)

    # Standard NCAA matchups
    pairs = [
        (0,15), (7,8), (4,11), (3,12),
        (5,10), (2,13), (6,9), (1,14)
    ]

    matchups = []
    for a,b in pairs:
        matchups.append((teams.iloc[a], teams.iloc[b]))

    return matchups

# ==============================
# PLAY ROUND
# ==============================

def play_round(matchups, round_name, bracket, region=None):

    winners = []
    games = []

    for teamA, teamB in matchups:

        winner = simulate_game(teamA, teamB)

        label = f"{teamA['TEAM']} vs {teamB['TEAM']} → {winner['TEAM']}"
        games.append(label)

        winners.append(winner)

    key = f"{region} - {round_name}" if region else round_name
    bracket[key] = games

    return winners

# ==============================
# SIMULATE REGION
# ==============================

def simulate_region(teams, region_name, bracket):

    r64 = play_round(create_region(teams), "Round of 64", bracket, region_name)

    def to_matchups(t):
        return [(t[i], t[i+1]) for i in range(0, len(t), 2)]

    r32 = play_round(to_matchups(r64), "Round of 32", bracket, region_name)
    s16 = play_round(to_matchups(r32), "Sweet 16", bracket, region_name)
    e8 = play_round(to_matchups(s16), "Elite 8", bracket, region_name)

    # Region winner
    return e8[0]

# ==============================
# SINGLE BRACKET SIMULATION
# ==============================

def simulate_single_bracket(teams):

    bracket = {}

    # Split into 4 regions
    teams = teams.sort_values("RK").reset_index(drop=True)

    regions = {
        "East": teams.iloc[0:16],
        "West": teams.iloc[16:32],
        "South": teams.iloc[32:48],
        "Midwest": teams.iloc[48:64]
    }

    region_winners = {}

    for name, region_teams in regions.items():
        winner = simulate_region(region_teams, name, bracket)
        region_winners[name] = winner

    # Final Four
    ff_matchups = [
        (region_winners["East"], region_winners["West"]),
        (region_winners["South"], region_winners["Midwest"])
    ]

    f4 = play_round(ff_matchups, "Final Four", bracket)

    # Championship
    champ_match = [(f4[0], f4[1])]
    final = play_round(champ_match, "Championship", bracket)

    champion = final[0]["TEAM"]
    bracket["Champion"] = [champion]

    return bracket, champion

# ==============================
# MAIN FUNCTION
# ==============================

def run_simulation(num_simulations=1000):

    teams_2024 = df[df["YEAR"] == 2024].copy()

    # Ensure we only take 64 teams
    teams_2024 = teams_2024.sort_values("RK").head(64)

    champion_counts = {}
    sample_bracket = None

    for i in range(num_simulations):

        bracket, champion = simulate_single_bracket(teams_2024)

        champion_counts[champion] = champion_counts.get(champion, 0) + 1

        if i == 0:
            sample_bracket = bracket

    champion_probabilities = {
        team: count / num_simulations
        for team, count in champion_counts.items()
    }

    return champion_probabilities, sample_bracket