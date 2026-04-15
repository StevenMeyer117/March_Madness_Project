# -----------------------------
# 1 Imports
# -----------------------------

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# -----------------------------
# 2 Load Dataset
# -----------------------------

df = pd.read_csv("cbb2_prepared.csv")

# -----------------------------
# 3 Clean Data
# -----------------------------

df["RK"] = df["RK"].fillna(df["RK"].max() + 1)

df["TOURNAMENT_TEAM"] = (df["POSTSEASON_NUM"] > 0).astype(int)

# -----------------------------
# 4 Features
# -----------------------------

features = ["BARTHAG","ADJOE","ADJDE"]

target = "TOURNAMENT_TEAM"

train = df[df["YEAR"] <= 2023]
test = df[df["YEAR"] == 2024]

X_train = train[features]
y_train = train[target]

X_test = test[features]
y_test = test[target]

# -----------------------------
# 5 Train Model
# -----------------------------

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, predictions))

# -----------------------------
# 6 Team Strength Scores
# -----------------------------

probabilities = model.predict_proba(X_test)[:,1]

test = test.copy()
test["TEAM_STRENGTH"] = probabilities

ranked = test.sort_values("TEAM_STRENGTH", ascending=False)

print("\nTop Teams:")
print(ranked[["TEAM","TEAM_STRENGTH"]].head(10))

# -----------------------------
# 7 Game Prediction Function
# -----------------------------

def predict_game(teamA, teamB):

    statsA = ranked[ranked["TEAM"] == teamA].iloc[0]
    statsB = ranked[ranked["TEAM"] == teamB].iloc[0]

    probA = statsA["TEAM_STRENGTH"]
    probB = statsB["TEAM_STRENGTH"]

    win_prob = probA / (probA + probB)

    if win_prob > 0.5:
        return teamA
    else:
        return teamB

# -----------------------------
# 8 Round Simulation
# -----------------------------

def simulate_round(teams):

    winners = []

    for i in range(0,len(teams),2):

        teamA = teams[i]
        teamB = teams[i+1]

        winner = predict_game(teamA,teamB)

        print(teamA,"vs",teamB,"→",winner)

        winners.append(winner)

    return winners

# -----------------------------
# 9 Example Tournament
# -----------------------------

teams = [
"Connecticut","Duke",
"Houston","Kansas",
"Purdue","Arizona",
"Tennessee","Baylor",
"Gonzaga","Alabama",
"Marquette","Creighton",
"Illinois","Auburn",
"Texas","Kentucky"
]

print("\nROUND OF 16")
round16 = simulate_round(teams)

print("\nELITE 8")
elite8 = simulate_round(round16)

print("\nFINAL FOUR")
final4 = simulate_round(elite8)

print("\nCHAMPIONSHIP")
champion = simulate_round(final4)

print("\n🏆 Champion:", champion[0])