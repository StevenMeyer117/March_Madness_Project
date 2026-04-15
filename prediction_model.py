import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# -----------------------------
# 1. Load Dataset
# -----------------------------

df = pd.read_csv("cbb2_prepared.csv")

print("Dataset Loaded")
print("Rows:", len(df))

# -----------------------------
# 2. Clean Data
# -----------------------------

df["RK"] = df["RK"].fillna(df["RK"].max() + 1)

# Only keep tournament teams
df = df[df["POSTSEASON_NUM"] > 0]

# -----------------------------
# 3. Create Matchups
# -----------------------------

matchups = []

for year in df["YEAR"].unique():
    
    season = df[df["YEAR"] == year]
    
    teams = season.to_dict("records")
    
    for i in range(len(teams)):
        for j in range(i+1, len(teams)):
            
            teamA = teams[i]
            teamB = teams[j]
            
            row = {
                "YEAR": year,
                
                "ADJOE_diff": teamA["ADJOE"] - teamB["ADJOE"],
                "ADJDE_diff": teamA["ADJDE"] - teamB["ADJDE"],
                "BARTHAG_diff": teamA["BARTHAG"] - teamB["BARTHAG"],
                "TOR_diff": teamA["TOR"] - teamB["TOR"],
                "ORB_diff": teamA["ORB"] - teamB["ORB"],
                "DRB_diff": teamA["DRB"] - teamB["DRB"],
                "FTR_diff": teamA["FTR"] - teamB["FTR"],
                "RK_diff": teamA["RK"] - teamB["RK"]
            }
            
            # Fake target for now (higher BARTHAG assumed stronger)
            row["TEAM_A_WIN"] = 1 if teamA["BARTHAG"] > teamB["BARTHAG"] else 0
            
            matchups.append(row)

matchups_df = pd.DataFrame(matchups)

print("Total Matchups Created:", len(matchups_df))

# -----------------------------
# 4. Features
# -----------------------------

features = [
    "ADJOE_diff",
    "ADJDE_diff",
    "BARTHAG_diff",
    "TOR_diff",
    "ORB_diff",
    "DRB_diff",
    "FTR_diff",
    "RK_diff"
]

target = "TEAM_A_WIN"

# -----------------------------
# 5. Train / Test Split
# -----------------------------

train = matchups_df[matchups_df["YEAR"] <= 2023]
test = matchups_df[matchups_df["YEAR"] == 2024]

X_train = train[features]
y_train = train[target]

X_test = test[features]
y_test = test[target]

print("Training rows:", len(train))
print("Testing rows:", len(test))

# -----------------------------
# 6. Train Logistic Regression
# -----------------------------

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

print("Model trained")

# -----------------------------
# 7. Predict Games
# -----------------------------

predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:,1]

# -----------------------------
# 8. Evaluate Model
# -----------------------------

accuracy = accuracy_score(y_test, predictions)

print("\nModel Accuracy:", accuracy)
print("\nClassification Report:")
print(classification_report(y_test, predictions))

# -----------------------------
# 9. Show Example Predictions
# -----------------------------

test = test.copy()
test["WIN_PROBABILITY"] = probabilities

print("\nSample Game Predictions:")
print(test.head(10))