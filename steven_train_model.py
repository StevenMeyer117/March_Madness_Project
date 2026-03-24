import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Load the prepared dataset
df = pd.read_csv("cbb2_prepared.csv")
df["NET_EFF"] = df["ADJOE"] - df["ADJDE"]

# Fill missing RK values
df["RK"] = df["RK"].fillna(df["RK"].max() + 1)

# Create binary tournament indicator
df["TOURNAMENT_TEAM"] = (df["POSTSEASON_NUM"] > 0).astype(int)

# Feature columns
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

target = "TOURNAMENT_TEAM"

# Check for missing values
print("\nChecking for missing values in features:")
print(df[features].isna().sum())

# Split train vs test
train = df[df["YEAR"] <= 2023]
test = df[df["YEAR"] == 2024]

scaler = StandardScaler()

X_train = scaler.fit_transform(train[features])
X_test = scaler.transform(test[features])

joblib.dump(scaler, "scaler.pkl")

y_train = train[target]
y_test = test[target]

# Create Logistic Regression model
model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

# Train the model
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, "march_madness_model.pkl")
print("\nModel saved as march_madness_model.pkl")

# Predict 2024 outcomes
predictions = model.predict(X_test)

# Evaluate the model
print("\nModel Accuracy:", accuracy_score(y_test, predictions))
print("\nClassification Report:\n")
print(classification_report(y_test, predictions))

# Probability of being strong team
probabilities = model.predict_proba(X_test)[:, 1]

test["TEAM_STRENGTH"] = probabilities

# Rank teams
top_teams = test.sort_values("TEAM_STRENGTH", ascending=False)

top_teams[["TEAM", "TEAM_STRENGTH"]].to_csv(
    "team_strenghts_2024.csv", index=False
)

print("\nTeam strenghts exported to team_strenghts_2024.csv")

print("\nTop Predicted Teams (2024):")
print(top_teams[["TEAM", "TEAM_STRENGTH"]].head(10))