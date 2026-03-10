import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the prepared dataset
df = pd.read_csv("cbb2_prepared.csv")

# Fill missing RK values
df["RK"] = df["RK"].fillna(df["RK"].max() + 1)

# Create binary tournament indicator
df["TOURNAMENT_TEAM"] = (df["POSTSEASON_NUM"] > 0).astype(int)

# Feature columns
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

target = "TOURNAMENT_TEAM"

# Check for missing values
print("\nChecking for missing values in features:")
print(df[features].isna().sum())

# Split train vs test
train = df[df["YEAR"] <= 2023]
test = df[df["YEAR"] == 2024]

X_train = train[features]
y_train = train[target]

X_test = test[features]
y_test = test[target]

# Create Random Forest Model
model = RandomForestClassifier(
    n_estimators=500,
    max_depth=10,
    random_state=42
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

# Features importance
importance = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
}).sort_values("Importance", ascending=False)

print("\nFeature Importance:")
print(importance)