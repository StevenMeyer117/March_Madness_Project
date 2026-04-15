import pandas as pd

# Load 2025 data
df = pd.read_csv("cbb26.csv")

# =========================
# Create WP (Win Percentage)
# =========================
df["WP"] = (df["W"] / df["G"]).round(3)

# =========================
# Create NET_EFF
# =========================
df["NET_EFF"] = df["ADJOE"] - df["ADJDE"]

# =========================
# Handle missing RK
# =========================
df["RK"] = df["RK"].fillna(df["RK"].max() + 1)

# =========================
# Add YEAR column (IMPORTANT)
# =========================
df["YEAR"] = 2025

# =========================
# Save prepared file
# =========================
df.to_csv("cbb26_prepared.csv", index=False)

print("2025 data prepared successfully!")