import pandas as pd

# Load cbb2_ranked.csv into a DataFrame
df = pd.read_csv("cbb2_ranked.csv")

# Create Win Percentage
df["WP"] = (df["W"] / df["G"]).round(3)

# Insert new WP column after W column
cols = list(df.columns)
w_index = cols.index("W")

cols.insert(w_index +1, cols.pop(cols.index("WP")))

df = df[cols]

# Convert POSTSEASON to Numeric
post_map = {
    "R64": 1,
    "R32": 2,
    "S16": 3,
    "E8": 4,
    "F4": 5,
    "2ND": 6,
    "Champions": 7
}

df["POSTSEASON_NUM"] = df["POSTSEASON"].map(post_map).fillna(0)

# Save the modified DataFrame to a new CSV file
df.to_csv("cbb2_prepared.csv", index=False)

print("Prepared dataset saved as cbb2_prepared.csv")