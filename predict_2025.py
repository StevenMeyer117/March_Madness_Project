def predict_winner(team1, team2, df):

    if team1 is None or team2 is None:
        return None

    team1_n = apply_alias(team1)
    team2_n = apply_alias(team2)

    t1 = df.loc[df["TEAM_CLEAN"] == team1_n]
    t2 = df.loc[df["TEAM_CLEAN"] == team2_n]

    if t1.empty or t2.empty:
        print(f"\nMISSING TEAM:")
        print(f"{team1} vs {team2}")
        return None

    s1 = t1["TEAM_STRENGTH"].values[0]
    s2 = t2["TEAM_STRENGTH"].values[0]

    diff = s1 - s2

    # 🔥 SCALE DOWN DIFFERENCE (CRITICAL FIX)
    prob = 1 / (1 + math.exp(-0.1 * diff))

    return team1 if random.random() < prob else team2