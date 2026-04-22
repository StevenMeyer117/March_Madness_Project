import streamlit as st
import pandas as pd
from simulation import run_simulation

# ==============================
# PAGE CONFIG
# ==============================

st.set_page_config(
    page_title="March Madness Predictor",
    page_icon="🏀",
    layout="wide"
)

# ==============================
# TITLE + DESCRIPTION
# ==============================

st.title("🏀 March Madness Predictor")

st.markdown(
    "Run Monte Carlo simulations using a machine learning model to estimate each team's probability of winning the NCAA Tournament."
)

# ==============================
# INPUT CONTROLS
# ==============================

st.subheader("Simulation Settings")

num_sims = st.slider(
    "Number of Simulations",
    min_value=100,
    max_value=5000,
    value=1000,
    step=100
)

run_button = st.button("Run Simulation")

# ==============================
# RUN SIMULATION
# ==============================

if run_button:

    st.write(f"Running {num_sims} simulations...")

    with st.spinner("Simulating tournament..."):
        champion_probs, bracket = run_simulation(num_sims)

    st.success("Simulation complete!")

    # ==============================
    # PROBABILITY RESULTS
    # ==============================

    df = pd.DataFrame(
        list(champion_probs.items()),
        columns=["Team", "Probability"]
    )

    df["Probability"] = df["Probability"] * 100
    df = df.sort_values("Probability", ascending=False)

    # Top team
    st.subheader("🥇 Most Likely Champion")
    st.success(f"{df.iloc[0]['Team']} ({df.iloc[0]['Probability']:.2f}%)")

    # Table
    st.subheader("🏆 Champion Probabilities")
    st.dataframe(df)

    # Chart
    st.bar_chart(df.set_index("Team"))

    # ==============================
    # BRACKET DISPLAY (UPDATED)
    # ==============================

    st.subheader("🏀 Tournament Bracket")

    # Loop through all rounds/regions cleanly
    for round_name, games in bracket.items():
        st.markdown(f"### {round_name}")
        for game in games:
            st.write(game)

    # Final Champion Highlight
    if "Champion" in bracket:
        st.markdown("---")
        st.success(f"🏆 Tournament Winner: {bracket['Champion'][0]}")