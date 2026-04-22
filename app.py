import streamlit as st
import pandas as pd
from simulation import run_simulation

# ==============================
# PAGE CONFIG
# ==============================

st.set_page_config(
    page_title="March Madness Predictor",
    page_icon="🏀",
    layout="centered"
)

# ==============================
# UI SECTION (Teammate 1)
# ==============================

st.title("🏀 March Madness Predictor")

st.write(
    "Run Monte Carlo simulations to estimate NCAA Tournament champion probabilities."
)

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
# DATA SECTION (DO NOT TOUCH)
# ==============================

if run_button:

    with st.spinner("Running simulations..."):
        results, sample_bracket = run_simulation(num_sims)

    st.success("Simulation complete!")

    st.subheader("🏆 Champion Probabilities")

    df = pd.DataFrame(
        list(results.items()),
        columns=["Team", "Probability"]
    )

    df["Probability"] = df["Probability"] * 100
    df = df.sort_values("Probability", ascending=False).reset_index(drop=True)

    st.dataframe(df)

    st.bar_chart(
        df,
        x="Team",
        y="Probability",
        horizontal=True,
        sort=False
    )

    st.subheader("🧾 Sample Bracket Data")
    st.json(sample_bracket)