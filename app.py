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
# TITLE
# ==============================

st.title("🏀 March Madness Predictor")

st.write(
    "Run Monte Carlo simulations to predict NCAA Tournament champion probabilities."
)

# ==============================
# INPUT
# ==============================

st.subheader("Simulation Settings")

num_sims = st.slider(
    "Number of Simulations",
    min_value=100,
    max_value=5000,
    value=1000,
    step=100
)

# ==============================
# RUN BUTTON
# ==============================

if st.button("Run Simulation"):

    with st.spinner("Running simulations..."):

        results = run_simulation(num_sims)

    st.success("Simulation complete!")

    # ==============================
    # RESULTS
    # ==============================

    st.subheader("🏆 Champion Probabilities")

    df = pd.DataFrame(
        list(results.items()),
        columns=["Team", "Probability"]
    )

    df["Probability"] = df["Probability"] * 100
    df = df.sort_values("Probability", ascending=False)

    st.dataframe(df)
    st.bar_chart(df.set_index("Team"))