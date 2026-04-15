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
# UI SECTION
# ==============================

# Title
st.title("🏀 March Madness Predictor")

# Description
st.markdown(
    "Run Monte Carlo simulations to predict which teams have the highest chance of winning the NCAA March Madness tournament."
)

# Input Controls
st.subheader("Simulation Settings")

num_sims = st.slider(
    "Number of Simulations",
    min_value=100,
    max_value=5000,
    value=1000,
    step=100
)

# Run Button
run_button = st.button("Run Simulation")

# ==============================
# OUTPUT SECTION
# ==============================

if run_button:

    st.write(f"Running {num_sims} simulations...")

    with st.spinner("Simulating tournament..."):
        results = run_simulation(num_sims)

    st.success("Simulation complete!")

    # Convert results to DataFrame
    df = pd.DataFrame(
        list(results.items()),
        columns=["Team", "Probability"]
    )

    # Format probabilities
    df["Probability"] = df["Probability"] * 100
    df = df.sort_values("Probability", ascending=False)

    # ⭐ NEW: Show most likely champion
    st.subheader("🥇 Most Likely Champion")
    st.write(f"{df.iloc[0]['Team']} ({df.iloc[0]['Probability']:.2f}%)")

    # Display results
    st.subheader("🏆 Champion Probabilities")
    st.dataframe(df)

    # Bar chart
    st.bar_chart(df.set_index("Team"))