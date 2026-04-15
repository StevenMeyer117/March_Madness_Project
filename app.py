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
<<<<<<< HEAD
# UI SECTION
# ==============================

# Title
st.title("🏀 March Madness Predictor")

# Description
st.markdown(
    "Run Monte Carlo simulations to predict which teams have the highest chance of winning the NCAA March Madness tournament."
)

# Input Controls
=======
# UI SECTION (Teammate 1)
# ==============================

st.title("🏀 March Madness Predictor")

st.write(
    "Run Monte Carlo simulations to estimate NCAA Tournament champion probabilities."
)

>>>>>>> origin/main
st.subheader("Simulation Settings")

num_sims = st.slider(
    "Number of Simulations",
    min_value=100,
    max_value=5000,
    value=1000,
    step=100
)

<<<<<<< HEAD
# Run Button
run_button = st.button("Run Simulation")

# ==============================
# OUTPUT SECTION
=======
run_button = st.button("Run Simulation")

# ==============================
# DATA SECTION (DO NOT TOUCH)
>>>>>>> origin/main
# ==============================

if run_button:

<<<<<<< HEAD
    st.write(f"Running {num_sims} simulations...")

    with st.spinner("Simulating tournament..."):
=======
    with st.spinner("Running simulations..."):
>>>>>>> origin/main
        results = run_simulation(num_sims)

    st.success("Simulation complete!")

<<<<<<< HEAD
    # Convert results to DataFrame
=======
    # ==============================
    # VISUALIZATION SECTION (Teammate 2)
    # ==============================

    st.subheader("🏆 Champion Probabilities")

>>>>>>> origin/main
    df = pd.DataFrame(
        list(results.items()),
        columns=["Team", "Probability"]
    )

<<<<<<< HEAD
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
=======
    df["Probability"] = df["Probability"] * 100
    df = df.sort_values("Probability", ascending=False)

    st.dataframe(df)

>>>>>>> origin/main
    st.bar_chart(df.set_index("Team"))