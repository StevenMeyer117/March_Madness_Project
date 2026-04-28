import streamlit as st
import pandas as pd
from simulation import run_simulation

# ==============================
# CONFIG
# ==============================

st.set_page_config(
    page_title="March Madness Predictor",
    page_icon="🏀",
    layout="wide"
)

st.title("🏀 March Madness Predictor")

st.markdown("Monte Carlo NCAA tournament simulation using ML.")

# ==============================
# INPUT
# ==============================

num_sims = st.slider("Simulations", 100, 5000, 1000, 100)

run = st.button("Run Simulation")

# ==============================
# RUN
# ==============================

if run:

    probs, bracket = run_simulation(num_sims)

    st.success("Simulation complete!")

    # PROBABILITIES
    df = pd.DataFrame(list(probs.items()), columns=["Team", "Prob"])
    df["Prob"] *= 100
    df = df.sort_values("Prob", ascending=False)

    st.subheader("🏆 Champion Odds")
    st.dataframe(df)
    st.bar_chart(df.set_index("Team"))

    # BRACKET
    st.subheader("🏀 Tournament Bracket")

    for round_name, games in bracket.items():

        st.markdown(f"### {round_name}")

        for game in games:

            st.markdown(
                f"""
                <div style="
                    padding:10px;
                    margin:6px 0;
                    border-radius:10px;
                    background:#f4f4f4;
                    color:#000;
                ">
                    {game}
                </div>
                """,
                unsafe_allow_html=True
            )