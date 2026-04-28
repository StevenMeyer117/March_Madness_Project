import streamlit as st
import pandas as pd
from simulation import run_simulation

<<<<<<< HEAD
# ==============================
# CONFIG
# ==============================

st.set_page_config(
    page_title="March Madness Predictor",
    page_icon="🏀",
    layout="wide"
)
=======
st.set_page_config(page_title="March Madness Predictor", layout="wide")
>>>>>>> origin/Steven

st.title("🏀 March Madness Predictor")
st.write("Simulate the NCAA Tournament and view predicted brackets.")

<<<<<<< HEAD
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
=======
num_sims = st.slider("Number of Simulations", 100, 5000, 1000, 100)

if st.button("Run Simulation"):

    with st.spinner("Running simulations..."):
        probs, bracket = run_simulation("bracket_2025_round1.csv", num_sims)

    st.success("Simulation complete!")

    # ==============================
    # PROBABILITIES
    # ==============================

    st.subheader("🏆 Champion Probabilities")

    prob_df = pd.DataFrame(list(probs.items()), columns=["Team", "Probability"])
    prob_df["Probability"] = (prob_df["Probability"] * 100).round(2)
    prob_df.rename(columns={"Probability": "Win %"}, inplace=True)

    st.dataframe(prob_df)
    st.bar_chart(prob_df.set_index("Team"))

    # ==============================
    # DISPLAY FUNCTION
    # ==============================

    def display_matchups(title, matchups):
        st.markdown(f"### {title}")
        for t1, t2 in matchups:
            col1, col2, col3 = st.columns([3,1,3])
            col1.write(t1)
            col2.write("vs")
            col3.write(t2)

    # ==============================
    # BRACKET DISPLAY
    # ==============================

    st.subheader("📊 Predicted Tournament Flow")

    display_matchups("Round of 64 (Actual)", bracket["R64_matchups"])
    display_matchups("Round of 32", bracket["R32_matchups"])
    display_matchups("Sweet 16", bracket["S16_matchups"])
    display_matchups("Elite 8", bracket["E8_matchups"])
    display_matchups("Final Four", bracket["F4_matchups"])
    display_matchups("Championship", bracket["CHAMP_matchup"])

    st.markdown("## 🏆 Champion")
    st.success(bracket["CHAMP"])
>>>>>>> origin/Steven
