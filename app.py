import streamlit as st
import pandas as pd
from simulation import run_simulation

st.set_page_config(page_title="March Madness Predictor", layout="wide")

st.title("🏀 March Madness Predictor")
st.write("Simulate the NCAA Tournament and view predicted brackets.")

num_sims = st.slider("Number of Simulations", 100, 5000, 1000, 100)

# ==============================
# BRACKET DRAW FUNCTION
# ==============================

def draw_bracket(bracket):
    st.subheader("📊 Predicted Tournament Flow")

    col1, col2, col3, col4, col5, col6 = st.columns(6)

    # Round of 64
    with col1:
        st.markdown("### Round of 64")
        for t1, t2 in bracket["R64_matchups"]:
            st.write(t1)
            st.write(t2)
            st.write("---")

    # Round of 32
    with col2:
        st.markdown("### Round of 32")
        for t1, t2 in bracket["R32_matchups"]:
            st.write(t1)
            st.write(t2)
            st.write("---")

    # Sweet 16
    with col3:
        st.markdown("### Sweet 16")
        for t1, t2 in bracket["S16_matchups"]:
            st.write(t1)
            st.write(t2)
            st.write("---")

    # Elite 8
    with col4:
        st.markdown("### Elite 8")
        for t1, t2 in bracket["E8_matchups"]:
            st.write(t1)
            st.write(t2)
            st.write("---")

    # Final Four
    with col5:
        st.markdown("### Final Four")
        for t1, t2 in bracket["F4_matchups"]:
            st.write(t1)
            st.write(t2)
            st.write("---")

    # Championship (SAFE VERSION)
    with col6:
        st.markdown("### Championship")

        champ_matchup = bracket.get("CHAMP_matchup", [])

        if isinstance(champ_matchup, (list, tuple)) and len(champ_matchup) == 2:
            t1, t2 = champ_matchup
            st.write(t1)
            st.write(t2)
        else:
            st.write("Final matchup not available")

        st.write("---")
        st.success(f"🏆 {bracket['CHAMP']}")

# ==============================
# MAIN BUTTON
# ==============================

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
    # BRACKET DISPLAY
    # ==============================

    draw_bracket(bracket)

    # ==============================
    # FINAL CHAMPION
    # ==============================

    st.markdown("## 🏆 Champion")
    st.success(bracket["CHAMP"])