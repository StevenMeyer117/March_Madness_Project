import streamlit as st
import pandas as pd
import altair as alt
import matplotlib.pyplot as plt
import numpy as np
from simulation import run_simulation

@st.cache_data
def load_cbb():
    return pd.read_csv("cbb2_prepared.csv")

cbb = load_cbb()

st.set_page_config(page_title="March Madness Predictor", layout="wide")

st.title("🏀 March Madness Predictor")
st.write("Simulate the NCAA Tournament and view predicted brackets.")

num_sims = st.slider("Number of Simulations", 100, 5000, 1000, 100)

if "probs" not in st.session_state:
    st.session_state.probs = None

if "sample_bracket" not in st.session_state:
    st.session_state.sample_bracket = None

if "r64_win_probs" not in st.session_state:
    st.session_state.r64_win_probs = None
    
if st.button("Run Simulation"):
    with st.spinner("Running simulations..."):
        probs, sample_bracket, r64_win_probs = run_simulation("bracket_2025_round1.csv", num_sims)

    st.session_state.probs = probs
    st.session_state.sample_bracket = sample_bracket
    st.session_state.r64_win_probs = r64_win_probs

    st.success("Simulation complete!")

    # ==============================
    # PROBABILITIES
    # ==============================
if st.session_state.probs is not None:
    probs = st.session_state.probs
    sample_bracket = st.session_state.sample_bracket
    r64_win_probs = st.session_state.r64_win_probs
    
    st.subheader("🏆 Champion Probabilities")

    prob_df = pd.DataFrame(list(probs.items()), columns=["Team", "Probability"])
    prob_df["Probability"] = (prob_df["Probability"] * 100).round(2)
    prob_df.rename(columns={"Probability": "Win %"}, inplace=True)

    st.dataframe(prob_df)
    chart_df = pd.DataFrame(
        list(probs.items()),
        columns=["Team", "Probability"]
    ).sort_values("Probability", ascending=False).reset_index(drop=True)

    chart = (
        alt.Chart(chart_df)
        .mark_bar()
        .encode(
            y=alt.Y("Team:N", sort=None, title="Team"),
            x=alt.X(
                "Probability:Q",
                title="Champion Probability",
                axis=alt.Axis(format="%")
            ),
            tooltip=[
                alt.Tooltip("Team:N", title="Team"),
                alt.Tooltip("Probability:Q", title="Probability", format=".1%")
            ]
        )
    )

    st.altair_chart(chart, use_container_width=True)

    
    st.subheader("🏀 First-Round Upset Probabilities (Seeds 11-16)")


    double_digit_seed_teams = [
        "Siena",
        "South Florida",
        "North Dakota St.",
        "Furman",
        "Northern Iowa",
        "Cal Baptist",
        "LIU",
        "High Point",
        "Hawaii",
        "Texas",
        "Kennesaw St.",
        "Queens",
        "Howard",
        "Miami OH",
        "Tennessee St",
        "Akron",
        "Hofstra",
        "Wright St.",
        "Prairie View A&M",
        "UMBC",
        "Troy",
        "McNeese St.",
        "VCU",
        "Penn",
        "Idaho",
        "Kennesaw St."
    ]

    upset_df = pd.DataFrame(
        [(team, r64_win_probs.get(team, 0)) for team in double_digit_seed_teams],
        columns=["Team", "UpsetProbability"]
    )

    upset_df = upset_df.sort_values("UpsetProbability", ascending=False).head(10).reset_index(drop=True)
    
    upset_chart = (
        alt.Chart(upset_df)
        .mark_bar()
        .encode(
            y=alt.Y("Team:N", sort=None, title="Team"),
            x=alt.X(
                "UpsetProbability:Q",
                title="First-Round Win Probability",
                axis=alt.Axis(format="%")
            ),
            tooltip=[
                alt.Tooltip("Team:N", title="Team"),
                alt.Tooltip("UpsetProbability:Q", title="Upset %", format=".1%")
            ]
        )
    )

    st.altair_chart(upset_chart, use_container_width=True)

    def show_postseason_relationships(cbb):
        st.subheader("📈 Team Metrics vs Postseason Success")

        if "POSTSEASON_NUM" not in cbb.columns:
            st.warning("POSTSEASON_NUM is not in this dataset, so this graph cannot be created.")
            return

        top_relationships = ["SEED", "BARTHAG", "WAB", "ADJOE", "ADJDE"]

        selected_metric = st.selectbox(
            "Choose a metric to compare",
            top_relationships
        )

        plot_df = cbb[[selected_metric, "POSTSEASON_NUM"]].copy()

        plot_df[selected_metric] = pd.to_numeric(plot_df[selected_metric], errors="coerce")
        plot_df["POSTSEASON_NUM"] = pd.to_numeric(plot_df["POSTSEASON_NUM"], errors="coerce")

        plot_df = plot_df.replace([np.inf, -np.inf], np.nan).dropna()

        if len(plot_df) < 2:
            st.warning("Not enough valid data points to plot this relationship.")
            return

        fig, ax = plt.subplots(figsize=(5, 3))

        x = plot_df[selected_metric]
        y = plot_df["POSTSEASON_NUM"]

        ax.scatter(x, y, alpha=0.35)

        if x.nunique() > 1:
            m, b = np.polyfit(x, y, 1)
            x_line = np.array([x.min(), x.max()])
            y_line = m * x_line + b
            ax.plot(x_line, y_line, linewidth=2)

        ax.set_title(f"{selected_metric} vs Postseason Success")
        ax.set_xlabel(selected_metric)
        ax.set_ylabel("Round Eliminated")

        fig.tight_layout()
        st.pyplot(fig, width="content")
    show_postseason_relationships(cbb)
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

    display_matchups("Round of 64 (Actual)", sample_bracket["R64_matchups"])
    display_matchups("Round of 32", sample_bracket["R32_matchups"])
    display_matchups("Sweet 16", sample_bracket["S16_matchups"])
    display_matchups("Elite 8", sample_bracket["E8_matchups"])
    display_matchups("Final Four", sample_bracket["F4_matchups"])
    display_matchups("Championship", sample_bracket["CHAMP_matchup"])

    st.markdown("## 🏆 Champion")
    st.success(sample_bracket["CHAMP"])