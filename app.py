import streamlit as st
import pandas as pd
from simulation import run_simulation

st.set_page_config(page_title="March Madness Predictor", layout="wide")

st.title("🏀 March Madness Predictor")
st.write("Simulate the NCAA Tournament and view predicted regional brackets.")

num_sims = st.slider("Number of Simulations", 100, 5000, 1000, 100)


def render_probability_dashboard(probs, num_sims):
    st.subheader("🏆 Champion Probabilities")

    if not probs:
        st.warning("No probabilities were generated.")
        return

    prob_df = pd.DataFrame(list(probs.items()), columns=["Team", "Probability"])
    prob_df["Win %"] = (prob_df["Probability"] * 100).round(2)
    prob_df = prob_df.drop(columns=["Probability"])

    top_team = prob_df.iloc[0]

    col1, col2, col3 = st.columns(3)
    col1.metric("Most Likely Champion", top_team["Team"])
    col2.metric("Title Chance", f"{top_team['Win %']}%")
    col3.metric("Simulations Run", f"{num_sims:,}")

    st.bar_chart(prob_df.set_index("Team"))

    with st.expander("View probability table"):
        st.dataframe(prob_df, use_container_width=True)


def render_region(region_name, region_data):
    st.markdown(f"### {region_name}")

    rounds = [
        ("R64_matchups", "Round of 64"),
        ("R32_matchups", "Round of 32"),
        ("S16_matchups", "Sweet 16"),
        ("E8_matchups", "Elite 8"),
    ]

    cols = st.columns(4)

    for col, (round_key, round_label) in zip(cols, rounds):
        with col:
            st.markdown(f"**{round_label}**")

            for t1, t2 in region_data[round_key]:
                st.container(border=True).markdown(
                    f"""
                    **{t1}**  
                    vs  
                    **{t2}**
                    """
                )

    st.success(f"🏆 {region_name} Champion: {region_data['REGION_CHAMP']}")


def render_final_four(bracket):
    st.markdown("## 🔥 Final Four")

    for t1, t2 in bracket["Final Four_matchups"]:
        st.container(border=True).markdown(
            f"""
            **{t1}**  
            vs  
            **{t2}**
            """
        )

    st.markdown("## 🏀 Championship")

    for t1, t2 in bracket["Championship_matchup"]:
        st.container(border=True).markdown(
            f"""
            **{t1}**  
            vs  
            **{t2}**
            """
        )

    st.success(f"🏆 National Champion: {bracket['CHAMP']}")


if st.button("Run Simulation"):
    with st.spinner("Running simulations..."):
        probs, bracket = run_simulation("bracket_2025_round1.csv", num_sims)

    if bracket is None:
        st.error("Simulation failed. Check that all team names match your dataset.")
        st.stop()

    st.success("Simulation complete!")

    render_probability_dashboard(probs, num_sims)

    st.markdown("---")
    st.markdown("## 📊 Regional Brackets")

    regions = ["East", "West", "South", "Midwest"]
    tabs = st.tabs(regions)

    for tab, region in zip(tabs, regions):
        with tab:
            render_region(region, bracket["regions"][region])

    st.markdown("---")
    render_final_four(bracket)