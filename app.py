import streamlit as st
import pandas as pd
import altair as alt
import matplotlib.pyplot as plt
import numpy as np
from simulation import run_simulation
from team_logos import TEAM_LOGOS

# ==============================
# LOAD DATA
# ==============================

@st.cache_data
def load_cbb():
    return pd.read_csv("cbb2_prepared.csv")


cbb = load_cbb()

# ==============================
# PAGE CONFIG
# ==============================

st.set_page_config(page_title="March Madness Predictor", layout="wide")

st.title("🏀 March Madness Predictor")
st.write("Simulate the NCAA Tournament and view predicted brackets.")

num_sims = st.slider("Number of Simulations", 100, 5000, 1000, 100)

# ==============================
# SESSION STATE
# ==============================

if "probs" not in st.session_state:
    st.session_state.probs = None

if "sample_bracket" not in st.session_state:
    st.session_state.sample_bracket = None

if "r64_win_probs" not in st.session_state:
    st.session_state.r64_win_probs = None


# ==============================
# TEAM LOGO HELPERS
# ==============================

TEAM_NAME_ALIASES = {
    "miami fl": "miami",
    "miami (fl)": "miami",
    "saint marys": "saint marys",
    "saint mary's": "saint marys",
    "st marys": "st marys",
    "st mary's": "st marys",
    "north dakota st": "north dakota state",
    "north dakota st.": "north dakota state",
    "kennesaw st": "kennesaw state",
    "kennesaw st.": "kennesaw state",
    "tennessee st": "tennessee state",
    "tennessee st.": "tennessee state",
    "wright st": "wright state",
    "wright st.": "wright state",
    "utah st": "utah state",
    "utah st.": "utah state",
    "ohio st": "ohio state",
    "ohio st.": "ohio state",
    "michigan st": "michigan state",
    "michigan st.": "michigan state",
    "ca baptist": "cal baptist",
    "cal baptist": "cal baptist",
    "queens nc": "queens",
    "queens (nc)": "queens",
    "queens (n.c.)": "queens",
    "texas a&m": "texas am",
}

TEAM_NAME_ALIASES.update({
    "ohio st": "ohio state",
    "ohio st.": "ohio state",
    "st john's": "st johns",
    "st johns": "st johns",
    "ca baptist": "cal baptist",
    "north dakota st": "north dakota state",
    "north dakota st.": "north dakota state",
    "utah st": "utah state",
    "utah st.": "utah state",
    "mcneese": "mcneese",
    "mcneese st": "mcneese",
    "mcneese st.": "mcneese",
    "queens (n c)": "queens",
    "queens (n. c.)": "queens",
    "queens nc": "queens",
    "texas aandm": "texas am",
    "texas a&m": "texas am",
    "miami (ohio)": "miami ohio",
    "wright st": "wright state",
    "wright st.": "wright state",
    "tennessee st": "tennessee state",
    "tennessee st.": "tennessee state",
    "iowa st": "iowa state",
    "iowa st.": "iowa state",
    "louisville": "louisville",
    "south florida": "south florida",
})


def normalize_team_name(name):
    return (
        str(name)
        .strip()
        .lower()
        .replace(".", "")
        .replace("'", "")
        .replace("&", "and")
        .replace("(", "")
        .replace(")", "")
    )


def get_logo_url(team):
    key = normalize_team_name(team)
    key = TEAM_NAME_ALIASES.get(key, key)

    team_id = TEAM_LOGOS.get(key)

    if team_id:
        return f"https://a.espncdn.com/i/teamlogos/ncaa/500/{team_id}.png"

    return None


# ==============================
# WINNER HELPERS
# ==============================

def get_winners_for_round(region_data, round_key):
    if round_key == "R64_matchups":
        return {team for matchup in region_data["R32_matchups"] for team in matchup}

    if round_key == "R32_matchups":
        return {team for matchup in region_data["S16_matchups"] for team in matchup}

    if round_key == "S16_matchups":
        return {team for matchup in region_data["E8_matchups"] for team in matchup}

    if round_key == "E8_matchups":
        return {region_data["REGION_CHAMP"]}

    return set()


def is_winner(team, winners):
    normalized_team = normalize_team_name(team)
    normalized_winners = {normalize_team_name(winner) for winner in winners}
    return normalized_team in normalized_winners


def render_team(team, winners):
    winner = is_winner(team, winners)
    logo_url = get_logo_url(team)

    logo_col, team_col = st.columns([1, 5])

    with logo_col:
        if logo_url:
            st.image(logo_url, width=28)
        else:
            st.write("🏀")

    with team_col:
        if winner:
            st.success(f"🏆 {team}")
        else:
            st.write(f"**{team}**")


# ==============================
# RUN SIMULATION
# ==============================

if st.button("Run Simulation"):
    with st.spinner("Running simulations..."):
        probs, sample_bracket, r64_win_probs = run_simulation(
            "bracket_2025_round1.csv",
            num_sims
        )

    st.session_state.probs = probs
    st.session_state.sample_bracket = sample_bracket
    st.session_state.r64_win_probs = r64_win_probs

    st.success("Simulation complete!")


# ==============================
# BUILD REGIONAL BRACKET
# ==============================

def build_regional_bracket(sample_bracket):
    # Assumes bracket order stays consistent throughout:
    # East, West, South, Midwest
    region_order = ["East", "West", "South", "Midwest"]

    final_four_teams = []

    for t1, t2 in sample_bracket["F4_matchups"]:
        final_four_teams.extend([t1, t2])

    regions = {}

    for i, region in enumerate(region_order):
        regions[region] = {
            "R64_matchups": sample_bracket["R64_matchups"][i * 8:(i + 1) * 8],
            "R32_matchups": sample_bracket["R32_matchups"][i * 4:(i + 1) * 4],
            "S16_matchups": sample_bracket["S16_matchups"][i * 2:(i + 1) * 2],
            "E8_matchups": sample_bracket["E8_matchups"][i:i + 1],
            "REGION_CHAMP": final_four_teams[i] if i < len(final_four_teams) else None,
        }

    return {
        "regions": regions,
        "Final Four_matchups": sample_bracket["F4_matchups"],
        "Championship_matchup": sample_bracket["CHAMP_matchup"],
        "CHAMP": sample_bracket["CHAMP"],
    }


# ==============================
# REGION BRACKET DISPLAY
# ==============================

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
        winners = get_winners_for_round(region_data, round_key)

        with col:
            st.markdown(f"**{round_label}**")

            for t1, t2 in region_data[round_key]:
                with st.container(border=True):
                    render_team(t1, winners)
                    st.caption("vs")
                    render_team(t2, winners)

    st.success(f"🏆 {region_name} Champion: {region_data['REGION_CHAMP']}")


# ==============================
# FINAL FOUR DISPLAY
# ==============================

def render_final_four(bracket):
    st.markdown("## 🔥 Final Four")

    championship_teams = {
        team
        for matchup in bracket["Championship_matchup"]
        for team in matchup
    }

    cols = st.columns(2)

    for col, (t1, t2) in zip(cols, bracket["Final Four_matchups"]):
        with col:
            with st.container(border=True):
                st.markdown("### Final Four Matchup")
                render_team(t1, championship_teams)
                st.caption("vs")
                render_team(t2, championship_teams)

    st.markdown("## 🏀 Championship")

    champion = bracket["CHAMP"]

    for t1, t2 in bracket["Championship_matchup"]:
        with st.container(border=True):
            st.markdown("### National Championship")
            render_team(t1, {champion})
            st.caption("vs")
            render_team(t2, {champion})

    st.success(f"🏆 National Champion: {bracket['CHAMP']}")


# ==============================
# DISPLAY RESULTS
# ==============================

if st.session_state.probs is not None:
    probs = st.session_state.probs
    sample_bracket = st.session_state.sample_bracket
    r64_win_probs = st.session_state.r64_win_probs

    # ==============================
    # CHAMPION PROBABILITIES
    # ==============================

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

    # ==============================
    # FIRST-ROUND UPSET PROBABILITIES
    # ==============================

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

    upset_df = (
        upset_df
        .sort_values("UpsetProbability", ascending=False)
        .head(10)
        .reset_index(drop=True)
    )

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

    # ==============================
    # TEAM METRICS VS POSTSEASON SUCCESS
    # ==============================

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

        plot_df[selected_metric] = pd.to_numeric(
            plot_df[selected_metric],
            errors="coerce"
        )

        plot_df["POSTSEASON_NUM"] = pd.to_numeric(
            plot_df["POSTSEASON_NUM"],
            errors="coerce"
        )

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
    # BRACKET DISPLAY
    # ==============================

    st.markdown("---")
    st.markdown("## 📊 Regional Brackets")

    bracket = build_regional_bracket(sample_bracket)

    regions = ["East", "West", "South", "Midwest"]
    tabs = st.tabs(regions)

    for tab, region in zip(tabs, regions):
        with tab:
            render_region(region, bracket["regions"][region])

    st.markdown("---")
    render_final_four(bracket)