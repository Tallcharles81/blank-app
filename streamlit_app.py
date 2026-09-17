import io

import pandas as pd
import streamlit as st

from optimizer import (
    ROSTER_SLOTS,
    SALARY_CAP,
    generate_lineups,
    load_player_pool,
)

st.set_page_config(page_title="NBA DK Lineup Optimizer", page_icon="🏀", layout="wide")

st.title("🏀 NBA DraftKings Lineup Optimizer")
st.write(
    "Upload a DraftKings NBA Classic player-pool CSV (or use the bundled sample "
    "slate) and build optimal, salary-cap-legal lineups: PG / SG / SF / PF / C / "
    "G / F / UTIL, $50,000 cap, players from at least two different games."
)

with st.sidebar:
    st.header("1. Player pool")
    uploaded = st.file_uploader("DraftKings CSV export", type="csv")
    use_sample = st.checkbox("Use bundled sample slate", value=uploaded is None)

    st.header("2. Lineups to generate")
    n_lineups = st.number_input("Number of lineups", min_value=1, max_value=20, value=1)
    max_overlap = st.slider(
        "Max shared players between lineups",
        min_value=0,
        max_value=7,
        value=5,
        help="Lower values force more distinct lineups across your set.",
    )

raw_df = None
source_error = None
if uploaded is not None:
    try:
        raw_df = pd.read_csv(uploaded)
    except Exception as exc:  # noqa: BLE001
        source_error = f"Could not read uploaded CSV: {exc}"
elif use_sample:
    raw_df = pd.read_csv("sample_data.csv")

if source_error:
    st.error(source_error)
    st.stop()

if raw_df is None:
    st.info("Upload a CSV or check 'Use bundled sample slate' to get started.")
    st.stop()

try:
    pool = load_player_pool(raw_df)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

if pool.empty:
    st.error("No valid players found after parsing the CSV.")
    st.stop()

st.header("Player pool")
st.caption(
    "Edit the Projection column to tune your own rankings, then check the boxes "
    "on the right to lock a player into every lineup or exclude them entirely."
)

display_df = pool[["Name", "Team", "Position", "Salary", "Projection"]].copy()
display_df.insert(0, "Lock", False)
display_df.insert(1, "Exclude", False)

edited = st.data_editor(
    display_df,
    column_config={
        "Salary": st.column_config.NumberColumn(format="$%d"),
        "Projection": st.column_config.NumberColumn(format="%.1f"),
        "Lock": st.column_config.CheckboxColumn(),
        "Exclude": st.column_config.CheckboxColumn(),
    },
    disabled=["Name", "Team", "Position", "Salary"],
    hide_index=True,
    use_container_width=True,
    height=420,
)

pool = pool.copy()
pool["Projection"] = edited["Projection"].values
locked_ids = set(pool.loc[edited["Lock"].values, "PlayerId"])
excluded_ids = set(pool.loc[edited["Exclude"].values, "PlayerId"])

overlap_conflicts = locked_ids & excluded_ids
if overlap_conflicts:
    st.error("A player can't be both locked and excluded at the same time.")
    st.stop()

if len(locked_ids) > len(ROSTER_SLOTS):
    st.error(f"You can lock at most {len(ROSTER_SLOTS)} players.")
    st.stop()

st.header("Generate lineups")
if st.button("Optimize", type="primary"):
    with st.spinner("Solving..."):
        lineups = generate_lineups(
            pool,
            n_lineups=int(n_lineups),
            salary_cap=SALARY_CAP,
            locked_ids=locked_ids,
            excluded_ids=excluded_ids,
            max_overlap=int(max_overlap),
        )

    if not lineups:
        st.error(
            "No feasible lineup found. Try unlocking/un-excluding players, "
            "raising the shared-player limit, or check that the pool has "
            "enough eligible players at every position."
        )
    else:
        if len(lineups) < n_lineups:
            st.warning(
                f"Only found {len(lineups)} of {n_lineups} requested lineups "
                "given the diversity and constraint settings."
            )

        export_frames = []
        for i, lineup in enumerate(lineups, start=1):
            st.subheader(f"Lineup {i}")
            col1, col2 = st.columns(2)
            col1.metric("Projected points", f"{lineup.total_projection:.1f}")
            col2.metric("Salary used", f"${lineup.total_salary:,} / ${SALARY_CAP:,}")

            lineup_df = lineup.as_dataframe()
            st.dataframe(lineup_df, hide_index=True, use_container_width=True)

            export_df = lineup_df.copy()
            export_df.insert(0, "Lineup", i)
            export_frames.append(export_df)

        all_lineups_df = pd.concat(export_frames, ignore_index=True)
        csv_buffer = io.StringIO()
        all_lineups_df.to_csv(csv_buffer, index=False)
        st.download_button(
            "Download all lineups as CSV",
            data=csv_buffer.getvalue(),
            file_name="dk_nba_lineups.csv",
            mime="text/csv",
        )
