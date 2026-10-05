import pytest
from sqlalchemy import text

from models.projections import (
    MAX_STALENESS_WEEKS,
    MISSED_TEAM_GAMES_MULTIPLIER,
    _load_qb_starts,
    _load_recent_stats,
    _missed_team_games,
    _real_week_sequence,
    _staleness_reference_week,
)

# Regression coverage for the real bug this was built to fix: Christian
# McCaffrey's 2024 week-8 as-of projection was being built entirely from
# late-2023 games (~30 real weeks stale) with no signal anywhere that the
# data was that old, and on the live slate, Brandon Aiyuk (last real game
# 2024 week 7) was passing through with a real proj_ceiling of 20.72 built
# from 30-week-stale data. See models/projections.py's module comments.


def test_staleness_reference_week_live_mode():
    week_sequence = [(2025, 1), (2025, 2), (2025, 3), (2026, 1)]
    assert _staleness_reference_week(week_sequence, before=None) == (2026, 1)


def test_staleness_reference_week_backtest_mode_excludes_current_and_future():
    # before=(2025, 3) must land on the latest week STRICTLY earlier than that -
    # including week (2025, 3) itself would be lookahead bias in a backtest.
    week_sequence = [(2025, 1), (2025, 2), (2025, 3), (2026, 1)]
    assert _staleness_reference_week(week_sequence, before=(2025, 3)) == (2025, 2)


def test_staleness_reference_week_no_data():
    assert _staleness_reference_week([], before=None) is None


TEST_STALE_ID = "TEST_STALE_REGRESSION_PLAYER"
TEST_FRESH_ID = "TEST_FRESH_REGRESSION_PLAYER"


def _insert_test_stat_row(conn, player_id, season, week, points):
    conn.execute(
        text(
            """
            INSERT INTO player_weekly_stats (player_id, player_name, position, team, season, week, fantasy_points_ppr)
            VALUES (:player_id, :player_name, 'WR', 'ZZ', :season, :week, :points)
            """
        ),
        {"player_id": player_id, "player_name": player_id, "season": season, "week": week, "points": points},
    )


def test_load_recent_stats_drops_a_player_stale_beyond_the_threshold(engine):
    # Uses real weeks already in the table (not hardcoded season/week numbers)
    # so this stays correct as real data keeps accumulating over time.
    week_sequence = _real_week_sequence(engine)
    assert len(week_sequence) > MAX_STALENESS_WEEKS + 2, "not enough real week history in this DB to run this test"

    reference_week = week_sequence[-1]
    stale_week = week_sequence[-(MAX_STALENESS_WEEKS + 2)]  # guaranteed more than MAX_STALENESS_WEEKS behind

    try:
        with engine.begin() as conn:
            _insert_test_stat_row(conn, TEST_STALE_ID, *stale_week, 12.0)
            _insert_test_stat_row(conn, TEST_FRESH_ID, *reference_week, 12.0)

        history = _load_recent_stats([TEST_STALE_ID, TEST_FRESH_ID], engine)

        assert TEST_STALE_ID not in history, (
            "a player whose most recent game is more than MAX_STALENESS_WEEKS stale must be dropped "
            "entirely, not silently projected from old data"
        )
        assert TEST_FRESH_ID in history
    finally:
        with engine.begin() as conn:
            conn.execute(
                text("DELETE FROM player_weekly_stats WHERE player_id IN (:a, :b)"),
                {"a": TEST_STALE_ID, "b": TEST_FRESH_ID},
            )


# --- missed team games (MISSED_TEAM_GAMES_MULTIPLIER) ----------------------
# Real players/weeks from the dev DB's 2025-2026 player_weekly_stats. Jalen
# McMillan (TB) last played 2025 week 18 and recorded nothing in 2026 weeks
# 1-2, yet was projected from his 2025 games at full strength - the case
# this discount exists for.
MCMILLAN, EGBUKA, ZAY, EVANS = "00-0039855", "00-0040129", "00-0039064", "00-0031408"


def test_missed_team_games_counts_consecutive_team_games_sat_out(engine):
    teams = {MCMILLAN: "TB", EGBUKA: "TB", ZAY: "BAL", EVANS: "SF"}
    before_week2 = _missed_team_games(teams, engine, before=(2026, 2))
    before_week3 = _missed_team_games(teams, engine, before=(2026, 3))

    assert before_week2[MCMILLAN] == 1
    assert before_week3[MCMILLAN] == 2
    assert before_week3[EGBUKA] == 0
    assert before_week2[ZAY] == 0  # played week 1
    assert before_week3[ZAY] == 1  # sat out week 2
    # Counted against his CURRENT team (traded TB -> SF, played both 2026
    # weeks), not the old team's schedule.
    assert before_week3[EVANS] == 0


def test_missed_team_games_multiplier_shrinks_with_more_games_missed():
    values = [MISSED_TEAM_GAMES_MULTIPLIER[k] for k in sorted(MISSED_TEAM_GAMES_MULTIPLIER)]
    assert sorted(MISSED_TEAM_GAMES_MULTIPLIER) == [1, 2, 3, 4]
    assert all(0 < v < 1 for v in values)
    assert values == sorted(values, reverse=True)


E2E_SLATE = "TEST_MISSED_GAMES_SLATE"


def _a_wr_currently_missing_team_games(engine):
    # Picked from live data rather than hardcoded: which WR is currently
    # sitting out changes every time a new week is loaded (McMillan, the
    # original pick, returned in 2026 week 3).
    with engine.connect() as conn:
        rows = conn.execute(text(
            """
            WITH last AS (
                SELECT DISTINCT ON (player_id) player_id, player_name, team, season, week
                FROM player_weekly_stats WHERE position = 'WR' AND season = 2026
                ORDER BY player_id, season DESC, week DESC
            )
            SELECT l.player_id, l.player_name, l.team FROM last l
            WHERE l.week < (SELECT MAX(week) FROM player_weekly_stats t WHERE t.team = l.team AND t.season = 2026)
            ORDER BY l.player_id
            """
        )).fetchall()
    for row in rows:
        if _missed_team_games({row.player_id: row.team}, engine).get(row.player_id, 0) >= 1:
            return row
    pytest.skip("no WR currently missing team games in the loaded data")


def test_generate_projections_discounts_a_player_who_missed_team_games(engine):
    # End-to-end on a throwaway slate (never a real one - rewriting a real
    # slate's stored projections would change historical data). No
    # game_time, so no Vegas adjustment muddies the comparison.
    from models.projections import _project_from_history, generate_projections

    sitter = _a_wr_currently_missing_team_games(engine)
    rows = [("E2E_SITTER", sitter.player_name, sitter.player_id, sitter.team), ("E2E_EGBUKA", "Emeka Egbuka", EGBUKA, "TB")]
    try:
        with engine.begin() as conn:
            for dk_id, name, _, team in rows:
                conn.execute(
                    text(
                        "INSERT INTO slate_player_pool (slate_id, player_id, name, position, salary, team, opponent) "
                        "VALUES (:s, :p, :n, 'WR', 5000, :t, 'ATL')"
                    ),
                    {"s": E2E_SLATE, "p": dk_id, "n": name, "t": team},
                )
        generate_projections(E2E_SLATE, engine=engine)

        history = _load_recent_stats([sitter.player_id, EGBUKA], engine)
        missed = _missed_team_games({sitter.player_id: sitter.team, EGBUKA: "TB"}, engine)
        with engine.connect() as conn:
            stored = dict(conn.execute(
                text("SELECT player_id, proj_median FROM projections WHERE slate_id = :s"), {"s": E2E_SLATE}
            ).fetchall())
            # A player with a running baseline is stored as the model/baseline
            # blend; the discount under test is applied to the model side,
            # which generate_projections records as source 'model'.
            stored.update(conn.execute(
                text("SELECT player_id, proj_median FROM projection_sources WHERE slate_id = :s AND source = 'model'"),
                {"s": E2E_SLATE},
            ).fetchall())

        assert missed[sitter.player_id] >= 1 and missed[EGBUKA] == 0
        if sitter.player_id not in history:
            pytest.skip(f"{sitter.player_name}'s history is past the staleness cutoff - nothing to discount")
        expected = _project_from_history(history[sitter.player_id], "WR")["proj_median"]
        expected *= MISSED_TEAM_GAMES_MULTIPLIER[min(missed[sitter.player_id], 4)]
        assert float(stored["E2E_SITTER"]) == pytest.approx(expected, abs=0.02)
        assert float(stored["E2E_EGBUKA"]) == pytest.approx(_project_from_history(history[EGBUKA], "WR")["proj_median"], abs=0.02)
    finally:
        with engine.begin() as conn:
            conn.execute(text("DELETE FROM projections WHERE slate_id = :s"), {"s": E2E_SLATE})
            conn.execute(text("DELETE FROM projection_sources WHERE slate_id = :s"), {"s": E2E_SLATE})
            conn.execute(text("DELETE FROM slate_player_pool WHERE slate_id = :s"), {"s": E2E_SLATE})


def test_promoted_backup_qb_is_projected_from_his_real_starts_only(engine):
    # Tyson Bagent (00-0038416): four 2023 starts at 96-100% snaps, plus
    # 1-16% relief cameos in 2024-2026 that must not count as starts.
    starts = _load_qb_starts({"00-0038416"}, engine, before=(2026, 3))
    assert starts["00-0038416"] == [7.68, 19.80, 13.18, 12.88]


def test_injury_returner_gets_the_lighter_discount_only_with_a_clean_report():
    import pandas as pd
    from models.projections import _injury_returners

    injuries = pd.DataFrame({
        "gsis_id": ["RET", "RET", "QSTN", "QSTN", "QSTN", "ROLE", "LONG", "LONG", "LONG", "RET", "NODESIG", "NODESIG"],
        "week": [2, 3, 2, 3, 4, 3, 1, 2, 3, 4, 2, 4],
        "report_status": ["Out", "Out", "Out", "Doubtful", "Questionable", None, "Out", "Out", "Out", None, "Out", None],
    })
    missed = {
        "RET": [(2026, 2), (2026, 3)],      # Out both weeks, on report week 4 with no designation -> returner
        "QSTN": [(2026, 2), (2026, 3)],     # still Questionable this week -> general discount
        "ROLE": [(2026, 3)],                # missed with no injury listed (role loss) -> general discount
        "LONG": [(2026, 1), (2026, 2), (2026, 3)],  # 3 missed games -> beyond the returner window
        "NODESIG": [(2026, 2), (2026, 3)],  # week 3 absence not on the report -> not every absence was injury
        "OLD": [(2025, 18)],                # absence spans seasons -> not an in-season injury
    }
    assert _injury_returners(missed, injuries, 2026, 4) == {"RET"}
    # Listed Questionable, then sat: still an injury absence (the Puka Nacua case).
    puka = pd.DataFrame({"gsis_id": ["PUKA"] * 3, "week": [2, 3, 4], "report_status": ["Questionable", "Doubtful", None],
                         "practice_status": ["Did Not Participate In Practice"] * 2 + ["Full Participation in Practice"]})
    assert _injury_returners({"PUKA": [(2026, 2), (2026, 3)]}, puka, 2026, 4) == {"PUKA"}
    # Questionable but practicing in full, then sat: not an injury absence (the Jalen McMillan case).
    mcm = pd.DataFrame({"gsis_id": ["MCM"] * 3, "week": [1, 2, 3], "report_status": ["Doubtful", "Questionable", None],
                        "practice_status": ["Limited Participation in Practice", "Full Participation in Practice", "Full Participation in Practice"]})
    assert _injury_returners({"MCM": [(2026, 1), (2026, 2)]}, mcm, 2026, 3) == set()


def test_missed_team_games_counts_match_missed_team_weeks(engine):
    from models.projections import _missed_team_games, _missed_team_weeks

    with engine.connect() as conn:
        rows = conn.execute(text(
            "SELECT DISTINCT player_id, team FROM player_weekly_stats WHERE season = 2026 AND position = 'WR' LIMIT 40"
        )).fetchall()
    team_by_gsis = {r.player_id: r.team for r in rows}
    weeks = _missed_team_weeks(team_by_gsis, engine)
    assert _missed_team_games(team_by_gsis, engine) == {g: len(w) for g, w in weeks.items()}
