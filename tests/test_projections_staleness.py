import pytest
from sqlalchemy import text

from models.projections import (
    MAX_STALENESS_WEEKS,
    MISSED_TEAM_GAMES_MULTIPLIER,
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


def test_generate_projections_discounts_a_player_who_missed_team_games(engine):
    # End-to-end on a throwaway slate (never a real one - rewriting a real
    # slate's stored projections would change historical data). No
    # game_time, so no Vegas adjustment muddies the comparison.
    from models.projections import _project_from_history, generate_projections

    rows = [("E2E_MCMILLAN", "Jalen McMillan", MCMILLAN), ("E2E_EGBUKA", "Emeka Egbuka", EGBUKA)]
    try:
        with engine.begin() as conn:
            for dk_id, name, _ in rows:
                conn.execute(
                    text(
                        "INSERT INTO slate_player_pool (slate_id, player_id, name, position, salary, team, opponent) "
                        "VALUES (:s, :p, :n, 'WR', 5000, 'TB', 'ATL')"
                    ),
                    {"s": E2E_SLATE, "p": dk_id, "n": name},
                )
        generate_projections(E2E_SLATE, engine=engine)

        history = _load_recent_stats([MCMILLAN, EGBUKA], engine)
        missed = _missed_team_games({MCMILLAN: "TB", EGBUKA: "TB"}, engine)
        with engine.connect() as conn:
            stored = dict(conn.execute(
                text("SELECT player_id, proj_median FROM projections WHERE slate_id = :s"), {"s": E2E_SLATE}
            ).fetchall())

        assert missed[MCMILLAN] >= 1 and missed[EGBUKA] == 0
        expected_mcmillan = _project_from_history(history[MCMILLAN], "WR")["proj_median"]
        expected_mcmillan *= MISSED_TEAM_GAMES_MULTIPLIER[min(missed[MCMILLAN], 4)]
        assert float(stored["E2E_MCMILLAN"]) == pytest.approx(expected_mcmillan, abs=0.02)
        assert float(stored["E2E_EGBUKA"]) == pytest.approx(_project_from_history(history[EGBUKA], "WR")["proj_median"], abs=0.02)
    finally:
        with engine.begin() as conn:
            conn.execute(text("DELETE FROM projections WHERE slate_id = :s"), {"s": E2E_SLATE})
            conn.execute(text("DELETE FROM slate_player_pool WHERE slate_id = :s"), {"s": E2E_SLATE})
