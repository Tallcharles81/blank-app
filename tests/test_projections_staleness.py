from sqlalchemy import text

from models.projections import MAX_STALENESS_WEEKS, _load_recent_stats, _real_week_sequence, _staleness_reference_week

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
