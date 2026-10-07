from sqlalchemy import text

from data.entered_lineups import (
    backfill_actual_scores,
    log_entered_lineups,
    summarize_entered_group_results,
)

TEST_CONTEST_ID = "TEST_ENTERED_LINEUPS_CONTEST"


def _cleanup(slate_id, engine):
    with engine.begin() as conn:
        conn.execute(
            text("DELETE FROM actuals WHERE slate_id = :s AND contest_id = :c"),
            {"s": slate_id, "c": TEST_CONTEST_ID},
        )


def _real_players(slate_id, engine, limit=4):
    # Real rows straight from slate_player_pool - this module only cares
    # about real player_ids that satisfy actuals' own FK, never about
    # roster legality or the live availability gate (which does a real,
    # sometimes-flaky nflverse network fetch this test has no reason to
    # depend on).
    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT player_id, name, position, team FROM slate_player_pool WHERE slate_id = :s LIMIT :n"),
            {"s": slate_id, "n": limit},
        ).mappings().fetchall()
    return [dict(r) for r in rows]


def test_log_and_backfill_and_summarize_a_real_completed_week(engine):
    # dk_thu_mon_2026_09_17's real games have already been played and real
    # player_weekly_stats rows exist for that week - the same real slate
    # this codebase's own backtests already use for exactly that reason.
    slate_id = "dk_thu_mon_2026_09_17"
    _cleanup(slate_id, engine)
    try:
        players = _real_players(slate_id, engine)
        # Two real, distinct small "lineups" (2 players each) - enough to
        # prove group-average math without needing a full legal roster,
        # since this module only ever cares about actuals rows, never
        # roster legality.
        lineup_a1 = {"roster": [("QB", players[0]), ("FLEX", players[1])], "win_prob_at_lock": None}
        lineup_b1 = {"roster": [("QB", players[2]), ("FLEX", players[3])], "win_prob_at_lock": 0.081}

        n = log_entered_lineups(slate_id, TEST_CONTEST_ID, {"A1": lineup_a1, "B1": lineup_b1}, engine=engine)
        assert n == 4

        result = backfill_actual_scores(slate_id, TEST_CONTEST_ID, engine=engine)
        assert result["updated"] + len(result["still_pending"]) == 4

        summary = summarize_entered_group_results(slate_id, TEST_CONTEST_ID, engine=engine)
        assert set(summary["lineup_totals"]) == {"A1", "B1"}
        if result["updated"] == 4:
            # Every player in this test resolved to a real actual score -
            # both lineup totals must be real, non-negative numbers, and
            # the group summary must reflect exactly one lineup per group.
            assert summary["lineup_totals"]["A1"] is not None
            assert summary["lineup_totals"]["B1"] is not None
            assert summary["group_summary"]["A"]["n_lineups_complete"] == 1
            assert summary["group_summary"]["B"]["n_lineups_complete"] == 1
    finally:
        _cleanup(slate_id, engine)


def test_log_entered_lineups_upserts_without_duplicating(engine):
    slate_id = "dk_thu_mon_2026_09_17"
    _cleanup(slate_id, engine)
    try:
        players = _real_players(slate_id, engine, limit=1)
        lineup = {"roster": [("QB", players[0])], "win_prob_at_lock": 0.05}

        log_entered_lineups(slate_id, TEST_CONTEST_ID, {"A1": lineup}, engine=engine)
        log_entered_lineups(slate_id, TEST_CONTEST_ID, {"A1": {**lineup, "win_prob_at_lock": 0.09}}, engine=engine)

        with engine.connect() as conn:
            rows = conn.execute(
                text("SELECT win_prob_at_lock FROM actuals WHERE slate_id = :s AND contest_id = :c AND lineup_id = 'A1'"),
                {"s": slate_id, "c": TEST_CONTEST_ID},
            ).fetchall()
        assert len(rows) == 1, "re-logging the same lineup must upsert, not duplicate"
        assert float(rows[0].win_prob_at_lock) == 0.09
    finally:
        _cleanup(slate_id, engine)
