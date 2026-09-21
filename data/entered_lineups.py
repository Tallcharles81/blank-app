"""
Real live-entry tracking: log the actual lineups entered into a real DK
contest into the `actuals` table (see db/schema.sql - built with exactly
this shape, contest_id/lineup_id/win_prob_at_lock/result columns included,
but never wired to anything until now), then backfill each player's real
actual_score once nflverse publishes that week's results - so a live
comparison between two real construction approaches (e.g. the
deterministic ceiling pick vs. generate_simulation_selected_lineup) can be
scored against real outcomes, the live counterpart to models/
calibration.py::run_simulation_selection_backtest's retrospective version.
"""
from collections import defaultdict

from sqlalchemy import text

from data.player_availability import resolve_slate_season_week
from data.player_crosswalk import resolve_dk_players_to_gsis
from db.migrate import get_engine


def log_entered_lineups(slate_id, contest_id, lineups_by_id, engine=None):
    """lineups_by_id: {lineup_id: {"roster": [(slot, player_dict), ...],
    "win_prob_at_lock": float | None}}. Inserts one row per real player in
    every lineup into `actuals` - actual_score/result stay NULL until
    backfill_actual_scores runs after the real games finish. Re-running for
    the same (contest_id, lineup_id, player_id) upserts win_prob_at_lock
    rather than duplicating, using the table's own real UNIQUE constraint -
    safe to call again if a lineup gets edited before lock.
    """
    engine = engine or get_engine()
    rows = []
    for lineup_id, entry in lineups_by_id.items():
        win_prob = entry.get("win_prob_at_lock")
        for _, p in entry["roster"]:
            rows.append(
                {
                    "slate_id": slate_id,
                    "player_id": p["player_id"],
                    "contest_id": contest_id,
                    "lineup_id": lineup_id,
                    "win_prob_at_lock": win_prob,
                }
            )

    with engine.begin() as conn:
        for row in rows:
            conn.execute(
                text(
                    """
                    INSERT INTO actuals (slate_id, player_id, contest_id, lineup_id, win_prob_at_lock)
                    VALUES (:slate_id, :player_id, :contest_id, :lineup_id, :win_prob_at_lock)
                    ON CONFLICT (slate_id, player_id, contest_id, lineup_id) DO UPDATE SET
                        win_prob_at_lock = EXCLUDED.win_prob_at_lock
                    """
                ),
                row,
            )
    return len(rows)


def backfill_actual_scores(slate_id, contest_id, engine=None):
    """Fills in actuals.actual_score for every real player already logged
    under (slate_id, contest_id) whose actual_score is still NULL, using
    that week's real player_weekly_stats (resolve_slate_season_week + the
    same crosswalk every other real scoring path in this codebase uses -
    see models/backtest.py::load_actual_scores). Safe to call before real
    results are published: a player with no matching (season, week) row yet
    is left NULL (not zeroed) and reported in still_pending, so a later
    re-call just fills in whatever has newly appeared without ever
    fabricating a 0 for "hasn't been scored yet."

    Returns {"updated": n, "still_pending": [player_id, ...]}.
    """
    engine = engine or get_engine()
    season_week = resolve_slate_season_week(slate_id, engine)
    if season_week is None:
        raise RuntimeError(f"Could not resolve (season, week) for slate {slate_id}")
    season, week = season_week

    with engine.connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT DISTINCT a.player_id, sp.name, sp.position, sp.team
                FROM actuals a
                JOIN slate_player_pool sp ON sp.slate_id = a.slate_id AND sp.player_id = a.player_id
                WHERE a.slate_id = :slate_id AND a.contest_id = :contest_id AND a.actual_score IS NULL
                """
            ),
            {"slate_id": slate_id, "contest_id": contest_id},
        ).mappings().fetchall()
    if not rows:
        return {"updated": 0, "still_pending": []}

    dk_players = [dict(r) for r in rows]
    gsis_by_dk_id, _, _ = resolve_dk_players_to_gsis(dk_players, engine)

    ids = list(gsis_by_dk_id.values())
    with engine.connect() as conn:
        actual_rows = conn.execute(
            text(
                "SELECT player_id, fantasy_points_ppr FROM player_weekly_stats "
                "WHERE player_id = ANY(:ids) AND season = :season AND week = :week"
            ),
            {"ids": ids, "season": season, "week": week},
        ).fetchall()
    actual_by_gsis = {row.player_id: float(row.fantasy_points_ppr) for row in actual_rows}

    updated = 0
    still_pending = []
    with engine.begin() as conn:
        for p in dk_players:
            gsis_id = gsis_by_dk_id.get(p["player_id"])
            score = actual_by_gsis.get(gsis_id) if gsis_id else None
            if score is None:
                still_pending.append(p["player_id"])
                continue
            conn.execute(
                text(
                    "UPDATE actuals SET actual_score = :score "
                    "WHERE slate_id = :slate_id AND player_id = :player_id AND contest_id = :contest_id"
                ),
                {"score": score, "slate_id": slate_id, "player_id": p["player_id"], "contest_id": contest_id},
            )
            updated += 1

    return {"updated": updated, "still_pending": still_pending}


def summarize_entered_group_results(slate_id, contest_id, engine=None):
    """Per-lineup total actual_score (sum across its real roster - None if
    any of that lineup's real players is still pending a backfill) and a
    per-GROUP average, read off lineup_id's own leading character (the
    convention this codebase's live A/B entries use - "A1".."A10"/
    "B1".."B10"). A lineup with any still-pending player is EXCLUDED from
    its group's average rather than treated as a 0 - same real-vs-unscored
    distinction backfill_actual_scores itself makes.
    """
    engine = engine or get_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT lineup_id, player_id, actual_score FROM actuals WHERE slate_id = :s AND contest_id = :c"),
            {"s": slate_id, "c": contest_id},
        ).fetchall()

    by_lineup = defaultdict(list)
    for row in rows:
        by_lineup[row.lineup_id].append(row.actual_score)

    lineup_totals = {
        lineup_id: (None if any(s is None for s in scores) else round(sum(float(s) for s in scores), 2))
        for lineup_id, scores in by_lineup.items()
    }

    by_group = defaultdict(list)
    for lineup_id, total in lineup_totals.items():
        if total is not None:
            by_group[lineup_id[0]].append(total)

    group_summary = {
        group: {
            "n_lineups_complete": len(totals),
            "avg_total": round(sum(totals) / len(totals), 2) if totals else None,
            "best": round(max(totals), 2) if totals else None,
            "worst": round(min(totals), 2) if totals else None,
        }
        for group, totals in by_group.items()
    }

    return {"lineup_totals": lineup_totals, "group_summary": group_summary}
