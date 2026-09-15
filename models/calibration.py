import json
import random
from collections import defaultdict

from sqlalchemy import text

from data.player_crosswalk import resolve_dk_players_to_gsis
from db.migrate import get_engine
from models.backtest import DEFAULT_RANDOM_FIELD_SIZE, load_actual_scores, load_slate_pool
from models.optimizer import SALARY_CAP, build_lineups_from_pool
from models.projections import _load_recent_stats, _project_from_history

# The stored percentile ladder (models/projections.py's PERCENTILE_Z) has
# 10/25/50/75/90 but not 20 or 80 directly - both interpolated between their
# neighboring stored points the same way models/simulation.py already
# interpolates arbitrary percentiles from that ladder, rather than
# approximated by the nearest stored bucket.
_P20_LOWER = (0.10, "10")
_P20_UPPER = (0.25, "25")
_P80_LOWER = (0.75, "75")
_P80_UPPER = (0.90, "90")


def _interpolate(percentiles, target, lower, upper):
    x0, key0 = lower
    x1, key1 = upper
    y0, y1 = percentiles[key0], percentiles[key1]
    return y0 + (y1 - y0) * (target - x0) / (x1 - x0)


def _available_weeks(engine):
    query = text("SELECT DISTINCT season, week FROM player_weekly_stats ORDER BY season, week")
    with engine.connect() as conn:
        return [(row.season, row.week) for row in conn.execute(query)]


def _asof_projections(slate_players, gsis_by_dk_id, season, week, engine):
    history = _load_recent_stats(gsis_by_dk_id.values(), engine, before=(season, week))

    result = {}
    for player in slate_players:
        gsis_id = gsis_by_dk_id.get(player["player_id"])
        games = history.get(gsis_id) if gsis_id else None
        if not games:
            continue
        proj = _project_from_history(games, player["position"])
        percentiles = proj["proj_percentiles"]
        result[player["player_id"]] = {
            "median": proj["proj_median"],
            "p20": _interpolate(percentiles, 0.20, _P20_LOWER, _P20_UPPER),
            "p80": _interpolate(percentiles, 0.80, _P80_LOWER, _P80_UPPER),
        }
    return result


def _played_gsis_ids(gsis_ids, season, week, engine):
    # snap_pct > 0 is a much more precise "did they actually play" signal than
    # "scored exactly 0 points" - checked against real data first: of all
    # exact-zero-point rows, 98% also had zero recorded snaps, confirming this
    # correlates the way it should before relying on it.
    query = text(
        "SELECT player_id FROM player_weekly_stats "
        "WHERE player_id = ANY(:ids) AND season = :season AND week = :week AND snap_pct > 0"
    )
    with engine.connect() as conn:
        return {row.player_id for row in conn.execute(query, {"ids": list(gsis_ids), "season": season, "week": week})}


def run_weekly_calibration(source_slate_id, seasons=None, num_lineups=1, random_field_size=100, engine=None):
    """Backtest source_slate_id's real salaries against every available real
    historical week (or just `seasons` if given), one independent backtest per
    week. A week is skipped (not failed) if there's no overlap between
    as-of-projectable and actually-scored players for it - e.g. the very
    first available week has no prior history to project from at all.

    MAE and the p20/p50/p80 hit rates only count players who actually played
    that week (snap_pct > 0, or any DST - see _played_gsis_ids). A bye/
    inactive/injured week is a real, correct 0 for lineup-scoring purposes
    (see avg_field_percentile below, which does NOT apply this filter - a real
    lineup that rosters a player who goes inactive should be penalized for
    that), but it isn't a fair test of projection accuracy: the projection
    model has no injury/inactivity awareness, so comparing its "if they play"
    estimate against a "didn't play" actual isn't measuring the same thing.
    This was caught for real: MAE and hit rates without this filter looked
    biased, but excluding confirmed-DNP weeks alone moved the median hit rate
    from ~37% to ~53.5%, right around the 50% target.

    Returns (weekly_results, skipped_weeks). Each weekly_results entry has the
    per-week counts needed to pool correctly across weeks afterward (see
    summarize_calibration) - not pre-averaged.
    """
    engine = engine or get_engine()

    slate_players = load_slate_pool(source_slate_id, engine)
    if not slate_players:
        raise ValueError(f"No players found in slate_player_pool for slate {source_slate_id}")
    gsis_by_dk_id, _, _ = resolve_dk_players_to_gsis(slate_players, engine)

    weeks = _available_weeks(engine)
    if seasons is not None:
        weeks = [(s, w) for s, w in weeks if s in seasons]

    rng = random.Random()
    weekly_results = []
    skipped = []

    for season, week in weeks:
        asof = _asof_projections(slate_players, gsis_by_dk_id, season, week, engine)
        actual_points, _ = load_actual_scores(slate_players, season, week, engine)

        eligible_ids = set(asof) & set(actual_points)
        if not eligible_ids:
            skipped.append((season, week))
            continue

        eligible_players = [p for p in slate_players if p["player_id"] in eligible_ids]

        played_gsis_ids = _played_gsis_ids(gsis_by_dk_id.values(), season, week, engine)
        accuracy_players = [
            p
            for p in eligible_players
            if p["position"] == "DST" or gsis_by_dk_id.get(p["player_id"]) in played_gsis_ids
        ]

        abs_error_by_position = defaultdict(lambda: [0.0, 0])
        p20_hits = p50_hits = p80_hits = 0
        for p in accuracy_players:
            pid = p["player_id"]
            actual = actual_points[pid]
            error = abs(asof[pid]["median"] - actual)
            abs_error_by_position[p["position"]][0] += error
            abs_error_by_position[p["position"]][1] += 1

            if actual <= asof[pid]["p20"]:
                p20_hits += 1
            if actual >= asof[pid]["median"]:
                p50_hits += 1
            if actual >= asof[pid]["p80"]:
                p80_hits += 1
        opportunities = len(accuracy_players)

        asof_pool = [{**p, "points": asof[p["player_id"]]["median"]} for p in eligible_players]
        try:
            lineups, _ = build_lineups_from_pool(asof_pool, num_lineups=num_lineups, salary_cap=SALARY_CAP)
        except ValueError:
            skipped.append((season, week))
            continue

        def score_actual(roster):
            return sum(actual_points[p["player_id"]] for _, p in roster)

        field_scores = []
        for _ in range(random_field_size):
            randomized_pool = [{**p, "points": rng.random()} for p in eligible_players]
            try:
                field_lineup, _ = build_lineups_from_pool(randomized_pool, num_lineups=1, salary_cap=SALARY_CAP)
            except ValueError:
                continue
            field_scores.append(score_actual(field_lineup[0]["roster"]))

        lineup_percentiles = []
        for lu in lineups:
            actual = score_actual(lu["roster"])
            if field_scores:
                better_than = sum(1 for s in field_scores if actual > s)
                lineup_percentiles.append(better_than / len(field_scores))

        weekly_results.append(
            {
                "season": season,
                "week": week,
                "num_players_evaluated": len(eligible_ids),
                "mae_by_position": {
                    pos: {"mae": round(total / n, 4), "n": n} for pos, (total, n) in abs_error_by_position.items()
                },
                "p20_hits": p20_hits,
                "p20_opportunities": opportunities,
                "p50_hits": p50_hits,
                "p50_opportunities": opportunities,
                "p80_hits": p80_hits,
                "p80_opportunities": opportunities,
                "avg_field_percentile": (
                    round(sum(lineup_percentiles) / len(lineup_percentiles), 4) if lineup_percentiles else None
                ),
                "field_size": len(field_scores),
            }
        )

    return weekly_results, skipped


def store_calibration_results(weekly_results, engine=None):
    engine = engine or get_engine()
    upsert_sql = text(
        """
        INSERT INTO calibration_weekly
            (season, week, num_players_evaluated, mae_by_position,
             p20_hits, p20_opportunities, p50_hits, p50_opportunities, p80_hits, p80_opportunities,
             avg_field_percentile, field_size)
        VALUES
            (:season, :week, :num_players_evaluated, :mae_by_position,
             :p20_hits, :p20_opportunities, :p50_hits, :p50_opportunities, :p80_hits, :p80_opportunities,
             :avg_field_percentile, :field_size)
        ON CONFLICT (season, week) DO UPDATE SET
            num_players_evaluated = EXCLUDED.num_players_evaluated,
            mae_by_position = EXCLUDED.mae_by_position,
            p20_hits = EXCLUDED.p20_hits,
            p20_opportunities = EXCLUDED.p20_opportunities,
            p50_hits = EXCLUDED.p50_hits,
            p50_opportunities = EXCLUDED.p50_opportunities,
            p80_hits = EXCLUDED.p80_hits,
            p80_opportunities = EXCLUDED.p80_opportunities,
            avg_field_percentile = EXCLUDED.avg_field_percentile,
            field_size = EXCLUDED.field_size,
            computed_at = now()
        """
    )
    with engine.begin() as conn:
        for row in weekly_results:
            conn.execute(upsert_sql, {**row, "mae_by_position": json.dumps(row["mae_by_position"])})


def summarize_calibration(seasons=None, engine=None):
    """Pools stored calibration_weekly rows into the cross-week numbers: MAE by
    position (weighted by each week's player count, not a plain average of
    per-week MAEs), the overall p80 hit rate (total hits / total
    opportunities), and the average/worst/best week by field percentile.
    """
    engine = engine or get_engine()
    query = "SELECT * FROM calibration_weekly"
    params = {}
    if seasons is not None:
        query += " WHERE season = ANY(:seasons)"
        params["seasons"] = list(seasons)
    query += " ORDER BY season, week"

    with engine.connect() as conn:
        rows = conn.execute(text(query), params).mappings().fetchall()
    if not rows:
        raise ValueError("No calibration_weekly rows to summarize")

    pooled_error = defaultdict(lambda: [0.0, 0])
    totals = {"p20": [0, 0], "p50": [0, 0], "p80": [0, 0]}
    field_percentiles_by_week = []

    for row in rows:
        mae_by_position = row["mae_by_position"]
        if isinstance(mae_by_position, str):
            mae_by_position = json.loads(mae_by_position)
        for pos, info in (mae_by_position or {}).items():
            pooled_error[pos][0] += info["mae"] * info["n"]
            pooled_error[pos][1] += info["n"]

        for key in totals:
            totals[key][0] += row[f"{key}_hits"] or 0
            totals[key][1] += row[f"{key}_opportunities"] or 0

        if row["avg_field_percentile"] is not None:
            field_percentiles_by_week.append((row["season"], row["week"], float(row["avg_field_percentile"])))

    mae_by_position_overall = {pos: round(total / n, 4) for pos, (total, n) in pooled_error.items() if n}
    hit_rates = {key: round(hits / opps, 4) if opps else None for key, (hits, opps) in totals.items()}

    field_only = [fp for _, _, fp in field_percentiles_by_week]
    worst = min(field_percentiles_by_week, key=lambda t: t[2]) if field_percentiles_by_week else None
    best = max(field_percentiles_by_week, key=lambda t: t[2]) if field_percentiles_by_week else None

    return {
        "weeks_evaluated": len(rows),
        "mae_by_position": mae_by_position_overall,
        # p20: fraction of players scoring AT OR BELOW their as-of 20th-
        # percentile (floor) projection - should land near 20% if calibrated.
        "p20_hits": totals["p20"][0],
        "p20_opportunities": totals["p20"][1],
        "p20_hit_rate": hit_rates["p20"],
        # p50: fraction scoring AT OR ABOVE their median projection - should
        # land near 50%. Well below 50% means the median itself runs high
        # (a bias problem); near 50% while p20/p80 still miss their targets
        # means the distribution is too narrow (a spread/variance problem).
        "p50_hits": totals["p50"][0],
        "p50_opportunities": totals["p50"][1],
        "p50_hit_rate": hit_rates["p50"],
        "p80_hits": totals["p80"][0],
        "p80_opportunities": totals["p80"][1],
        "p80_hit_rate": hit_rates["p80"],
        "avg_field_percentile": round(sum(field_only) / len(field_only), 4) if field_only else None,
        "worst_week": {"season": worst[0], "week": worst[1], "avg_field_percentile": worst[2]} if worst else None,
        "best_week": {"season": best[0], "week": best[1], "avg_field_percentile": best[2]} if best else None,
    }
