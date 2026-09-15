import json
import random
from collections import defaultdict

from sqlalchemy import text

from data.player_crosswalk import resolve_dk_players_to_gsis
from db.migrate import get_engine
from models.backtest import DEFAULT_RANDOM_FIELD_SIZE, load_actual_scores, load_slate_pool
from models.optimizer import SALARY_CAP, build_lineups_from_pool
from models.projections import _load_recent_stats, _project_from_history

# The stored percentile ladder (models/projections.py's PERCENTILE_Z) has 75th
# and 90th but not 80th - interpolated between them the same way
# models/simulation.py already interpolates arbitrary percentiles from that
# ladder, rather than approximated by the nearest stored bucket.
_P80_LOWER = (0.75, "75")
_P80_UPPER = (0.90, "90")


def _interpolated_p80(percentiles):
    x0, key0 = _P80_LOWER
    x1, key1 = _P80_UPPER
    y0, y1 = percentiles[key0], percentiles[key1]
    return y0 + (y1 - y0) * (0.80 - x0) / (x1 - x0)


def _available_weeks(engine):
    query = text("SELECT DISTINCT season, week FROM player_weekly_stats ORDER BY season, week")
    with engine.connect() as conn:
        return [(row.season, row.week) for row in conn.execute(query)]


def _asof_projections_with_p80(slate_players, gsis_by_dk_id, season, week, engine):
    history = _load_recent_stats(gsis_by_dk_id.values(), engine, before=(season, week))

    result = {}
    for player in slate_players:
        gsis_id = gsis_by_dk_id.get(player["player_id"])
        games = history.get(gsis_id) if gsis_id else None
        if not games:
            continue
        proj = _project_from_history(games, player["position"])
        result[player["player_id"]] = {
            "median": proj["proj_median"],
            "p80": _interpolated_p80(proj["proj_percentiles"]),
        }
    return result


def run_weekly_calibration(source_slate_id, seasons=None, num_lineups=1, random_field_size=100, engine=None):
    """Backtest source_slate_id's real salaries against every available real
    historical week (or just `seasons` if given), one independent backtest per
    week. A week is skipped (not failed) if there's no overlap between
    as-of-projectable and actually-scored players for it - e.g. the very
    first available week has no prior history to project from at all.

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
        asof = _asof_projections_with_p80(slate_players, gsis_by_dk_id, season, week, engine)
        actual_points, _ = load_actual_scores(slate_players, season, week, engine)

        eligible_ids = set(asof) & set(actual_points)
        if not eligible_ids:
            skipped.append((season, week))
            continue

        eligible_players = [p for p in slate_players if p["player_id"] in eligible_ids]

        abs_error_by_position = defaultdict(lambda: [0.0, 0])
        p80_hits = p80_opportunities = 0
        for p in eligible_players:
            pid = p["player_id"]
            error = abs(asof[pid]["median"] - actual_points[pid])
            abs_error_by_position[p["position"]][0] += error
            abs_error_by_position[p["position"]][1] += 1

            p80_opportunities += 1
            if actual_points[pid] >= asof[pid]["p80"]:
                p80_hits += 1

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
                "p80_hits": p80_hits,
                "p80_opportunities": p80_opportunities,
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
            (season, week, num_players_evaluated, mae_by_position, p80_hits, p80_opportunities,
             avg_field_percentile, field_size)
        VALUES
            (:season, :week, :num_players_evaluated, :mae_by_position, :p80_hits, :p80_opportunities,
             :avg_field_percentile, :field_size)
        ON CONFLICT (season, week) DO UPDATE SET
            num_players_evaluated = EXCLUDED.num_players_evaluated,
            mae_by_position = EXCLUDED.mae_by_position,
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
    total_p80_hits = total_p80_opportunities = 0
    field_percentiles_by_week = []

    for row in rows:
        mae_by_position = row["mae_by_position"]
        if isinstance(mae_by_position, str):
            mae_by_position = json.loads(mae_by_position)
        for pos, info in (mae_by_position or {}).items():
            pooled_error[pos][0] += info["mae"] * info["n"]
            pooled_error[pos][1] += info["n"]

        total_p80_hits += row["p80_hits"] or 0
        total_p80_opportunities += row["p80_opportunities"] or 0

        if row["avg_field_percentile"] is not None:
            field_percentiles_by_week.append((row["season"], row["week"], float(row["avg_field_percentile"])))

    mae_by_position_overall = {pos: round(total / n, 4) for pos, (total, n) in pooled_error.items() if n}
    p80_hit_rate = round(total_p80_hits / total_p80_opportunities, 4) if total_p80_opportunities else None

    field_only = [fp for _, _, fp in field_percentiles_by_week]
    worst = min(field_percentiles_by_week, key=lambda t: t[2]) if field_percentiles_by_week else None
    best = max(field_percentiles_by_week, key=lambda t: t[2]) if field_percentiles_by_week else None

    return {
        "weeks_evaluated": len(rows),
        "mae_by_position": mae_by_position_overall,
        "p80_hits": total_p80_hits,
        "p80_opportunities": total_p80_opportunities,
        "p80_hit_rate": p80_hit_rate,
        "avg_field_percentile": round(sum(field_only) / len(field_only), 4) if field_only else None,
        "worst_week": {"season": worst[0], "week": worst[1], "avg_field_percentile": worst[2]} if worst else None,
        "best_week": {"season": best[0], "week": best[1], "avg_field_percentile": best[2]} if best else None,
    }
