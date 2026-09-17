import random

from sqlalchemy import text

from data.player_crosswalk import resolve_dk_players_to_gsis
from db.migrate import get_engine
from models.optimizer import SALARY_CAP, build_lineups_from_pool
from models.projections import _load_recent_stats, _project_from_history

# A naive "field": legal-but-random lineups from the same real slate, scored on
# the same real actual results. This is a baseline for percentile ranking, not
# a model of skilled human opponents - it answers "did this lineup beat a
# random legal roster," not "what place would it have finished in a real
# contest" (which would need real opponent-lineup data we don't have).
DEFAULT_RANDOM_FIELD_SIZE = 300


def load_slate_pool(slate_id, engine):
    query = text("SELECT player_id, name, position, salary, team FROM slate_player_pool WHERE slate_id = :slate_id")
    with engine.connect() as conn:
        rows = conn.execute(query, {"slate_id": slate_id}).mappings().fetchall()
    return [dict(row) for row in rows]


def load_actual_scores(slate_players, season, week, engine):
    """Resolve each DK player to a real fantasy_points_ppr for (season, week).

    Returns (scores, excluded_ids):
    - scores: {dk_player_id: actual_points}. A player who resolves to a real
      historical identity, HAS other rows fetched for that season, but has no
      row for this exact (season, week) is scored 0 - the correct real-contest
      outcome for a bye/inactive/didn't-play week. A player with zero rows for
      the entire season is a different situation - no real data was ever
      fetched for them that year at all (e.g. only DST has been fetched for
      2025/2026 so far, no skill positions) - and gets excluded rather than
      defaulted to a fabricated 0, which would silently fake an entire
      season's worth of "actual" skill-position scores.
    - excluded_ids: the above, plus DK player_ids the crosswalk couldn't
      resolve to any historical identity at all (see
      data/player_crosswalk.py) - there is no real score to use for either
      case, so both are left out of the candidate pool entirely rather than
      guessed at.
    """
    gsis_by_dk_id, unmatched, ambiguous = resolve_dk_players_to_gsis(slate_players, engine)
    excluded_ids = set(unmatched) | set(ambiguous)

    ids = list(gsis_by_dk_id.values())
    season_query = text("SELECT DISTINCT player_id FROM player_weekly_stats WHERE player_id = ANY(:ids) AND season = :season")
    week_query = text(
        "SELECT player_id, fantasy_points_ppr FROM player_weekly_stats "
        "WHERE player_id = ANY(:ids) AND season = :season AND week = :week"
    )
    with engine.connect() as conn:
        in_season_scope = {row.player_id for row in conn.execute(season_query, {"ids": ids, "season": season})}
        actual_by_gsis = {
            row.player_id: float(row.fantasy_points_ppr)
            for row in conn.execute(week_query, {"ids": ids, "season": season, "week": week})
        }

    scores = {}
    for dk_id, gsis_id in gsis_by_dk_id.items():
        if gsis_id in actual_by_gsis:
            scores[dk_id] = actual_by_gsis[gsis_id]
        elif gsis_id in in_season_scope:
            scores[dk_id] = 0.0
        else:
            excluded_ids.add(dk_id)

    return scores, excluded_ids


def _asof_projected_points(slate_players, gsis_by_dk_id, season, week, engine, field="proj_median"):
    # Same recency-weighted model as models/projections.py, just restricted to
    # history strictly before (season, week) - see _load_recent_stats's
    # `before` param - so this can't see the very outcome it's being
    # backtested against. `field` selects which of _project_from_history's
    # three outputs (proj_floor/proj_median/proj_ceiling) drives the lineup -
    # models/calibration.py's run_gpp_ceiling_backtest uses proj_ceiling here,
    # the same field real GPP-mode lineups (models/optimizer.generate_lineups)
    # are built from.
    history = _load_recent_stats(gsis_by_dk_id.values(), engine, before=(season, week))

    points_by_dk_id = {}
    for player in slate_players:
        gsis_id = gsis_by_dk_id.get(player["player_id"])
        games = history.get(gsis_id) if gsis_id else None
        if not games:
            continue
        points_by_dk_id[player["player_id"]] = _project_from_history(games, player["position"])[field]
    return points_by_dk_id


def backtest_slate(
    source_slate_id,
    season,
    week,
    num_lineups=1,
    salary_cap=SALARY_CAP,
    max_players_per_team=None,
    min_uniques=1,
    max_exposure=None,
    min_exposure=None,
    random_field_size=DEFAULT_RANDOM_FIELD_SIZE,
    engine=None,
):
    """Build lineup(s) from `source_slate_id`'s real DK salaries using only
    data available before (season, week), then score them against the real
    actual results for that week.

    Note the one real constraint this works around: there's no historical DK
    salary archive fetched (or generally available) for past slates, so this
    backtests using the CURRENT/given slate's real salary structure against a
    PAST week's real outcomes for the same players - not "what DK would have
    actually priced this slate at back then." It's an honest, useful backtest
    of the projections + optimizer pipeline against real results, not a
    reconstruction of a real historical contest.

    Returns a dict with:
    - lineups: [{roster, projected_points, actual_points, pct_of_ceiling,
      field_percentile}, ...]
    - ceiling_points: the best possible actual score from any legal lineup on
      this slate, with perfect hindsight.
    - field_size: how many random-legal comparison lineups were sampled.
    - excluded_no_asof_projection / excluded_no_actual_result: DK player_ids
      dropped from the candidate pool because either couldn't be produced -
      see load_actual_scores and _asof_projected_points.
    """
    engine = engine or get_engine()

    slate_players = load_slate_pool(source_slate_id, engine)
    if not slate_players:
        raise ValueError(f"No players found in slate_player_pool for slate {source_slate_id}")

    gsis_by_dk_id, unmatched, ambiguous = resolve_dk_players_to_gsis(slate_players, engine)
    crosswalk_excluded = set(unmatched) | set(ambiguous)

    actual_points, _ = load_actual_scores(slate_players, season, week, engine)
    asof_points = _asof_projected_points(slate_players, gsis_by_dk_id, season, week, engine)

    # A lineup can only be built from players we could both have projected
    # blind (as-of) AND verify against a real outcome (actual) - missing
    # either side excludes a player from the whole backtest.
    eligible_ids = set(asof_points) & set(actual_points)
    eligible_players = [p for p in slate_players if p["player_id"] in eligible_ids]
    if not eligible_players:
        raise ValueError("No players have both an as-of projection and a real actual score for this week")

    asof_pool = [{**p, "points": asof_points[p["player_id"]]} for p in eligible_players]
    our_lineups, _ = build_lineups_from_pool(
        asof_pool,
        num_lineups=num_lineups,
        salary_cap=salary_cap,
        max_players_per_team=max_players_per_team,
        min_uniques=min_uniques,
        max_exposure=max_exposure,
        min_exposure=min_exposure,
    )

    def score_actual(roster):
        return sum(actual_points[p["player_id"]] for _, p in roster)

    # Ceiling: the single best lineup that could have been built from this
    # slate with perfect hindsight of the real scores - how close our blind,
    # as-of lineup(s) got to the maximum possible.
    actual_pool = [{**p, "points": actual_points[p["player_id"]]} for p in eligible_players]
    ceiling_lineups, _ = build_lineups_from_pool(
        actual_pool, num_lineups=1, salary_cap=salary_cap, max_players_per_team=max_players_per_team
    )
    ceiling_points = ceiling_lineups[0]["total_points"]

    field_scores = []
    rng = random.Random()
    for _ in range(random_field_size):
        randomized_pool = [{**p, "points": rng.random()} for p in eligible_players]
        try:
            field_lineup, _ = build_lineups_from_pool(
                randomized_pool, num_lineups=1, salary_cap=salary_cap, max_players_per_team=max_players_per_team
            )
        except ValueError:
            continue
        field_scores.append(score_actual(field_lineup[0]["roster"]))

    results = []
    for lu in our_lineups:
        actual = score_actual(lu["roster"])
        better_than = sum(1 for s in field_scores if actual > s)
        results.append(
            {
                "roster": lu["roster"],
                "projected_points": lu["total_points"],
                "actual_points": actual,
                "pct_of_ceiling": round(actual / ceiling_points, 4) if ceiling_points else None,
                "field_percentile": round(better_than / len(field_scores), 4) if field_scores else None,
            }
        )

    return {
        "lineups": results,
        "ceiling_points": ceiling_points,
        "field_size": len(field_scores),
        "excluded_no_asof_projection": sorted(crosswalk_excluded | (set(actual_points) - set(asof_points))),
        "excluded_no_actual_result": sorted(crosswalk_excluded | (set(asof_points) - set(actual_points))),
    }
