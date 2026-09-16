import math
from collections import defaultdict

from sqlalchemy import text

from data.nflverse_fetch import fetch_pfr_def_advstats
from data.player_crosswalk import resolve_dk_players_to_gsis
from db.migrate import get_engine
from models.backtest import load_actual_scores, load_slate_pool
from models.calibration import _asof_projections, _available_weeks, _played_gsis_ids

# ---------------------------------------------------------------------------
# What this module is, and what it deliberately is NOT.
#
# Verified before writing any of this, against real source and real
# downloaded files - not table names or documentation summaries:
#   - load_nextgen_stats() (ngs_passing/receiving/rushing, tag nextgen_stats):
#     downloaded all three real files and printed their columns. Pure weekly
#     PLAYER aggregates (cushion, separation, air yards, YAC vs expected) -
#     no play-level rows, no opponent or defender fields of any kind.
#   - load_participation() (tag pbp_participation): downloaded the real file
#     and confirmed its 26 real columns against nflreadr's own dictionary.
#     Has play-level offense_players/defense_players (unordered GSIS-id
#     lists) and a single `route` field documented as "the route THE PRIMARY
#     RECEIVER on a play took" - singular. No field anywhere maps a specific
#     defender to a specific receiver, and no field tags whether a receiver
#     lined up in the slot or wide on a given play.
#   - Searched every dictionary/source file in nflreadr for "slot" or
#     "alignment": the only hit is depth_charts' pos_slot, a formation depth-
#     chart position number, unrelated to receiver alignment.
#
# Conclusion: a true per-play "who covered whom" fact and a slot-vs-wide
# alignment proxy are BOTH unbuildable from real nflverse data - there is
# nothing to compute either one from. Guessing alignment from
# offense_personnel groupings (e.g. "3 WR" doesn't say which WR was where)
# would be exactly the "guess dressed up as insight" this project's
# conventions rule out.
#
# What IS real: PFR's per-defender weekly charting (tag pfr_advstats,
# advstats_week_def_{season}.csv.gz - see fetch_pfr_def_advstats), which has
# named-defender targets/completions/yards/TDs allowed each week. That's a
# real, defender-attributed coverage-performance signal, but it's still PFR's
# own judgment call about who was "the" defender in coverage on a given
# target - not a verified per-play assignment fact, and not broken out by
# receiver alignment. This module aggregates it to TEAM level (summed across
# all of a team's charted defenders in a week) rather than keeping it
# per-defender, because a live DK slate has no way to know pre-snap which
# specific defender will cover which specific receiver.
#
# Confidence scale used throughout this project's matchup adjustments (no
# prior spec for this existed anywhere in this project's history - checked
# the full session transcript, found nothing - so this is defined here):
#   HIGH   - a confirmed, player-specific fact (e.g. an actual injury/role
#            change reflected in real recent games).
#   MEDIUM - a real signal that is indirect or aggregated rather than a
#            confirmed one-on-one fact. Every adjustment this module produces
#            is MEDIUM, never higher - see above for exactly why.
#   LOW    - a small-sample or heavily-inferred signal.
MATCHUP_CONFIDENCE = "MEDIUM"

# Matches models/projections.py's own recency-decay half-life, for consistency.
HALF_LIFE_WEEKS = 4
LOOKBACK_WEEKS = 8
# This was an arbitrary starting guess ("a single bad week shouldn't swing a
# lineup call"), not derived from any variance analysis - see
# run_matchup_backtest_comparison's `clamp` param, which was added specifically
# to test that. Result: loosening it to +-30%/+-50%/uncapped made WR's MAE
# monotonically WORSE at every step (never better, never flat) while TE's
# improvement barely moved past +-15% - i.e. the cap width isn't why WR looks
# bad, and it isn't propping up TE's result either. Left at +-15% since
# nothing in that test argued for a different value.
FACTOR_CLAMP = (0.85, 1.15)
# WR was deliberately removed after backtesting (see
# run_matchup_backtest_comparison and models/matchups.py's module history):
# across seasons 2024-2026, 97.7% of eligible WR/TE player-weeks had real
# opponent coverage data (2025-2026 100%, 2024 94.9% - not a thin-sample
# artifact), and loosening FACTOR_CLAMP from +-15% up to uncapped made WR's
# MAE strictly worse at every step (4.84 baseline -> 4.88 -> 4.90 -> 4.91 ->
# 4.91), never better and never flat. That's the signature of a genuinely
# harmful signal for WR, not an under- or over-tuned magnitude - so WR is
# excluded rather than shipped with a "best-guess" cap. TE's small MAE
# improvement (3.735 -> ~3.71, ~0.7%) held steady across every cap width
# tested on the same real data, which is consistent with a real, if modest,
# effect - kept, and still opt-in only (see run_matchup_backtest_comparison's
# own docstring for why it isn't the projection pipeline's default).
MATCHUP_ELIGIBLE_POSITIONS = {"TE"}
# Positions run_matchup_backtest_comparison evaluates, independent of which
# ones are currently live in MATCHUP_ELIGIBLE_POSITIONS - kept broader so a
# future re-run still shows WR's (expected no-op) numbers side by side with
# TE's, rather than silently losing the comparison that justified excluding
# WR in the first place.
_BACKTEST_SCAN_POSITIONS = {"WR", "TE"}


def _decay_weight(weeks_ago):
    return math.exp(-math.log(2) / HALF_LIFE_WEEKS * weeks_ago)


def refresh_team_pass_defense_weekly(seasons, engine=None):
    """Aggregate PFR's per-defender weekly coverage charting up to team level
    and store it. Summed (not averaged) across a team's charted defenders in
    a week - targets/completions/yards/TDs are additive counts, so a sum is a
    real team total; averaging per-defender rate stats (e.g. passer rating
    allowed) across defenders with very different target volumes would not be.
    """
    engine = engine or get_engine()
    df = fetch_pfr_def_advstats(seasons)

    agg = (
        df.groupby(["team", "season", "week"])
        .agg(
            def_targets=("def_targets", "sum"),
            def_completions_allowed=("def_completions_allowed", "sum"),
            def_yards_allowed=("def_yards_allowed", "sum"),
            def_receiving_td_allowed=("def_receiving_td_allowed", "sum"),
        )
        .reset_index()
    )

    upsert_sql = text(
        """
        INSERT INTO team_pass_defense_weekly
            (team, season, week, def_targets, def_completions_allowed,
             def_yards_allowed, def_receiving_td_allowed)
        VALUES (:team, :season, :week, :def_targets, :def_completions_allowed,
                :def_yards_allowed, :def_receiving_td_allowed)
        ON CONFLICT (team, season, week) DO UPDATE SET
            def_targets = EXCLUDED.def_targets,
            def_completions_allowed = EXCLUDED.def_completions_allowed,
            def_yards_allowed = EXCLUDED.def_yards_allowed,
            def_receiving_td_allowed = EXCLUDED.def_receiving_td_allowed
        """
    )

    row_counts = defaultdict(int)
    for season in sorted(int(s) for s in agg["season"].unique()):
        season_rows = agg[agg["season"] == season].to_dict("records")
        with engine.begin() as conn:
            conn.execute(text("DELETE FROM team_pass_defense_weekly WHERE season = :season"), {"season": season})
            for row in season_rows:
                conn.execute(
                    upsert_sql,
                    {
                        "team": row["team"],
                        "season": int(row["season"]),
                        "week": int(row["week"]),
                        "def_targets": int(row["def_targets"]),
                        "def_completions_allowed": int(row["def_completions_allowed"]),
                        "def_yards_allowed": float(row["def_yards_allowed"]),
                        "def_receiving_td_allowed": int(row["def_receiving_td_allowed"]),
                    },
                )
        row_counts[season] = len(season_rows)
    return dict(row_counts)


def _load_prior_weeks(engine, before_season, before_week):
    query = text(
        """
        SELECT team, season, week, def_targets, def_yards_allowed
        FROM team_pass_defense_weekly
        WHERE def_targets > 0
          AND (season < :before_season OR (season = :before_season AND week < :before_week))
        ORDER BY season DESC, week DESC
        """
    )
    with engine.connect() as conn:
        return conn.execute(query, {"before_season": before_season, "before_week": before_week}).fetchall()


def compute_defense_factors(before, engine=None, lookback_weeks=LOOKBACK_WEEKS, clamp=FACTOR_CLAMP):
    """Recency-weighted opponent pass-defense factor per team, as of strictly
    before `before=(season, week)` - the target week itself is never included,
    so this has no lookahead bias when used for backtesting, and works
    unchanged for a live upcoming slate (whose own week hasn't happened yet).

    Returns {team: factor}. factor > 1.0 means that team has allowed more
    yards-per-target than league average recently (a plus matchup for an
    opposing pass-catcher); < 1.0 means the opposite. A team with no prior
    data gets no entry - callers must leave those players unadjusted, not
    default them to a neutral 1.0 that would look like a real "no edge"
    finding rather than "no data".
    """
    engine = engine or get_engine()
    rows = _load_prior_weeks(engine, *before)

    by_team = defaultdict(list)
    for row in rows:
        by_team[row.team].append(row)

    team_ratio = {}
    league_weighted_yards = 0.0
    league_weighted_targets = 0.0
    for team, team_rows in by_team.items():
        team_rows = team_rows[:lookback_weeks]  # already ordered most-recent-first
        weighted_yards = sum(_decay_weight(i) * float(r.def_yards_allowed or 0) for i, r in enumerate(team_rows))
        weighted_targets = sum(_decay_weight(i) * float(r.def_targets or 0) for i, r in enumerate(team_rows))
        if weighted_targets > 0:
            team_ratio[team] = weighted_yards / weighted_targets
            league_weighted_yards += weighted_yards
            league_weighted_targets += weighted_targets

    if not team_ratio or league_weighted_targets == 0:
        return {}

    league_avg_ratio = league_weighted_yards / league_weighted_targets
    return {
        team: min(max(ratio / league_avg_ratio, clamp[0]), clamp[1]) for team, ratio in team_ratio.items()
    }


def apply_matchup_adjustment(values, position, opponent, factors, fields):
    """Scale the given percentile-like fields by the opponent's pass-defense
    factor. WR/TE only - RB/QB/DST pass-defense exposure is different enough
    that this factor shouldn't be reused for them without separate
    verification. Returns (adjusted_dict, factor_or_None); a missing opponent
    or missing factor is a no-op with factor=None (meaning "not adjusted"),
    never a silent default to 1.0 that would look like a verified neutral
    matchup rather than an absence of data.
    """
    factor = factors.get(opponent) if opponent else None
    if position not in MATCHUP_ELIGIBLE_POSITIONS or factor is None:
        return dict(values), None

    adjusted = dict(values)
    for field in fields:
        adjusted[field] = adjusted[field] * factor
    return adjusted, factor


def _opponents_for_week(gsis_ids, season, week, engine):
    query = text(
        "SELECT player_id, opponent FROM player_weekly_stats "
        "WHERE player_id = ANY(:ids) AND season = :season AND week = :week"
    )
    with engine.connect() as conn:
        rows = conn.execute(query, {"ids": list(gsis_ids), "season": season, "week": week}).fetchall()
    return {row.player_id: row.opponent for row in rows}


def run_matchup_backtest_comparison(source_slate_id, seasons=None, engine=None, clamp=FACTOR_CLAMP):
    """WR/TE-only: does layering the opponent pass-defense factor on top of
    the existing as-of projections improve MAE and p20/p50/p80 hit rates
    versus the current baseline, across every real historical week this can
    be tested against? Mirrors models/calibration.py's own methodology
    exactly (same as-of projections, same snap_pct-based played filter, same
    pooled-not-averaged aggregation, same no-lookahead `before` cutoff) so
    the two numbers are a fair before/after comparison, not two different
    tests. Never writes to calibration_weekly - this is a side-by-side
    diagnostic, not a replacement for the stored baseline.

    Reports pooled metrics AND an "adjusted-only" subset: for the majority of
    player-weeks with no prior team_pass_defense_weekly coverage for their
    opponent, apply_matchup_adjustment is a literal no-op (adjusted == base),
    so pooling those in with the ones that actually got a real factor dilutes
    the pooled numbers toward "no difference" regardless of whether the
    signal itself is any good. The adjusted-only subset isolates the actual
    effect where the layer did something; per_season_coverage separately
    reports what fraction of each season's sample that subset actually is.
    """
    engine = engine or get_engine()

    slate_players = load_slate_pool(source_slate_id, engine)
    wr_te_players = [p for p in slate_players if p["position"] in _BACKTEST_SCAN_POSITIONS]
    if not wr_te_players:
        raise ValueError(f"No WR/TE players found in slate_player_pool for slate {source_slate_id}")
    gsis_by_dk_id, _, _ = resolve_dk_players_to_gsis(wr_te_players, engine)
    players_by_id = {p["player_id"]: p for p in wr_te_players}

    weeks = _available_weeks(engine)
    if seasons is not None:
        weeks = [(s, w) for s, w in weeks if s in seasons]

    baseline_mae = defaultdict(lambda: [0.0, 0])
    matchup_mae = defaultdict(lambda: [0.0, 0])
    baseline_hits = {"p20": [0, 0], "p50": [0, 0], "p80": [0, 0]}
    matchup_hits = {"p20": [0, 0], "p50": [0, 0], "p80": [0, 0]}
    # Adjusted-only subset - see docstring above for why this is tracked
    # separately from the pooled (all-player) numbers.
    adj_baseline_mae = defaultdict(lambda: [0.0, 0])
    adj_matchup_mae = defaultdict(lambda: [0.0, 0])
    adj_baseline_hits = {"p20": [0, 0], "p50": [0, 0], "p80": [0, 0]}
    adj_matchup_hits = {"p20": [0, 0], "p50": [0, 0], "p80": [0, 0]}
    per_season_coverage = defaultdict(lambda: [0, 0])  # season -> [adjusted, total]
    weeks_evaluated = 0

    for season, week in weeks:
        asof = _asof_projections(wr_te_players, gsis_by_dk_id, season, week, engine)
        actual_points, _ = load_actual_scores(wr_te_players, season, week, engine)
        eligible_ids = set(asof) & set(actual_points)
        if not eligible_ids:
            continue

        played_gsis_ids = _played_gsis_ids(gsis_by_dk_id.values(), season, week, engine)
        eligible_dk_ids = [dk_id for dk_id in eligible_ids if gsis_by_dk_id.get(dk_id) in played_gsis_ids]
        if not eligible_dk_ids:
            continue
        weeks_evaluated += 1

        factors = compute_defense_factors(before=(season, week), engine=engine, clamp=clamp)
        opponents_by_gsis = _opponents_for_week(gsis_by_dk_id.values(), season, week, engine)

        for dk_id in eligible_dk_ids:
            player = players_by_id[dk_id]
            position = player["position"]
            gsis_id = gsis_by_dk_id.get(dk_id)
            opponent = opponents_by_gsis.get(gsis_id)
            actual = actual_points[dk_id]
            base = asof[dk_id]
            per_season_coverage[season][1] += 1

            def _score(mae_bucket, hit_bucket, proj):
                mae_bucket[position][0] += abs(proj["median"] - actual)
                mae_bucket[position][1] += 1
                if actual <= proj["p20"]:
                    hit_bucket["p20"][0] += 1
                hit_bucket["p20"][1] += 1
                if actual >= proj["median"]:
                    hit_bucket["p50"][0] += 1
                hit_bucket["p50"][1] += 1
                if actual >= proj["p80"]:
                    hit_bucket["p80"][0] += 1
                hit_bucket["p80"][1] += 1

            _score(baseline_mae, baseline_hits, base)

            adjusted, factor = apply_matchup_adjustment(
                base, position, opponent, factors, fields=["median", "p20", "p80"]
            )
            _score(matchup_mae, matchup_hits, adjusted)

            if factor is not None:
                per_season_coverage[season][0] += 1
                _score(adj_baseline_mae, adj_baseline_hits, base)
                _score(adj_matchup_mae, adj_matchup_hits, adjusted)

    def _mae_summary(pooled):
        return {pos: round(total / n, 4) for pos, (total, n) in pooled.items() if n}

    def _hit_rate_summary(pooled):
        return {key: round(hits / opps, 4) if opps else None for key, (hits, opps) in pooled.items()}

    total_eligible = sum(total for _, total in per_season_coverage.values())
    total_adjusted = sum(adj for adj, _ in per_season_coverage.values())

    return {
        "weeks_evaluated": weeks_evaluated,
        "players_adjusted": total_adjusted,
        "players_eligible": total_eligible,
        "coverage_fraction": round(total_adjusted / total_eligible, 4) if total_eligible else None,
        "per_season_coverage": {
            season: {"adjusted": adj, "eligible": total, "fraction": round(adj / total, 4) if total else None}
            for season, (adj, total) in sorted(per_season_coverage.items())
        },
        "pooled": {
            "baseline_mae_by_position": _mae_summary(baseline_mae),
            "matchup_mae_by_position": _mae_summary(matchup_mae),
            "baseline_hit_rates": _hit_rate_summary(baseline_hits),
            "matchup_hit_rates": _hit_rate_summary(matchup_hits),
        },
        "adjusted_only": {
            "baseline_mae_by_position": _mae_summary(adj_baseline_mae),
            "matchup_mae_by_position": _mae_summary(adj_matchup_mae),
            "baseline_hit_rates": _hit_rate_summary(adj_baseline_hits),
            "matchup_hit_rates": _hit_rate_summary(adj_matchup_hits),
        },
    }
