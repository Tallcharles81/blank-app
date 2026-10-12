"""Project a defense from its opponent's Vegas implied total.

Measured 2026-10-12 (scratchpad dst_predictors.py): trained on 2023-24 DST
weeks (992), tested on 2025 + 2026 weeks 1-5 (698), no lookahead:

    predictor                              corr  within-week Spearman  top-3 a week avg
    DST's own last-8 average               0.06        0.07                 6.9
    opponent's points allowed to DSTs      0.17        0.19                 8.0
    opponent's implied total               0.31        0.33                 9.1
    (all DSTs averaged 6.1)

Adding own average, the spread, or the opponent's DST-points-allowed to the
opponent's implied total added nothing. So a defense's own history is
close to noise week to week, and what matters is who it plays. The old
projection was built from the DST's own history plus its OWN team's implied
total (the wrong side of the ball); on 2026-10-11 it ranked the Raiders
(vs NE) 3rd, and they scored -1 in 5 lineups.

The ladder is the real score distribution of past DSTs that faced a
similar implied total (within WINDOW points, widened until MIN_GAMES), so
it carries the real skew (a defensive TD is a big share of a ceiling). On
the test weeks it was calibrated: 11/26/52/76/90% of real scores fell below
p10/25/50/75/90.
"""
import numpy as np
from sqlalchemy import text

from db.migrate import get_engine

PERCENTILES = (10, 25, 50, 75, 90)
WINDOW = 1.5
MIN_GAMES = 150


def load_dst_history(before=None, engine=None):
    """[(dk_points, opponent_implied_total)] for every stored DST game,
    optionally only games before (season, week) so a retro build can't see
    its own week."""
    engine = engine or get_engine()
    with engine.connect() as conn:
        rows = conn.execute(text(
            """SELECT d.fantasy_points_ppr::float AS pts, o.vegas_implied_total::float AS opp_imp
               FROM player_weekly_stats d
               JOIN player_weekly_stats o ON o.position = 'DST' AND o.team = d.opponent
                                         AND o.season = d.season AND o.week = d.week
               WHERE d.position = 'DST' AND d.fantasy_points_ppr IS NOT NULL
                 AND o.vegas_implied_total IS NOT NULL
                 AND (CAST(:s AS int) IS NULL OR (d.season, d.week) < (CAST(:s AS int), CAST(:w AS int)))"""),
            {"s": before[0] if before else None, "w": before[1] if before else None}).fetchall()
    return [(r.pts, r.opp_imp) for r in rows]


def matchup_ladder(opponent_implied_total, history, window=WINDOW, min_games=MIN_GAMES):
    """{"10": .., "25": .., "50": .., "75": .., "90": ..} - the percentiles of
    real DST scores against opponents implied within `window` points."""
    if not history:
        raise ValueError("No DST history to build a matchup ladder from")
    pts = np.array([h[0] for h in history])
    imp = np.array([h[1] for h in history])
    # DST scores are whole numbers, so raw neighbor percentiles tie (six DSTs
    # at exactly 7.0 in week 6). Shifting each neighbor's score to the target
    # total along the overall slope (about -0.5 pts per implied point) keeps
    # the ladder's shape and still ranks a weaker opponent higher.
    slope = np.polyfit(imp, pts, 1)[0] if len(set(imp)) > 1 else 0.0
    while True:
        mask = np.abs(imp - opponent_implied_total) <= window
        if mask.sum() >= min(min_games, len(pts)):
            break
        window += 0.5
    near = pts[mask] + slope * (opponent_implied_total - imp[mask])
    return {str(q): round(float(v), 2) for q, v in zip(PERCENTILES, np.percentile(near, PERCENTILES))}


def matchup_projection(opponent_implied_total, history):
    ladder = matchup_ladder(opponent_implied_total, history)
    return {"proj_floor": ladder["25"], "proj_median": ladder["50"], "proj_ceiling": ladder["90"],
            "proj_percentiles": ladder}
