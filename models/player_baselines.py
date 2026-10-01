"""Running per-player baselines, seeded from an outside source and updated
weekly from real results.

Set by the user 2026-10-01: the one-time SaberSim export is the starting
point for the players it covers (PHI, CHI, PIT, CLE), and from then on each
player's numbers are updated from what actually happens. Every other player
has no baseline and keeps the model's own projection, which already rebuilds
from recent games each week.

Weekly flow:
  1. seed_from_sabersim(path, week_by_team)  - once.
  2. update_baselines(through=(season, week)) - after each week's games, once
     player_weekly_stats is refreshed. Each game moves the median
     BASELINE_UPDATE_WEIGHT of the way toward the actual score; the
     distribution's shape is kept (scaled with the median).
  3. generate_projections blends a baseline with the model's projection
     (BASELINE_BLEND_WEIGHT each) and records model / baseline / blend in
     projection_sources, so projection_accuracy shows whether it helps.

Both weights are starting values, not fitted: there is one seed file and no
history to fit them on yet. Revisit once projection_accuracy has several
weeks of model vs blend results.
"""
import csv
import json

from sqlalchemy import text

from data.player_crosswalk import resolve_dk_players_to_gsis
from data.sabersim_import import parse_sabersim_csv
from db.migrate import get_engine

BASELINE_UPDATE_WEIGHT = 0.2
BASELINE_BLEND_WEIGHT = 0.5
_USAGE_FIELDS = ("Pass Att", "Pass Yds", "Pass TD", "Rec", "Rec Yds", "Rec TD", "Rush Att", "Rush Yds", "Rush TD")


def smooth_ladder(ladder, actual, weight=BASELINE_UPDATE_WEIGHT):
    """Move the median `weight` toward `actual`, scaling the whole ladder so
    the distribution keeps its shape. A zero median can't be scaled - the
    ladder is shifted instead."""
    old = float(ladder["50"])
    new = (1 - weight) * old + weight * actual
    if old > 0:
        return {k: round(float(v) * new / old, 2) for k, v in ladder.items()}
    return {k: round(max(0.0, float(v) + (new - old)), 2) for k, v in ladder.items()}


def blend_ladders(model, baseline, weight=BASELINE_BLEND_WEIGHT):
    return {k: round((1 - weight) * float(model[k]) + weight * float(baseline[k]), 2) for k in model}


_REPLACE_OLDER_SQL = """UPDATE SET name = EXCLUDED.name, team = EXCLUDED.team, position = EXCLUDED.position,
                        ladder = EXCLUDED.ladder, usage = EXCLUDED.usage, source = EXCLUDED.source,
                        seeded_season = EXCLUDED.seeded_season, seeded_week = EXCLUDED.seeded_week,
                        updated_season = EXCLUDED.updated_season, updated_week = EXCLUDED.updated_week,
                        games_applied = 0, updated_at = now()
                    WHERE (player_baselines.seeded_season, player_baselines.seeded_week)
                          <= (EXCLUDED.seeded_season, EXCLUDED.seeded_week)"""


def _usage(path):
    out = {}
    with open(path, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            vals = {}
            for k in _USAGE_FIELDS:
                try:
                    v = float(r.get(k) or 0)
                except ValueError:
                    v = 0.0
                if v:
                    vals[k] = round(v, 2)
            out[(r["Name"].strip(), r["Team"].strip())] = vals
    return out


def seed_from_sabersim(path, season, week_by_team, engine=None, replace_older=False):
    """Create baselines for every player the export projects. week_by_team
    gives the week each team's game in the export belongs to (an export can
    span weeks, e.g. a Monday and the following Thursday); results before
    that week are treated as already reflected.

    By default existing baselines are left alone, so re-running never resets
    a player's updates. replace_older=True lets a newer export replace any
    baseline seeded for the same or an earlier week - the newer projection
    already reflects the games the old baseline was updated with."""
    engine = engine or get_engine()
    rows = [r for r in parse_sabersim_csv(path) if r["ladder"] and r["team"] in week_by_team]
    dk_like = [{"player_id": f"ss_{i}", "name": r["name"], "position": r["position"], "team": r["team"]} for i, r in enumerate(rows)]
    mapping, unmatched, ambiguous = resolve_dk_players_to_gsis(dk_like, engine)
    usage = _usage(path)
    seeded = 0
    with engine.begin() as conn:
        for d, r in zip(dk_like, rows):
            gsis = mapping.get(d["player_id"])
            if gsis is None:
                continue
            week = week_by_team[r["team"]]
            seeded += conn.execute(
                text(
                    """
                    INSERT INTO player_baselines (gsis_id, name, team, position, ladder, usage, source,
                        seeded_season, seeded_week, updated_season, updated_week)
                    VALUES (:g, :n, :t, :pos, :l, :u, 'sabersim', :s, :w, :s, :prev)
                    ON CONFLICT (gsis_id) DO {}
                    """.format(_REPLACE_OLDER_SQL if replace_older else "NOTHING")
                ),
                {"g": gsis, "n": r["name"], "t": r["team"], "pos": r["position"], "l": json.dumps(r["ladder"]),
                 "u": json.dumps(usage.get((r["name"], r["team"]), {})), "s": season, "w": week, "prev": week - 1},
            ).rowcount
    return {"seeded": seeded, "projected_rows": len(rows),
            "unmatched": [rows[int(i[3:])]["name"] for i in unmatched + ambiguous]}


def update_baselines(through, engine=None):
    """Apply every real game after each baseline's updated week, up to and
    including `through` (season, week), in order. A week a player didn't
    play (no stats row) changes nothing - an absence isn't a 0-point
    performance to learn from."""
    engine = engine or get_engine()
    season, week = through
    updated = []
    with engine.begin() as conn:
        baselines = conn.execute(text("SELECT * FROM player_baselines")).mappings().fetchall()
        for b in baselines:
            games = conn.execute(
                text(
                    """
                    SELECT season, week, fantasy_points_ppr::float AS pts FROM player_weekly_stats
                    WHERE player_id = :g AND fantasy_points_ppr IS NOT NULL
                      AND (season, week) > (:us, :uw) AND (season, week) <= (:s, :w)
                    ORDER BY season, week
                    """
                ),
                {"g": b["gsis_id"], "us": b["updated_season"], "uw": b["updated_week"], "s": season, "w": week},
            ).fetchall()
            ladder = b["ladder"] if isinstance(b["ladder"], dict) else json.loads(b["ladder"])
            for g in games:
                ladder = smooth_ladder(ladder, g.pts)
            if games:
                updated.append((b["name"], [g.pts for g in games], ladder["50"]))
            conn.execute(
                text(
                    """
                    UPDATE player_baselines SET ladder = :l, games_applied = games_applied + :n,
                        updated_season = :s, updated_week = :w, updated_at = now()
                    WHERE gsis_id = :g AND (updated_season, updated_week) < (:s, :w)
                    """
                ),
                {"l": json.dumps(ladder), "n": len(games), "s": season, "w": week, "g": b["gsis_id"]},
            )
    return {"through": through, "baselines": len(baselines), "updated": updated}


def load_baselines(gsis_ids, engine=None):
    engine = engine or get_engine()
    ids = [g for g in gsis_ids if g]
    if not ids:
        return {}
    with engine.connect() as conn:
        rows = conn.execute(text("SELECT gsis_id, ladder FROM player_baselines WHERE gsis_id = ANY(:ids)"), {"ids": ids}).fetchall()
    return {r.gsis_id: (r.ladder if isinstance(r.ladder, dict) else json.loads(r.ladder)) for r in rows}
