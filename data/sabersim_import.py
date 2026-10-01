"""SaberSim projections as the base projection source.

Set by the user (2026-10-01): SaberSim exports are the starting point for
every slate, with DK Edge adding what SaberSim doesn't cover (kickers on
Showdown slates - its exports are Classic) and the pre-build news
corrections on top. Both sources are kept in projection_sources so their
accuracy can be compared once games are played (projection_accuracy).

A SaberSim export is one row per player: "DFS ID" is the DK player id of
the Classic slate it was exported for, "SS Proj" the median, and
dk_25/50/75/85/95/99_percentile the distribution. Showdown slates use
different DK ids, so rows are matched by (name, team) there.
"""
import csv
import json
import os

from sqlalchemy import text

from db.migrate import get_engine

SOURCE = "sabersim"
CAPTAIN_MULTIPLIER = 1.5
# Normal-shape interpolation to our 10/25/50/75/90 ladder from SaberSim's
# 25/50/75/85/95: p10 = p50 - (z10/z25) * (p50 - p25) with z10/z25 = 1.90,
# and p90 sits 40% of the way from p85 to p95 (z: 1.04, 1.28, 1.645).
P10_SPREAD = 1.9
P90_FROM_P85 = 0.4


def _num(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def sabersim_ladder(row):
    """Our 10/25/50/75/90 ladder from one export row, or None if the row
    has no distribution (SaberSim leaves inactive players blank or 0)."""
    p25, p50, p75, p85, p95 = (_num(row.get(f"dk_{k}_percentile")) for k in (25, 50, 75, 85, 95))
    if None in (p25, p50, p75, p85, p95) or (p50 == 0 and p95 == 0):
        return None
    return {
        "10": round(max(0.0, p50 - P10_SPREAD * (p50 - p25)), 2),
        "25": round(p25, 2),
        "50": round(p50, 2),
        "75": round(p75, 2),
        "90": round(p85 + P90_FROM_P85 * (p95 - p85), 2),
    }


def parse_sabersim_csv(path):
    """[{dfs_id, name, team, position, status, proj, own, ladder}] - one per row."""
    rows = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            name = (r.get("Name") or "").strip()
            if not name:
                continue
            rows.append({
                "dfs_id": (r.get("DFS ID") or "").strip(),
                "name": name,
                "team": (r.get("Team") or "").strip(),
                "position": (r.get("Pos") or "").strip(),
                "status": (r.get("Status") or "").strip(),
                "proj": _num(r.get("SS Proj")),
                "own": _num(r.get("Adj Own")),
                "ladder": sabersim_ladder(r),
            })
    return rows


def _scaled(ladder, factor):
    return {k: round(v * factor, 2) for k, v in ladder.items()}


def record_projection_source(conn, slate_id, source, player_id, name, position, ladder, own=None, source_file=None):
    conn.execute(
        text(
            """
            INSERT INTO projection_sources
                (slate_id, source, player_id, name, position, proj_median, proj_percentiles, own_proj, source_file)
            VALUES (:s, :src, :p, :n, :pos, :m, :pp, :own, :f)
            ON CONFLICT (slate_id, source, player_id) DO UPDATE SET
                name = EXCLUDED.name, position = EXCLUDED.position, proj_median = EXCLUDED.proj_median,
                proj_percentiles = EXCLUDED.proj_percentiles, own_proj = EXCLUDED.own_proj,
                source_file = EXCLUDED.source_file, recorded_at = now()
            """
        ),
        {"s": slate_id, "src": source, "p": player_id, "n": name, "pos": position, "m": ladder["50"],
         "pp": json.dumps(ladder), "own": own, "f": source_file},
    )


def apply_sabersim(slate_id, path, engine=None, write_projections=True):
    """Make SaberSim the base projection for `slate_id`.

    1. Snapshots the slate's current projections as source 'model' (only
       the first time, so a re-apply never records SaberSim as 'model').
    2. Records every matched SaberSim row as source 'sabersim'.
    3. If write_projections, overwrites the projections row of each matched
       player with SaberSim's ladder (Captain rows x1.5). Unmatched slate
       players - kickers, players SaberSim projects at 0 - keep ours.

    write_projections=False only records the comparison (for a slate that
    has already been played). Returns a report dict.
    """
    engine = engine or get_engine()
    ss = parse_sabersim_csv(path)
    by_id = {r["dfs_id"]: r for r in ss if r["dfs_id"]}
    by_name_team = {(r["name"], r["team"]): r for r in ss}
    source_file = os.path.basename(path)

    with engine.begin() as conn:
        pool = conn.execute(
            text(
                """
                SELECT p.player_id, trim(p.name) AS name, p.position, p.team, pr.proj_percentiles, pr.locked_at
                FROM slate_player_pool p
                LEFT JOIN projections pr ON pr.slate_id = p.slate_id AND pr.player_id = p.player_id
                WHERE p.slate_id = :s
                """
            ),
            {"s": slate_id},
        ).mappings().fetchall()
        if not pool:
            raise ValueError(f"No slate_player_pool rows for {slate_id}")

        already_snapshotted = conn.execute(
            text("SELECT count(*) FROM projection_sources WHERE slate_id = :s AND source = 'model'"), {"s": slate_id}
        ).scalar()
        matched, kept_ours, locked = [], [], []
        for p in pool:
            ours = p["proj_percentiles"]
            if isinstance(ours, str):
                ours = json.loads(ours)
            if ours and not already_snapshotted:
                record_projection_source(conn, slate_id, "model", p["player_id"], p["name"], p["position"], ours, None, None)

            r = by_id.get(p["player_id"]) or by_name_team.get((p["name"], p["team"]))
            if r is None or r["ladder"] is None:
                kept_ours.append(p["name"])
                continue
            ladder = _scaled(r["ladder"], CAPTAIN_MULTIPLIER) if p["position"] == "CPT" else r["ladder"]
            record_projection_source(conn, slate_id, SOURCE, p["player_id"], p["name"], p["position"], ladder, r["own"], source_file)
            matched.append(p["name"])
            if not write_projections:
                continue
            if p["locked_at"] is not None:
                locked.append(p["name"])
                continue  # locked projections are what a lineup was built against - never rewritten
            conn.execute(
                text(
                    """
                    INSERT INTO projections (slate_id, player_id, proj_floor, proj_median, proj_ceiling, proj_percentiles)
                    VALUES (:s, :p, :f, :m, :c, :pp)
                    ON CONFLICT (slate_id, player_id) DO UPDATE SET
                        proj_floor = EXCLUDED.proj_floor, proj_median = EXCLUDED.proj_median,
                        proj_ceiling = EXCLUDED.proj_ceiling, proj_percentiles = EXCLUDED.proj_percentiles
                    WHERE projections.locked_at IS NULL
                    """
                ),
                {"s": slate_id, "p": p["player_id"], "f": ladder["25"], "m": ladder["50"], "c": ladder["90"],
                 "pp": json.dumps(ladder)},
            )
    return {
        "slate_id": slate_id,
        "source_file": source_file,
        "matched_rows": len(matched),
        "kept_model_rows": len(kept_ours),
        "kept_model_players": sorted(set(kept_ours)),
        "locked_rows_skipped": len(locked),
        "projections_written": write_projections,
    }


def projection_accuracy(slate_id, engine=None):
    """Per-source accuracy against real DK points for an already-played slate.

    Actual points come from imported contest standings (contest_ownership.
    fpts_contest, the FLEX/base value), so import standings first. Compares
    FLEX/Classic rows only, and only players every source projected above 0,
    so each source is scored on the same players.
    """
    engine = engine or get_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT ps.source, ps.name, ps.proj_median::float AS proj, avg(co.fpts_contest)::float AS actual
                FROM projection_sources ps
                JOIN contest_ownership co ON co.slate_id = ps.slate_id AND trim(co.name) = ps.name
                WHERE ps.slate_id = :s AND coalesce(ps.position, '') <> 'CPT' AND co.fpts_contest IS NOT NULL
                GROUP BY ps.source, ps.name, ps.proj_median
                """
            ),
            {"s": slate_id},
        ).fetchall()
    by_source = {}
    for r in rows:
        by_source.setdefault(r.source, {})[r.name] = (r.proj, r.actual)
    if not by_source:
        return {"slate_id": slate_id, "players": 0, "by_source": {}}
    common = set.intersection(*(set(n for n, (p, _) in v.items() if p and p > 0) for v in by_source.values()))
    result = {}
    for source, v in by_source.items():
        errs = [v[n][0] - v[n][1] for n in common]
        result[source] = {
            "mae": round(sum(abs(x) for x in errs) / len(errs), 2) if errs else None,
            "bias": round(sum(errs) / len(errs), 2) if errs else None,
        }
    return {"slate_id": slate_id, "players": len(common), "by_source": result}
