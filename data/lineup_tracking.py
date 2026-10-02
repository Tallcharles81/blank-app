"""Build-method tracking: tag every lineup DK Edge builds with how it was
built (e.g. 'optimizer' vs 'simulator'), score every tagged lineup against
each real contest on its slate from that contest's own standings export,
and compare methods across weeks.

Weekly flow:
  1. Build time:  export_grouped_lineups(slate_id, build_id,
                  {"optimizer": [...], "simulator": [...]}, path)
                  writes the DK upload CSV and registers every lineup.
  2. After games: data/ownership_calibration.py's bulk_import_contest_
                  standings scores registered builds automatically (or call
                  score_builds_against_contest directly).
  3. Report:      compare_build_groups() - per-slate paired comparison and
                  a verdict that stays INSUFFICIENT SAMPLE until enough
                  slates exist to tell the methods apart.

Built lineups are scored whether or not they were entered, so the two
methods are compared on equal terms; the user's actual entries are matched
back to their group separately (late-swapped entries are reported as
edited, with the group they came from).
"""
import csv
import math
import re
from collections import defaultdict

from sqlalchemy import text

from data.dk_salary_csv import CLASSIC_ROSTER, SHOWDOWN_ROSTER, write_dk_upload_csv
from db.migrate import get_engine

DEFAULT_ENTRY_NAME = "tallcharles81"
# Below this many slates with both groups scored, compare_build_groups
# reports INSUFFICIENT SAMPLE instead of a verdict: two slates of 10-vs-10
# correlated lineups pointed in opposite directions when this was built.
MIN_SLATES_FOR_VERDICT = 8
_SLOT_RE = re.compile(r"\s*\b(CPT|QB|RB|WR|TE|FLEX|DST)\s+")


# ---------------------------------------------------------------------------
# Registering builds
# ---------------------------------------------------------------------------

def _normalize_lineup(lineup, names_by_id):
    """-> (slots, player_ids, names) from an optimizer lineup dict
    ({"roster": [(slot, player_dict), ...]}) or a flat list of DK ids in
    upload slot order."""
    if isinstance(lineup, dict):
        roster = lineup["roster"]
        return [s for s, _ in roster], [p["player_id"] for _, p in roster], [p["name"] for _, p in roster]
    ids = list(lineup)
    slots = CLASSIC_ROSTER if len(ids) == len(CLASSIC_ROSTER) else SHOWDOWN_ROSTER
    return list(slots), ids, [names_by_id[pid] for pid in ids]


def register_built_lineups(slate_id, build_id, groups, export_file=None, details=None, engine=None):
    """Record every lineup of one build. `groups` is {group: [lineup, ...]};
    lineups get ids "<group>-01", "<group>-02", ... . `details` optionally
    maps lineup_id -> a note (e.g. which simulation scenario). Re-running
    for the same (slate_id, build_id) replaces that build.
    Returns the number of lineups registered."""
    engine = engine or get_engine()
    details = details or {}
    flat_ids = {pid for lus in groups.values() for lu in lus if not isinstance(lu, dict) for pid in lu}
    names_by_id = {}
    if flat_ids:
        with engine.connect() as conn:
            names_by_id = dict(conn.execute(
                text("SELECT player_id, name FROM slate_player_pool WHERE slate_id = :s AND player_id = ANY(:ids)"),
                {"s": slate_id, "ids": list(flat_ids)},
            ).fetchall())
    rows = []
    for group, lineups in groups.items():
        for i, lu in enumerate(lineups, start=1):
            slots, ids, names = _normalize_lineup(lu, names_by_id)
            lineup_id = f"{group}-{i:02d}"
            rows.append({"slate_id": slate_id, "build_id": build_id, "lineup_id": lineup_id, "build_group": group,
                         "detail": details.get(lineup_id), "slots": slots, "player_ids": ids,
                         "names": [n.strip() for n in names], "export_file": export_file})
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM built_lineups WHERE slate_id = :s AND build_id = :b"), {"s": slate_id, "b": build_id})
        for row in rows:
            conn.execute(text(
                """
                INSERT INTO built_lineups (slate_id, build_id, lineup_id, build_group, detail, slots, player_ids, names, export_file)
                VALUES (:slate_id, :build_id, :lineup_id, :build_group, :detail, :slots, :player_ids, :names, :export_file)
                """
            ), row)
    return len(rows)


def export_grouped_lineups(slate_id, build_id, groups, output_path, details=None, engine=None):
    """Write one DK upload CSV containing every group's lineups (group order
    as given) and register them under `build_id`. Use this instead of
    calling write_dk_upload_csv directly so every exported lineup is
    tracked. Returns {"lineups": n, "by_group": {group: n}}."""
    engine = engine or get_engine()
    ids = []
    for lineups in groups.values():
        for lu in lineups:
            ids.append([p["player_id"] for _, p in lu["roster"]] if isinstance(lu, dict) else list(lu))
    write_dk_upload_csv(slate_id, ids, output_path, engine=engine)
    n = register_built_lineups(slate_id, build_id, groups, export_file=str(output_path), details=details, engine=engine)
    return {"lineups": n, "by_group": {g: len(lus) for g, lus in groups.items()}}


# ---------------------------------------------------------------------------
# Scoring against a real contest
# ---------------------------------------------------------------------------

def parse_standings(csv_path):
    """A DK contest-standings export -> (entries, field_scores_sorted,
    fpts_by_name_and_slot). entries: [{"rank", "entry_id", "entry_name",
    "points", "roster": [(slot, name), ...]}]. The per-player FPTS side
    table is keyed by (name, roster position) so a Showdown CPT row keeps
    its own multiplied score."""
    entries, fpts = [], {}
    with open(csv_path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            player = (row.get("Player") or "").strip()
            if player:
                try:
                    fpts[(player, (row.get("Roster Position") or "").strip())] = float(row.get("FPTS") or 0)
                except ValueError:
                    pass
            rank = (row.get("Rank") or "").strip()
            if not rank:
                continue
            toks = _SLOT_RE.split(" " + (row.get("Lineup") or ""))
            entries.append({"rank": int(rank), "entry_id": row.get("EntryId"), "entry_name": row.get("EntryName") or "",
                            "points": float(row.get("Points") or 0),
                            "roster": list(zip(toks[1::2], (t.strip() for t in toks[2::2])))})
    return entries, sorted(e["points"] for e in entries), fpts


def _player_points(slot, name, fpts):
    """Official FPTS for one roster slot, or None if nobody in the contest
    rostered the player (DK only lists drafted players)."""
    if slot == "CPT":
        if (name, "CPT") in fpts:
            return fpts[(name, "CPT")]
        base = next((v for (n, s), v in fpts.items() if n == name and s != "CPT"), None)
        return None if base is None else 1.5 * base
    val = next((v for (n, s), v in fpts.items() if n == name and s != "CPT"), None)
    if val is None and (name, "CPT") in fpts:
        return fpts[(name, "CPT")] / 1.5
    return val


def _lineup_key(slots, names):
    cpt = tuple(n for s, n in zip(slots, names) if s == "CPT")
    return cpt, frozenset(n for s, n in zip(slots, names) if s != "CPT")


def score_builds_against_contest(csv_path, slate_id, contest_id, entry_name=DEFAULT_ENTRY_NAME, engine=None):
    """Score every lineup registered for `slate_id` against one real contest
    and store the results. A player absent from the export (no one in the
    field rostered him) scores 0 and is counted in missing_players.
    Returns {"scored": n, "field_size": n, "by_group": {...},
    "entries": [...], "no_builds": bool}."""
    engine = engine or get_engine()
    with engine.connect() as conn:
        built = [dict(r) for r in conn.execute(
            text("SELECT build_id, lineup_id, build_group, slots, names FROM built_lineups WHERE slate_id = :s"), {"s": slate_id}
        ).mappings()]
    if not built:
        return {"scored": 0, "no_builds": True}

    entries, field, fpts = parse_standings(csv_path)
    n_field = len(field)
    mine = [e for e in entries if entry_name.lower() in e["entry_name"].lower()]
    mine_keys = defaultdict(int)
    for e in mine:
        mine_keys[_lineup_key([s for s, _ in e["roster"]], [n for _, n in e["roster"]])] += 1

    results = []
    for b in built:
        pts, missing = 0.0, 0
        for slot, name in zip(b["slots"], b["names"]):
            v = _player_points(slot, name, fpts)
            if v is None:
                missing += 1
                v = 0.0
            pts += v
        rank = 1 + sum(1 for s in field if s > pts + 1e-9)
        results.append({**b, "points": round(pts, 2), "finish_rank": rank, "field_size": n_field,
                        "finish_pct": rank / n_field, "missing_players": missing,
                        "times_entered": mine_keys.get(_lineup_key(b["slots"], b["names"]), 0)})

    with engine.begin() as conn:
        for r in results:
            conn.execute(text(
                """
                INSERT INTO lineup_contest_results (contest_id, slate_id, build_id, lineup_id, build_group, points, finish_rank,
                                                    field_size, finish_pct, missing_players, times_entered)
                VALUES (:contest_id, :slate_id, :build_id, :lineup_id, :build_group, :points, :finish_rank,
                        :field_size, :finish_pct, :missing_players, :times_entered)
                ON CONFLICT (contest_id, build_id, lineup_id) DO UPDATE SET
                    points = EXCLUDED.points, finish_rank = EXCLUDED.finish_rank, field_size = EXCLUDED.field_size,
                    finish_pct = EXCLUDED.finish_pct, missing_players = EXCLUDED.missing_players,
                    times_entered = EXCLUDED.times_entered, build_group = EXCLUDED.build_group, scored_at = now()
                """
            ), {**r, "contest_id": contest_id, "slate_id": slate_id})

    # The user's real entries, matched back to a build group. An entry that
    # matches no built lineup exactly is attributed to the built lineup it
    # overlaps most (a late swap), reported as edited.
    built_keys = {(_lineup_key(b["slots"], b["names"])): b for b in built}
    entry_rows = []
    for e in sorted(mine, key=lambda e: e["rank"]):
        key = _lineup_key([s for s, _ in e["roster"]], [n for _, n in e["roster"]])
        exact = built_keys.get(key)
        if exact:
            source = f"{exact['build_group']} ({exact['build_id']}/{exact['lineup_id']})"
        else:
            names = {n for _, n in e["roster"]}
            best = max(built, key=lambda b: len(names & set(b["names"])))
            overlap = len(names & set(best["names"]))
            source = f"edited from {best['build_group']} ({overlap}/{len(names)} same)" if overlap >= len(names) - 2 else "not a DK Edge build"
        entry_rows.append({"rank": e["rank"], "points": e["points"], "finish_pct": e["rank"] / n_field, "source": source})

    by_group = defaultdict(list)
    for r in results:
        by_group[r["build_group"]].append(r)
    summary = {g: {"lineups": len(rs), "avg_points": round(sum(r["points"] for r in rs) / len(rs), 2),
                   "avg_finish_pct": round(sum(r["finish_pct"] for r in rs) / len(rs), 4),
                   "best_rank": min(r["finish_rank"] for r in rs),
                   "beat_field_median": sum(r["finish_pct"] < 0.5 for r in rs),
                   "top_10pct": sum(r["finish_pct"] <= 0.10 for r in rs)} for g, rs in by_group.items()}
    return {"scored": len(results), "field_size": n_field, "by_group": summary, "entries": entry_rows, "no_builds": False}


# ---------------------------------------------------------------------------
# Comparing methods across weeks
# ---------------------------------------------------------------------------

def compare_build_groups(group_a="optimizer", group_b="simulator", slate_ids=None, engine=None):
    """Per slate: each group's mean finish percentile and points (averaged
    over every scored contest on that slate), then the paired difference
    (group_b minus group_a) across slates. Negative finish-pct difference =
    group_b finished better. Verdict is INSUFFICIENT SAMPLE until
    MIN_SLATES_FOR_VERDICT slates have both groups."""
    engine = engine or get_engine()
    sql = "SELECT slate_id, contest_id, build_group, points, finish_pct FROM lineup_contest_results"
    params = {}
    if slate_ids is not None:
        sql += " WHERE slate_id = ANY(:slates)"
        params["slates"] = list(slate_ids)
    with engine.connect() as conn:
        rows = conn.execute(text(sql), params).fetchall()

    per = defaultdict(lambda: defaultdict(list))
    for r in rows:
        per[r.slate_id][r.build_group].append((float(r.finish_pct), float(r.points)))

    slates = []
    for slate, groups in sorted(per.items()):
        if group_a not in groups or group_b not in groups:
            continue
        a, b = groups[group_a], groups[group_b]
        entry = {"slate_id": slate}
        for name, vals in ((group_a, a), (group_b, b)):
            entry[name] = {"avg_finish_pct": sum(v[0] for v in vals) / len(vals), "avg_points": sum(v[1] for v in vals) / len(vals),
                           "beat_median_rate": sum(v[0] < 0.5 for v in vals) / len(vals), "top_10pct_rate": sum(v[0] <= 0.10 for v in vals) / len(vals)}
        entry["finish_pct_diff"] = entry[group_b]["avg_finish_pct"] - entry[group_a]["avg_finish_pct"]
        entry["points_diff"] = entry[group_b]["avg_points"] - entry[group_a]["avg_points"]
        slates.append(entry)

    n = len(slates)
    out = {"group_a": group_a, "group_b": group_b, "n_slates": n, "slates": slates, "min_slates_for_verdict": MIN_SLATES_FOR_VERDICT}
    if n == 0:
        out["verdict"] = "INSUFFICIENT SAMPLE (no slate has both groups scored yet)"
        return out
    diffs = [s["finish_pct_diff"] for s in slates]
    mean = sum(diffs) / n
    out["mean_finish_pct_diff"] = mean
    out["mean_points_diff"] = sum(s["points_diff"] for s in slates) / n
    out["slates_group_b_better"] = sum(d < 0 for d in diffs)
    if n >= 2:
        sd = math.sqrt(sum((d - mean) ** 2 for d in diffs) / (n - 1))
        z = mean / (sd / math.sqrt(n)) if sd > 0 else 0.0
        out["p"] = math.erfc(abs(z) / math.sqrt(2))
    if n < MIN_SLATES_FOR_VERDICT:
        out["verdict"] = f"INSUFFICIENT SAMPLE ({n} of {MIN_SLATES_FOR_VERDICT} slates needed)"
    elif out.get("p", 1) < 0.05:
        better = group_b if mean < 0 else group_a
        out["verdict"] = f"{better.upper()} FINISHES BETTER (p = {out['p']:.3f}, {n} slates)"
    else:
        out["verdict"] = f"NO DETECTABLE DIFFERENCE ({n} slates, p = {out.get('p', float('nan')):.2f})"
    return out


def compare_top_pick_to_pool(slate_ids=None, min_pool=10, engine=None):
    """Would a single entry do better taking the simulator's #1 lineup
    (simulator-01, generate_simulation_selected_lineup's ranked[0]) from the
    main build than an average lineup from that same build? Per slate and
    build with at least `min_pool` lineups: the pool's mean finish
    percentile, simulator-01's and optimizer-01's, each averaged over every
    scored contest on the slate. One-lineup builds (dedicated single-entry
    searches) are listed beside them for comparison.

    Added after PIT@CLE 2026-10-01, where the main build's simulator-01
    finished top 10.6% while the separate single-entry searches finished
    top 24-59%. Same sample rule as compare_build_groups: INSUFFICIENT
    SAMPLE until MIN_SLATES_FOR_VERDICT slates."""
    engine = engine or get_engine()
    sql = "SELECT slate_id, build_id, lineup_id, finish_pct FROM lineup_contest_results"
    params = {}
    if slate_ids is not None:
        sql += " WHERE slate_id = ANY(:slates)"
        params["slates"] = list(slate_ids)
    with engine.connect() as conn:
        rows = conn.execute(text(sql), params).fetchall()

    per = defaultdict(lambda: defaultdict(list))  # (slate, build) -> lineup -> [finish_pct per contest]
    for r in rows:
        per[(r.slate_id, r.build_id)][r.lineup_id].append(float(r.finish_pct))

    def avg(vals):
        return sum(vals) / len(vals)

    builds, singles = [], []
    for (slate, build), lineups in sorted(per.items()):
        means = {lid: avg(v) for lid, v in lineups.items()}
        if len(means) == 1:
            singles.append({"slate_id": slate, "build_id": build, "finish_pct": next(iter(means.values()))})
        elif len(means) >= min_pool and "simulator-01" in means:
            builds.append({"slate_id": slate, "build_id": build, "pool_size": len(means),
                           "pool_avg_finish_pct": avg(list(means.values())),
                           "simulator_01_finish_pct": means["simulator-01"],
                           "optimizer_01_finish_pct": means.get("optimizer-01")})

    # One number per slate (its largest build), so a slate with two builds isn't counted twice.
    by_slate = {}
    for b in builds:
        if b["slate_id"] not in by_slate or b["pool_size"] > by_slate[b["slate_id"]]["pool_size"]:
            by_slate[b["slate_id"]] = b
    diffs = [b["simulator_01_finish_pct"] - b["pool_avg_finish_pct"] for b in by_slate.values()]
    n = len(diffs)
    out = {"builds": builds, "single_entry_builds": singles, "n_slates": n, "min_slates_for_verdict": MIN_SLATES_FOR_VERDICT}
    if n:
        out["mean_top_pick_minus_pool"] = avg(diffs)
        out["slates_top_pick_better"] = sum(d < 0 for d in diffs)
    if n < MIN_SLATES_FOR_VERDICT:
        out["verdict"] = f"INSUFFICIENT SAMPLE ({n} of {MIN_SLATES_FOR_VERDICT} slates needed)"
    else:
        mean = avg(diffs)
        sd = math.sqrt(sum((d - mean) ** 2 for d in diffs) / (n - 1))
        z = mean / (sd / math.sqrt(n)) if sd > 0 else 0.0
        out["p"] = math.erfc(abs(z) / math.sqrt(2))
        if out["p"] < 0.05:
            out["verdict"] = f"TOP PICK {'BEATS' if mean < 0 else 'TRAILS'} THE POOL (p = {out['p']:.3f}, {n} slates)"
        else:
            out["verdict"] = f"NO DETECTABLE DIFFERENCE ({n} slates, p = {out['p']:.2f})"
    return out
