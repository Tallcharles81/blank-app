import re
from collections import defaultdict

from sqlalchemy import text

from db.migrate import get_engine

# DraftKings' player IDs and nflverse's GSIS IDs are different, unrelated ID
# schemes with no published crosswalk between them (see the KNOWN GAP note in
# models/projections.py). Name matching is the standard fallback every DFS tool
# that combines the two ends up using. Team is deliberately NOT part of the
# match key: a player's current (DK slate) team can differ from their team in
# older nflverse history after a trade or free-agent signing, and requiring
# equality there would silently drop otherwise-good matches.
_SUFFIX_RE = re.compile(r"\s+(jr|sr|ii|iii|iv|v)$")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9\s]")
_WHITESPACE_RE = re.compile(r"\s+")


def normalize_name(name):
    name = name.lower().strip().replace("-", " ")
    name = _NON_ALNUM_RE.sub("", name)
    name = _SUFFIX_RE.sub("", name)
    return _WHITESPACE_RE.sub(" ", name).strip()


def _nflverse_players_by_name_position(engine):
    query = text("SELECT DISTINCT player_id AS gsis_id, player_name, position FROM player_weekly_stats")
    with engine.connect() as conn:
        rows = conn.execute(query).fetchall()

    candidates = defaultdict(set)
    for row in rows:
        key = (normalize_name(row.player_name), row.position)
        candidates[key].add(row.gsis_id)
    return candidates


def resolve_dk_players_to_gsis(dk_players, engine=None):
    """Match DK players (dicts with player_id/name/position) to nflverse GSIS IDs by
    normalized name + position.

    Returns (mapping, unmatched, ambiguous):
    - mapping: {dk_player_id: gsis_id} for confident single matches.
    - unmatched: dk_player_ids with no candidate at all - expected for rookies
      with no prior-season history, and for DST, since nflverse's player_stats
      release has no team-defense rows (it's a per-player stats file); there is
      no historical source fetched for DST scoring yet.
    - ambiguous: dk_player_ids whose normalized name+position matched more than
      one distinct historical player - too risky to guess, so left unmatched
      rather than silently picking one.
    """
    engine = engine or get_engine()
    candidates = _nflverse_players_by_name_position(engine)

    mapping = {}
    unmatched = []
    ambiguous = []
    for player in dk_players:
        key = (normalize_name(player["name"]), player["position"])
        matches = candidates.get(key, set())
        if len(matches) == 1:
            mapping[player["player_id"]] = next(iter(matches))
        elif len(matches) == 0:
            unmatched.append(player["player_id"])
        else:
            ambiguous.append(player["player_id"])

    return mapping, unmatched, ambiguous
