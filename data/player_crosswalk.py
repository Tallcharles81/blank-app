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
    query = text(
        "SELECT DISTINCT player_id AS gsis_id, player_name, position "
        "FROM player_weekly_stats WHERE position != 'DST'"
    )
    with engine.connect() as conn:
        rows = conn.execute(query).fetchall()

    candidates = defaultdict(set)
    for row in rows:
        key = (normalize_name(row.player_name), row.position)
        candidates[key].add(row.gsis_id)
    return candidates


def _existing_dst_ids(engine):
    query = text("SELECT DISTINCT player_id FROM player_weekly_stats WHERE position = 'DST'")
    with engine.connect() as conn:
        return {row.player_id for row in conn.execute(query)}


def resolve_dk_players_to_gsis(dk_players, engine=None):
    """Match DK players (dicts with player_id/name/position/team) to their
    corresponding player_weekly_stats id.

    Returns (mapping, unmatched, ambiguous):
    - mapping: {dk_player_id: player_weekly_stats.player_id} for confident matches.
    - unmatched: dk_player_ids with no candidate at all - expected for rookies
      with no prior-season history, or a DST whose team has no
      refresh_dst_weekly_stats() data for the relevant seasons.
    - ambiguous: dk_player_ids whose normalized name+position matched more than
      one distinct historical player - too risky to guess, so left unmatched
      rather than silently picking one. Never happens for DST, which matches
      on team code instead of name.
    """
    engine = engine or get_engine()
    candidates = _nflverse_players_by_name_position(engine)
    existing_dst_ids = _existing_dst_ids(engine)

    mapping = {}
    unmatched = []
    ambiguous = []
    for player in dk_players:
        if player["position"] == "DST":
            # DK's DST name is the team nickname only (e.g. "49ers"), which has
            # no reliable, guessable relationship to any naming convention we'd
            # invent for the synthetic rows refresh_dst_weekly_stats() writes -
            # team code is the one unambiguous, stable key both sides share.
            dst_id = f"DST_{player['team']}"
            if dst_id in existing_dst_ids:
                mapping[player["player_id"]] = dst_id
            else:
                unmatched.append(player["player_id"])
            continue

        key = (normalize_name(player["name"]), player["position"])
        matches = candidates.get(key, set())
        if len(matches) == 1:
            mapping[player["player_id"]] = next(iter(matches))
        elif len(matches) == 0:
            unmatched.append(player["player_id"])
        else:
            ambiguous.append(player["player_id"])

    return mapping, unmatched, ambiguous
