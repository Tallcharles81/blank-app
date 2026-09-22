import re
from collections import defaultdict

from sqlalchemy import text

from data.nflverse_fetch import to_nflverse_team
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

# DraftKings labels EVERY row of a Showdown slate's player pool "CPT" or
# "FLEX" in the same Position field that holds a real football position
# (QB/RB/WR/TE/DST) on a Classic slate - see models/optimizer.py's own
# p["position"] == "CPT" checks, which already depend on this. Neither
# value is ever a real position in player_weekly_stats, so the (name,
# position) match key below can never succeed for a Showdown row - it was
# silently falling through to "unmatched" for every Showdown player,
# which meant the availability gate and the role-exclusion gate (both of
# which resolve identity through this function) were structurally unable
# to exclude ANYONE on a Showdown slate. Caught before any real Showdown
# slate had been run - see resolve_dk_players_to_gsis for the fix.
SHOWDOWN_PSEUDO_POSITIONS = {"CPT", "FLEX"}


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


def _nflverse_players_by_name(engine):
    # Same real historical rows as _nflverse_players_by_name_position, but
    # keyed on name alone - needed for a Showdown row, whose real position
    # can't be read from the CSV at all (see SHOWDOWN_PSEUDO_POSITIONS), so
    # matching can only fall back to name. A name matching more than one
    # distinct real position (e.g. two different real players who share a
    # name) is handled the same as any other ambiguous match - left
    # unmatched rather than guessed.
    query = text(
        "SELECT DISTINCT player_id AS gsis_id, player_name FROM player_weekly_stats WHERE position != 'DST'"
    )
    with engine.connect() as conn:
        rows = conn.execute(query).fetchall()

    candidates = defaultdict(set)
    for row in rows:
        candidates[normalize_name(row.player_name)].add(row.gsis_id)
    return candidates


def _existing_dst_ids(engine):
    query = text("SELECT DISTINCT player_id FROM player_weekly_stats WHERE position = 'DST'")
    with engine.connect() as conn:
        return {row.player_id for row in conn.execute(query)}


def _dst_nicknames_by_team(engine):
    # The real team nickname DK itself displays for a defense (e.g.
    # "Seahawks" for SEA) - DK doesn't document this mapping anywhere, so
    # rather than hardcode or guess at it, this is sourced from OUR OWN
    # already-loaded Classic slate_player_pool rows, where a DST row's
    # `position` field IS trustworthy (Classic never uses the CPT/FLEX
    # pseudo-position) and its `name` field is confirmed to be the real
    # nickname (see resolve_dk_players_to_gsis's existing Classic DST
    # comment). Used to recognize a Showdown DST row, whose `position`
    # field is uselessly "CPT"/"FLEX" the same as every other Showdown row.
    # Returns {} if no Classic slate has ever been loaded - a real,
    # disclosed limitation (a Showdown-only deployment with zero Classic
    # history would have nothing to cross-reference against), not silently
    # papered over.
    query = text("SELECT DISTINCT team, name FROM slate_player_pool WHERE position = 'DST'")
    with engine.connect() as conn:
        return {row.team: row.name for row in conn.execute(query)}


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
      rather than silently picking one. Never happens for a Classic DST, which
      matches on team code instead of name.

    Showdown rows (position "CPT"/"FLEX" - see SHOWDOWN_PSEUDO_POSITIONS)
    can't use the (name, position) key at all, since DK never reports a
    real position for them. Matched by name alone instead, after first
    checking whether the row is actually a defense via the real team
    nickname cross-reference in _dst_nicknames_by_team - a defense's
    "name" (the team nickname) would otherwise just fail every real-person
    name lookup and land in `unmatched`, silently reproducing the same
    gap this fix exists to close.
    """
    engine = engine or get_engine()
    candidates = _nflverse_players_by_name_position(engine)
    name_only_candidates = _nflverse_players_by_name(engine)
    existing_dst_ids = _existing_dst_ids(engine)
    dst_nicknames_by_team = _dst_nicknames_by_team(engine)

    mapping = {}
    unmatched = []
    ambiguous = []
    for player in dk_players:
        if player["position"] == "DST":
            # DK's DST name is the team nickname only (e.g. "49ers"), which has
            # no reliable, guessable relationship to any naming convention we'd
            # invent for the synthetic rows refresh_dst_weekly_stats() writes -
            # team code is the one unambiguous, stable key both sides share -
            # normalized to nflverse's own team convention first (DK and
            # nflverse don't always agree - see data/nflverse_fetch.py's
            # DK_TO_NFLVERSE_TEAM, found for real via the Rams: DK's "LAR"
            # vs nflverse's "LA", which silently orphaned this exact lookup).
            dst_id = f"DST_{to_nflverse_team(player['team'])}"
            if dst_id in existing_dst_ids:
                mapping[player["player_id"]] = dst_id
            else:
                unmatched.append(player["player_id"])
            continue

        if player["position"] in SHOWDOWN_PSEUDO_POSITIONS:
            dst_id = f"DST_{to_nflverse_team(player['team'])}"
            real_nickname = dst_nicknames_by_team.get(player["team"])
            if (
                dst_id in existing_dst_ids
                and real_nickname is not None
                and normalize_name(player["name"]) == normalize_name(real_nickname)
            ):
                mapping[player["player_id"]] = dst_id
                continue

            matches = name_only_candidates.get(normalize_name(player["name"]), set())
            if len(matches) == 1:
                mapping[player["player_id"]] = next(iter(matches))
            elif len(matches) == 0:
                unmatched.append(player["player_id"])
            else:
                ambiguous.append(player["player_id"])
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
