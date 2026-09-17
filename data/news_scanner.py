import json
import re

from data.ai_client import DEFAULT_MODEL, get_anthropic_client
from data.pre_lock_check import build_pre_lock_checklist, record_finding
from db.migrate import get_engine

# ---------------------------------------------------------------------------
# The automated version of the "research step" data/pre_lock_check.py's own
# module docstring describes as human/agent-driven ("There is no web-search
# capability inside this codebase's Python runtime"). This module gives the
# CODEBASE its own real search access (Claude's server-side web_search tool,
# via a real Anthropic API account - see data/ai_client.py) so the same
# per-game injury/inactive/starter-status sweep that procedure describes can
# run on its own, repeatedly, instead of needing an interactive agent to work
# through build_pre_lock_checklist() by hand every time.
#
# This does NOT replace get_availability_gate() or hard_role_exclusions() -
# same as the rest of this module, it's a cross-check that calls the
# existing record_finding()/get_contradictions() machinery, not a new gate.
#
# "Continuous... throughout the week" is a scheduling property, not something
# this function can do by itself - a single call scans the CURRENT state of
# the eligible pool once. Real continuous coverage means calling
# scan_slate_news() repeatedly over the week (a cron job, a scheduled task -
# see scripts/run_news_scan.py), each run's findings landing in the same
# pre_lock_checks table so get_contradictions()/summarize_checks() show the
# accumulated real history, not just the latest pass.
#
# IMPORTANT, real limitation worth knowing before running this against this
# specific project's own slate data: Claude's web_search tool searches the
# REAL, current internet. This codebase's own "current" slates (e.g.
# dk_thu_mon_2026_09_17) are a simulated/fictional continuation of the NFL
# season for this project's own testing purposes, with real player names but
# a storyline that has already diverged from reality (confirmed for real
# this session - e.g. Kenneth Walker III and Justin Fields appearing on
# rosters they are not really on). Searching the real internet against that
# fictional data will surface REAL news that contradicts the FICTIONAL
# storyline for reasons that have nothing to do with a real data-quality
# problem - every such "contradiction" on a fictional slate is expected
# noise, not a real finding. This module is built and tested to be real and
# correct; whether a given slate's data is real or fictional is the caller's
# call, not something this module can detect on its own.
# ---------------------------------------------------------------------------

_SOURCE_CATEGORIES = {"injury_news", "inactive_list", "weather", "starter_status"}

_SYSTEM_PROMPT = """You are a real-time NFL news researcher supporting a daily fantasy \
sports lineup tool. You will be given one real NFL game (two teams) and the list of \
players from that game currently in the tool's eligible player pool, each with the \
tool's own current gate verdict (CLEAR, or a flagged status like Questionable/Out).

Use real web search to check for CURRENT, real news relevant to these players' \
availability or role for this specific upcoming game: injury reports, inactive lists, \
practice participation, beat-reporter reports on who is confirmed starting, and \
(if the game is outdoors) severe weather that could affect the game. Only report a \
claim you can attribute to a specific, checkable real source (a team's official site, \
a named beat reporter, ESPN, NFL.com, or similar) - never speculate or guess.

After searching, respond with ONLY a JSON array (no other text) of every real, \
concrete finding you turned up, in this exact shape:
[
  {"player_name": "<exact name as given>", "source": "injury_news|inactive_list|weather|starter_status", \
"finding": "<the real claim, in your own words, naming the source>", "contradicts_gate": true|false}
]
"contradicts_gate" is true only when the finding conflicts with that player's own \
listed gate verdict, or contradicts the assumption that a rostered, healthy player is \
this week's actual starter. If you found nothing concrete for a player, omit them \
entirely - do not include a "nothing found" entry. If you found nothing at all for \
the whole game, respond with an empty JSON array: []"""


def _extract_json_array(text_response):
    # Claude is instructed to respond with ONLY the array, but real model
    # output can still wrap it in prose or a code fence despite the
    # instruction - take the last top-level [...] block in the response
    # rather than assuming byte-exact compliance.
    matches = re.findall(r"\[.*\]", text_response, re.DOTALL)
    if not matches:
        return []
    return json.loads(matches[-1])


def scan_game_news(client, model, team, opponent, players, game_time):
    """One real API call (with Claude's server-side web_search tool) for a
    single real game matchup - see this module's docstring for why per-game,
    not per-player. `players` is build_pre_lock_checklist's own per-game
    player list (player_id/name/position/team/salary/gate_status already
    attached). Returns a list of real findings matched back to a player_id:
    {"player_id", "name", "source", "finding", "gate_status_at_check",
    "contradicts_gate"} - a finding whose player_name doesn't match anyone
    in `players` (a real web search result naming someone outside this
    exact roster, or a name-format mismatch) is returned separately under
    "unmatched_names" rather than silently dropped.
    """
    player_lines = "\n".join(
        f"- {p['name']} ({p['position']}, {p['team']}) - current gate status: {p['gate_status']}"
        for p in players
    )
    user_message = (
        f"Game: {team} vs {opponent}, kickoff {game_time}.\n\n"
        f"Players in the eligible pool for this game:\n{player_lines}"
    )

    response = client.messages.create(
        model=model,
        max_tokens=4096,
        system=_SYSTEM_PROMPT,
        tools=[{"type": "web_search_20260209", "name": "web_search", "max_uses": 5}],
        messages=[{"role": "user", "content": user_message}],
    )

    text_blocks = [block.text for block in response.content if block.type == "text"]
    findings_raw = _extract_json_array("\n".join(text_blocks))

    by_name = {p["name"].strip().lower(): p for p in players}
    findings = []
    unmatched_names = []
    for f in findings_raw:
        name = (f.get("player_name") or "").strip()
        source = f.get("source")
        finding = f.get("finding")
        contradicts_gate = bool(f.get("contradicts_gate"))
        if not name or source not in _SOURCE_CATEGORIES or not finding:
            continue  # not a real, well-formed finding - skip rather than store garbage
        player = by_name.get(name.lower())
        if player is None:
            unmatched_names.append(f)
            continue
        findings.append(
            {
                "player_id": player["player_id"],
                "name": player["name"],
                "source": source,
                "finding": finding,
                "gate_status_at_check": player["gate_status"],
                "contradicts_gate": contradicts_gate,
            }
        )
    return findings, unmatched_names


def scan_slate_news(slate_id, engine=None, client=None, model=DEFAULT_MODEL):
    """Top-level entry point - one real news scan pass over EVERY currently-
    eligible player in `slate_id`, grouped by game (via build_pre_lock_
    checklist), storing every real finding via record_finding() the same
    way a human running the manual procedure would. Safe to call repeatedly
    (e.g. from a scheduled job - see scripts/run_news_scan.py) - each call
    is a fresh real search pass, and pre_lock_checks accumulates every run.

    One game's search failing (rate limit, network, a malformed response)
    is recorded and skipped rather than aborting the whole slate's scan -
    same convention as every other external-data fetch in this codebase.
    """
    engine = engine or get_engine()
    client = client or get_anthropic_client()

    games, meta = build_pre_lock_checklist(slate_id, engine)

    games_scanned = 0
    findings_recorded = 0
    contradictions_found = 0
    games_failed = []
    all_unmatched_names = []

    for (team, opponent, game_time), game_data in games.items():
        try:
            findings, unmatched_names = scan_game_news(
                client, model, team, opponent, game_data["players"], game_time
            )
        except Exception as exc:
            games_failed.append({"team": team, "opponent": opponent, "error": str(exc)})
            continue

        for f in findings:
            record_finding(
                slate_id,
                f["player_id"],
                f["source"],
                f["finding"],
                f["gate_status_at_check"],
                f["contradicts_gate"],
                engine,
            )
            findings_recorded += 1
            if f["contradicts_gate"]:
                contradictions_found += 1
        all_unmatched_names.extend(unmatched_names)
        games_scanned += 1

    return {
        "season": meta["season"],
        "week": meta["week"],
        "games_scanned": games_scanned,
        "games_failed": games_failed,
        "findings_recorded": findings_recorded,
        "contradictions_found": contradictions_found,
        "unmatched_names": all_unmatched_names,
    }
