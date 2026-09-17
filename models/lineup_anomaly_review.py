import json

from sqlalchemy import text

from data.ai_client import DEFAULT_MODEL, get_anthropic_client
from data.pre_lock_check import build_lineup_role_checklist
from db.migrate import get_engine

# ---------------------------------------------------------------------------
# The second-pair-of-eyes step this session's own human review kept catching
# real problems the hard gates alone didn't - a Marcus Mariota lineup
# mislabeled as Jameis Winston, a synthetic test row shaped like an illegal
# roster, a punt-priced Captain that only made sense because the CPT
# multiplier was silently missing from the projections. None of those were
# caught by a rule; they were caught by someone who plays real football
# looking at the actual roster and noticing it looked wrong. This function is
# that same read, done by an LLM instead of a human, run AFTER a lineup is
# already built and AFTER it already passed every hard gate - never a
# replacement for hard_role_exclusions/get_availability_gate/validate_lineup,
# only one more check layered on top, the same way build_lineup_role_
# checklist's own advisory HIGH/MEDIUM/LOW flags already are.
#
# Deliberately does NOT use web search or any live data - this is pure
# football-knowledge reasoning over data this codebase has already computed
# for real (salary, position, real recent role flags), the same read a
# knowledgeable human gives a finished roster. No fictional-vs-real-season
# conflict the way data/news_scanner.py has, since nothing here depends on
# what's actually happening in the real world right now.
# ---------------------------------------------------------------------------

_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "flags": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "severity": {"type": "string", "enum": ["HIGH", "MEDIUM", "LOW"]},
                    "concern": {"type": "string"},
                    "players_involved": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["severity", "concern", "players_involved"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["flags"],
    "additionalProperties": False,
}

_SYSTEM_PROMPT = """You are a knowledgeable, skeptical daily fantasy football veteran \
doing a final human-style sanity check on a lineup a solver already built and that \
already passed this tool's own hard gates (roster legality, injury/role exclusions). \
Your job is the same one a sharp human player does before hitting submit: does \
anything here look wrong to someone who actually knows football and this format? \
Look for things like: a player whose real position doesn't fit how they're being \
used, a "stack" that doesn't make real football sense (no real connection between \
the players paired together), salary usage that looks like a mistake (a lot of cap \
space left unused with no good reason, or a bafflingly cheap player at a premium \
slot for no real reason), a name that seems out of place for this specific game, or \
anything else that would make a knowledgeable person pause and go double-check. Do \
NOT re-flag anything the tool's own role-check already flagged (given to you \
separately below) - focus on what a second, independent set of eyes would add. If \
nothing looks off, return an empty flags array - do not invent a concern just to \
have something to say."""


def review_lineup_for_anomalies(lineup, slate_id, engine=None, client=None, model=DEFAULT_MODEL):
    """Real post-build AI review of one already-built, already-validated
    lineup (the `roster` shape models/optimizer.py's build_lineups_from_pool/
    generate_lineups return - a list of (slot, player_dict) pairs). Returns
    {"flags": [{"severity", "concern", "players_involved"}], "existing_role_
    check_flags": [...]} - the existing_role_check_flags key is what this
    function told Claude NOT to re-flag, included in the return so a caller
    can see the full real picture in one place without a second call.

    This is advisory only - it never blocks or modifies the lineup. Calling
    it is a deliberate, separate, real-API-billed step, not something wired
    automatically into every build (see this module's own docstring above).
    """
    engine = engine or get_engine()
    client = client or get_anthropic_client()

    player_ids = [p["player_id"] for _, p in lineup["roster"]]
    role_flags = [
        r for r in build_lineup_role_checklist(slate_id, player_ids, engine) if r["needs_check"]
    ]

    with engine.connect() as conn:
        proj_rows = conn.execute(
            text("SELECT player_id, proj_median, proj_ceiling FROM projections WHERE slate_id = :slate_id"),
            {"slate_id": slate_id},
        ).mappings().fetchall()
    proj_by_id = {row["player_id"]: dict(row) for row in proj_rows}

    roster_lines = []
    for slot, p in lineup["roster"]:
        proj = proj_by_id.get(p["player_id"])
        proj_str = (
            f", proj_median={proj['proj_median']}, proj_ceiling={proj['proj_ceiling']}"
            if proj is not None
            else ", no stored projection"
        )
        roster_lines.append(f"- [{slot}] {p['name']} ({p['position']}, {p['team']}) ${p['salary']}{proj_str}")

    role_flag_lines = (
        "\n".join(f"- {r['name']}: {r['severity']} - {r['reason']}" for r in role_flags)
        if role_flags
        else "(none)"
    )

    user_message = (
        f"Roster (total salary ${lineup['total_salary']}):\n"
        + "\n".join(roster_lines)
        + "\n\nThis tool's own role-check already flagged (do not repeat these):\n"
        + role_flag_lines
    )

    response = client.messages.create(
        model=model,
        max_tokens=2048,
        system=_SYSTEM_PROMPT,
        output_config={"format": {"type": "json_schema", "schema": _OUTPUT_SCHEMA}},
        messages=[{"role": "user", "content": user_message}],
    )
    text_block = next(b.text for b in response.content if b.type == "text")
    parsed = json.loads(text_block)

    return {"flags": parsed["flags"], "existing_role_check_flags": role_flags}
