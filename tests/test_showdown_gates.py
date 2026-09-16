import pytest
from sqlalchemy import text

from data.player_availability import get_availability_gate
from data.player_crosswalk import resolve_dk_players_to_gsis
from data.pre_lock_check import build_lineup_role_checklist, hard_role_exclusions

# Regression coverage for a real gap: DraftKings labels EVERY row of a
# Showdown slate's player pool "CPT" or "FLEX" in the same field that
# carries a real position (QB/RB/WR/TE/DST) on a Classic slate. Both hard
# gates originally trusted that field directly, so on a Showdown slate:
#   - the crosswalk's (name, position) match key could never succeed for
#     any player, and its position=="DST" check could never succeed for a
#     defense either -> resolve_dk_players_to_gsis returned every single
#     Showdown player as "unmatched"
#   - get_availability_gate's position != "DST" exemption for defenses
#     also never fired
#   - hard_role_exclusions' position != "DST" filter and position == "QB"
#     branch never fired either
# The net effect: neither hard gate could exclude ANYONE on a Showdown
# slate. Never exercised for real before this, since every slate discussed
# this session was Classic. No real DK Showdown CSV export was available to
# test against, so these use realistic Showdown-shaped rows (position CPT/
# FLEX, a real player appearing as both a CPT and a FLEX row, same as DK's
# real export) built from real player identities already in this session's
# data, rather than a synthetic stand-in with made-up names.

WINSTON_ROWS = [
    {"player_id": "SD_CPT_WINSTON", "name": "Jameis Winston", "position": "CPT", "team": "NYG"},
    {"player_id": "SD_FLEX_WINSTON", "name": "Jameis Winston", "position": "FLEX", "team": "NYG"},
]
AIYUK_ROWS = [
    {"player_id": "SD_CPT_AIYUK", "name": "Brandon Aiyuk", "position": "CPT", "team": "SF"},
    {"player_id": "SD_FLEX_AIYUK", "name": "Brandon Aiyuk", "position": "FLEX", "team": "SF"},
]
CLEAN_ROWS = [
    {"player_id": "SD_CPT_LAWRENCE", "name": "Trevor Lawrence", "position": "CPT", "team": "JAX"},
    {"player_id": "SD_FLEX_MCCAFFREY", "name": "Christian McCaffrey", "position": "FLEX", "team": "SF"},
]
DST_ROWS = [
    {"player_id": "SD_CPT_DST", "name": "Seahawks", "position": "CPT", "team": "SEA"},
    {"player_id": "SD_FLEX_DST", "name": "Seahawks", "position": "FLEX", "team": "SEA"},
]


def test_crosswalk_resolves_showdown_player_rows_by_name(engine):
    mapping, unmatched, ambiguous = resolve_dk_players_to_gsis(WINSTON_ROWS + CLEAN_ROWS, engine)
    assert not unmatched, f"real players should never be unmatched on a Showdown slate, got {unmatched}"
    assert not ambiguous
    # Both the CPT and FLEX row for the same real person must resolve to the same gsis id.
    assert mapping["SD_CPT_WINSTON"] == mapping["SD_FLEX_WINSTON"]


def test_crosswalk_resolves_showdown_dst_rows_via_real_nickname(engine):
    mapping, unmatched, ambiguous = resolve_dk_players_to_gsis(DST_ROWS, engine)
    assert not unmatched
    assert not ambiguous
    assert mapping["SD_CPT_DST"] == "DST_SEA"
    assert mapping["SD_FLEX_DST"] == "DST_SEA"


def test_availability_gate_catches_a_known_absent_player_on_a_showdown_slate():
    excluded, _flagged, _injury_report_available = get_availability_gate(AIYUK_ROWS + CLEAN_ROWS, 2026, 2)
    assert "SD_CPT_AIYUK" in excluded
    assert "SD_FLEX_AIYUK" in excluded
    assert "SD_CPT_LAWRENCE" not in excluded
    assert "SD_FLEX_MCCAFFREY" not in excluded


def test_availability_gate_does_not_wrongly_exclude_a_showdown_dst():
    excluded, _flagged, _injury_report_available = get_availability_gate(DST_ROWS, 2026, 2)
    assert not excluded, f"a real Showdown defense must never be excluded as a 'missing person', got {excluded}"


def test_hard_role_exclusions_catches_a_known_backup_qb_on_a_showdown_slate(engine):
    excluded = hard_role_exclusions(WINSTON_ROWS + CLEAN_ROWS, engine)
    assert "SD_CPT_WINSTON" in excluded
    assert "SD_FLEX_WINSTON" in excluded
    assert "SD_CPT_LAWRENCE" not in excluded
    assert "SD_FLEX_MCCAFFREY" not in excluded


def test_hard_role_exclusions_skips_a_showdown_dst_entirely(engine):
    excluded = hard_role_exclusions(DST_ROWS, engine)
    assert not excluded, f"a real Showdown defense must never be evaluated as a role-exclusion case, got {excluded}"


TEST_SHOWDOWN_SLATE_ID = "TEST_SHOWDOWN_ROLE_CHECKLIST"


@pytest.fixture
def showdown_checklist_slate(engine):
    # build_lineup_role_checklist reads from slate_player_pool by
    # (slate_id, player_ids) rather than taking player dicts directly, so
    # this needs real rows inserted under a dedicated test slate_id, unlike
    # hard_role_exclusions/get_availability_gate above which take dicts.
    rows = WINSTON_ROWS + CLEAN_ROWS + DST_ROWS
    with engine.begin() as conn:
        for row in rows:
            conn.execute(
                text(
                    """
                    INSERT INTO slate_player_pool (slate_id, player_id, name, position, salary, team)
                    VALUES (:slate_id, :player_id, :name, :position, 5000, :team)
                    ON CONFLICT (slate_id, player_id) DO NOTHING
                    """
                ),
                {"slate_id": TEST_SHOWDOWN_SLATE_ID, **row},
            )
    try:
        yield TEST_SHOWDOWN_SLATE_ID
    finally:
        with engine.begin() as conn:
            conn.execute(
                text("DELETE FROM slate_player_pool WHERE slate_id = :slate_id"),
                {"slate_id": TEST_SHOWDOWN_SLATE_ID},
            )


def test_build_lineup_role_checklist_flags_a_showdown_qb_as_high_not_a_structural_rb_wr_te_check(
    engine, showdown_checklist_slate
):
    player_ids = [row["player_id"] for row in WINSTON_ROWS + CLEAN_ROWS]
    results = build_lineup_role_checklist(showdown_checklist_slate, player_ids, engine)
    by_id = {r["player_id"]: r for r in results}

    # Trevor Lawrence is the real QB on the slate here (as a CPT row) - must
    # get the unconditional QB/HIGH flag, not be misrouted into the RB/WR/TE
    # structural volume check the way the pre-fix code would have (checking
    # raw position "CPT" instead of his real position).
    lawrence = by_id["SD_CPT_LAWRENCE"]
    assert lawrence["severity"] == "HIGH"
    assert "QB" in lawrence["reason"]

    # Jameis Winston (both rows) is also a real QB and must get the same flag.
    for player_id in ("SD_CPT_WINSTON", "SD_FLEX_WINSTON"):
        winston = by_id[player_id]
        assert winston["severity"] == "HIGH"
        assert "QB" in winston["reason"]


def test_build_lineup_role_checklist_skips_a_showdown_dst_entirely(engine, showdown_checklist_slate):
    player_ids = [row["player_id"] for row in DST_ROWS]
    results = build_lineup_role_checklist(showdown_checklist_slate, player_ids, engine)

    for result in results:
        assert result["needs_check"] is False, f"a real Showdown defense must never be flagged, got {result}"
        assert result["severity"] is None
