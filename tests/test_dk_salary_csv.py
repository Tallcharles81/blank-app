import csv
import os

from sqlalchemy import text

from data.dk_salary_csv import (
    load_slate_player_pool,
    parse_dk_salary_csv,
    write_dk_upload_csv,
)

# Real DK Showdown export (DET@BUF 09/17/2026) - the first real Showdown CSV
# this codebase has ever parsed through this exact ingestion path (every
# prior Showdown fix this session - the crosswalk, hard_role_exclusions,
# get_availability_gate, the correlation model - was built and tested only
# against hand-built dicts, never a real export). Parsing it for real
# revealed that DK's actual "Position" column is the true football position
# on EVERY slate type, including Showdown - it's "Roster Position" that
# carries the CPT/FLEX slot label there. See parse_dk_salary_csv's own
# comment for the real fix this exposed.
REAL_SHOWDOWN_CSV = os.path.join(os.path.dirname(__file__), "fixtures", "dk_showdown_real.csv")


def test_parse_real_showdown_csv_detects_showdown_slate_type():
    slate_type, _players = parse_dk_salary_csv(REAL_SHOWDOWN_CSV)
    assert slate_type == "showdown"


def test_parse_real_showdown_csv_stores_roster_slot_as_position():
    _slate_type, players = parse_dk_salary_csv(REAL_SHOWDOWN_CSV)
    by_id = {p["player_id"]: p for p in players}

    # Real captain/flex pair for the same real player, two different real DK
    # ids, 1.5x salary baked into the CPT row already (18000 vs 12000) - both
    # rows' real football position is "RB" in DK's own export, which must
    # end up as the roster slot label ("CPT"/"FLEX") here, not "RB" - the
    # downstream solver (_solve_showdown) can only tell rows apart by slot.
    cpt_row = by_id["44138478"]
    flex_row = by_id["44138432"]
    assert cpt_row["name"] == "Jahmyr Gibbs" == flex_row["name"]
    assert cpt_row["position"] == "CPT"
    assert flex_row["position"] == "FLEX"
    assert cpt_row["salary"] == 18000
    assert flex_row["salary"] == 12000

    # A real Showdown DST row must resolve the same way as any other
    # position - "DST" in DK's real Position column, still CPT/FLEX here.
    dst_cpt = by_id["44138496"]
    assert dst_cpt["name"].strip() == "Bills"
    assert dst_cpt["position"] == "CPT"


def test_parse_real_showdown_csv_every_row_is_cpt_or_flex():
    _slate_type, players = parse_dk_salary_csv(REAL_SHOWDOWN_CSV)
    positions = {p["position"] for p in players}
    assert positions == {"CPT", "FLEX"}


TEST_UPLOAD_SLATE_ID = "TEST_DK_UPLOAD_CSV_SHOWDOWN"


def test_write_dk_upload_csv_writes_one_row_per_lineup_under_a_shared_header(engine, tmp_path):
    # DK's own real bulk-upload template: one shared header row, then one
    # data row per lineup, all uploaded together in a single file - this is
    # what makes uploading several real lineups at once actually work.
    load_slate_player_pool(TEST_UPLOAD_SLATE_ID, REAL_SHOWDOWN_CSV, engine)
    try:
        lineup_a = ["44138478", "44138433", "44138480", "44138483", "44138485", "44138492"]
        lineup_b = ["44138479", "44138432", "44138434", "44138437", "44138439", "44138446"]
        output_path = tmp_path / "upload.csv"

        write_dk_upload_csv(TEST_UPLOAD_SLATE_ID, [lineup_a, lineup_b], str(output_path), engine)

        with open(output_path, newline="", encoding="utf-8") as f:
            rows = list(csv.reader(f))
        assert rows[0] == ["CPT", "FLEX", "FLEX", "FLEX", "FLEX", "FLEX"]
        assert len(rows) == 3  # header + 2 lineups
        assert rows[1][0] == "Jahmyr Gibbs (44138478)"
        assert rows[2][0] == "Josh Allen (44138479)"
    finally:
        with engine.begin() as conn:
            conn.execute(text("DELETE FROM slate_player_pool WHERE slate_id = :slate_id"), {"slate_id": TEST_UPLOAD_SLATE_ID})


def test_write_dk_upload_csv_still_accepts_a_single_flat_lineup(engine, tmp_path):
    load_slate_player_pool(TEST_UPLOAD_SLATE_ID, REAL_SHOWDOWN_CSV, engine)
    try:
        lineup = ["44138478", "44138433", "44138480", "44138483", "44138485", "44138492"]
        output_path = tmp_path / "upload_single.csv"

        write_dk_upload_csv(TEST_UPLOAD_SLATE_ID, lineup, str(output_path), engine)

        with open(output_path, newline="", encoding="utf-8") as f:
            rows = list(csv.reader(f))
        assert len(rows) == 2  # header + 1 lineup
    finally:
        with engine.begin() as conn:
            conn.execute(text("DELETE FROM slate_player_pool WHERE slate_id = :slate_id"), {"slate_id": TEST_UPLOAD_SLATE_ID})
