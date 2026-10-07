import csv
import zipfile

import pytest
from sqlalchemy import text

from data.ownership_calibration import bulk_import_contest_standings

# Same real player identities test_ownership_calibration.py uses (already
# loaded for dk_thu_mon_2026_09_17 with a real stored proj_median) - keeps
# this a real match/no-match check, not a made-up-name stand-in.
GIBBS = {"name": "Jahmyr Gibbs", "player_id": "44137072", "salary": 8500}
SLATE_ID = "dk_thu_mon_2026_09_17"

# Every synthetic contest_id these tests import uses this range - cleaned up
# after every test (same pattern test_ownership_calibration.py's own
# `imported_contest` fixture uses) so these fake, small-sample rows never
# pollute the REAL, accumulating contest_ownership calibration dataset
# test_position_ownership_calibration_still_reflects_real_cross_week_data
# pools across every imported contest.
TEST_CONTEST_IDS = [f"50000{i}" for i in range(1, 8)]


@pytest.fixture(autouse=True)
def _cleanup_test_contests(engine):
    yield
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM contest_ownership WHERE contest_id = ANY(:ids)"), {"ids": TEST_CONTEST_IDS})


def _write_contest_csv(path, player_rows):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["Rank", "EntryId", "EntryName", "TimeRemaining", "Points", "Lineup", "", "Player", "Roster Position", "%Drafted", "FPTS"]
        )
        for i, (name, roster_pos, pct, fpts) in enumerate(player_rows, start=1):
            writer.writerow([i, 1000 + i, f"entrant{i}", 0, 100.0, "some lineup text", "", name, roster_pos, f"{pct}%", fpts])


def _write_zip_csv(zip_path, csv_member_name, player_rows):
    import io

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(
        ["Rank", "EntryId", "EntryName", "TimeRemaining", "Points", "Lineup", "", "Player", "Roster Position", "%Drafted", "FPTS"]
    )
    for i, (name, roster_pos, pct, fpts) in enumerate(player_rows, start=1):
        writer.writerow([i, 1000 + i, f"entrant{i}", 0, 100.0, "some lineup text", "", name, roster_pos, f"{pct}%", fpts])
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr(csv_member_name, buf.getvalue())


def test_bulk_import_processes_both_plain_csv_and_zipped_csv(tmp_path, engine):
    _write_contest_csv(tmp_path / "contest-standings-500001.csv", [(GIBBS["name"], "RB", 39.23, 37.6)])
    _write_zip_csv(tmp_path / "contest-standings-500002.zip", "contest-standings-500002.csv", [(GIBBS["name"], "RB", 20.0, 37.6)])

    result = bulk_import_contest_standings(str(tmp_path), slate_id=SLATE_ID, engine=engine)

    assert result["skipped"] == []
    imported_ids = {r["contest_id"] for r in result["imported"]}
    assert imported_ids == {"500001", "500002"}
    for r in result["imported"]:
        assert r["matched_with_projection"] == 1  # real Gibbs match, real stored proj_median


def test_bulk_import_skips_and_reports_unrecognized_files(tmp_path, engine):
    _write_contest_csv(tmp_path / "contest-standings-500003.csv", [(GIBBS["name"], "RB", 39.23, 37.6)])
    (tmp_path / "readme.txt").write_text("not a contest file")
    (tmp_path / "contest-standings-NOTANUMBER.csv").write_text("Rank,EntryId\n")

    with zipfile.ZipFile(tmp_path / "contest-standings-500004.zip", "w") as zf:
        zf.writestr("a.csv", "x")
        zf.writestr("b.csv", "y")  # two csv members - ambiguous, must be skipped

    result = bulk_import_contest_standings(str(tmp_path), slate_id=SLATE_ID, engine=engine)

    assert {r["contest_id"] for r in result["imported"]} == {"500003"}
    skipped_files = {s.get("file") for s in result["skipped"]}
    assert "readme.txt" in skipped_files
    assert "contest-standings-NOTANUMBER.csv" in skipped_files
    assert "contest-standings-500004.zip" in skipped_files


def test_bulk_import_dedupes_a_contest_id_seen_twice(tmp_path, engine):
    _write_contest_csv(tmp_path / "contest-standings-500005.csv", [(GIBBS["name"], "RB", 39.23, 37.6)])
    _write_contest_csv(tmp_path / "contest-standings-500005_1.csv", [(GIBBS["name"], "RB", 39.23, 37.6)])

    result = bulk_import_contest_standings(str(tmp_path), slate_id=SLATE_ID, engine=engine)

    assert len(result["imported"]) == 1
    assert result["imported"][0]["contest_id"] == "500005"
    dup_reasons = [s["reason"] for s in result["skipped"] if s.get("contest_id") == "500005"]
    assert len(dup_reasons) == 1
    assert "duplicate contest_id" in dup_reasons[0]


def test_bulk_import_skips_a_contest_with_no_resolvable_slate_id(tmp_path, engine):
    _write_contest_csv(tmp_path / "contest-standings-500006.csv", [(GIBBS["name"], "RB", 39.23, 37.6)])

    result = bulk_import_contest_standings(str(tmp_path), slate_id=None, engine=engine)

    assert result["imported"] == []
    assert result["skipped"][0]["contest_id"] == "500006"
    assert "no slate_id" in result["skipped"][0]["reason"]


def test_bulk_import_slate_id_by_contest_overrides_the_flat_default(tmp_path, engine):
    _write_contest_csv(tmp_path / "contest-standings-500007.csv", [(GIBBS["name"], "RB", 39.23, 37.6)])

    result = bulk_import_contest_standings(
        str(tmp_path),
        slate_id="fallback_slate_that_does_not_exist",
        slate_id_by_contest={"500007": SLATE_ID},
        engine=engine,
    )

    assert result["imported"][0]["slate_id"] == SLATE_ID
    assert result["imported"][0]["matched_with_projection"] == 1
