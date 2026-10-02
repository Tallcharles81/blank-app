import json

import pytest
from sqlalchemy import text

from data.sabersim_import import apply_sabersim, projection_accuracy, sabersim_ladder

SLATE = "TEST_SABERSIM_SD"
HEADER = "DFS ID,Name,Pos,Team,Opp,Status,Salary,Actual,SS Proj,Adj Own,dk_25_percentile,dk_50_percentile,dk_75_percentile,dk_85_percentile,dk_95_percentile,dk_99_percentile\n"


def test_ladder_interpolates_p10_and_p90_from_sabersims_percentiles():
    lad = sabersim_ladder({"dk_25_percentile": 11.4, "dk_50_percentile": 16.6375, "dk_75_percentile": 23,
                           "dk_85_percentile": 26.9, "dk_95_percentile": 34.005})
    assert lad["25"] == 11.4 and lad["50"] == pytest.approx(16.64, abs=0.01) and lad["75"] == 23
    assert lad["10"] == pytest.approx(16.6375 - 1.9 * (16.6375 - 11.4), abs=0.01)
    assert lad["90"] == pytest.approx(26.9 + 0.4 * (34.005 - 26.9), abs=0.01)


def test_ladder_is_none_for_inactive_rows():
    assert sabersim_ladder({"dk_25_percentile": 0, "dk_50_percentile": 0, "dk_75_percentile": 0,
                            "dk_85_percentile": 0, "dk_95_percentile": 0}) is None


@pytest.fixture
def showdown_slate(engine, tmp_path):
    # Warren has SaberSim numbers; the kicker doesn't (SaberSim exports are Classic).
    players = [("C_W", "Jaylen Warren", "CPT"), ("F_W", "Jaylen Warren", "FLEX"), ("F_K", "Chris Boswell", "FLEX")]
    ours = {"10": 5, "25": 8, "50": 12, "75": 16, "90": 22}
    with engine.begin() as conn:
        for pid, name, pos in players:
            conn.execute(text("INSERT INTO slate_player_pool (slate_id, player_id, name, position, salary, team, opponent) "
                              "VALUES (:s, :p, :n, :pos, 5000, 'PIT', 'CLE')"), {"s": SLATE, "p": pid, "n": name, "pos": pos})
            lad = {k: v * (1.5 if pos == "CPT" else 1) for k, v in ours.items()}
            conn.execute(text("INSERT INTO projections (slate_id, player_id, proj_floor, proj_median, proj_ceiling, proj_percentiles) "
                              "VALUES (:s, :p, :f, :m, :c, :pp)"),
                         {"s": SLATE, "p": pid, "f": lad["25"], "m": lad["50"], "c": lad["90"], "pp": json.dumps(lad)})
    csv_path = tmp_path / "ss.csv"
    csv_path.write_text(HEADER + "999,Jaylen Warren,RB,PIT,CLE,,6200,,17.87,76.9,11.4,16.6375,23,26.9,34.005,42.8\n")
    yield str(csv_path)
    with engine.begin() as conn:
        for t in ("projections", "slate_player_pool", "projection_sources", "contest_ownership"):
            conn.execute(text(f"DELETE FROM {t} WHERE slate_id = :s"), {"s": SLATE})


def _proj(engine, pid):
    with engine.connect() as conn:
        return float(conn.execute(text("SELECT proj_median FROM projections WHERE slate_id = :s AND player_id = :p"),
                                  {"s": SLATE, "p": pid}).scalar())


def test_apply_writes_sabersim_with_captain_multiplier_and_keeps_ours_for_the_rest(engine, showdown_slate):
    report = apply_sabersim(SLATE, showdown_slate, engine=engine)
    assert report["matched_rows"] == 2 and report["kept_model_players"] == ["Chris Boswell"]
    assert _proj(engine, "F_W") == pytest.approx(16.64, abs=0.01)
    assert _proj(engine, "C_W") == pytest.approx(16.64 * 1.5, abs=0.02)
    assert _proj(engine, "F_K") == 12  # no SaberSim row - ours stays
    with engine.connect() as conn:
        model = conn.execute(text("SELECT proj_median FROM projection_sources WHERE slate_id = :s AND source = 'model' "
                                  "AND player_id = 'F_W'"), {"s": SLATE}).scalar()
    assert float(model) == 12  # our original projection was snapshotted before being replaced


def test_reapplying_never_records_sabersim_as_the_model(engine, showdown_slate):
    apply_sabersim(SLATE, showdown_slate, engine=engine)
    apply_sabersim(SLATE, showdown_slate, engine=engine)
    with engine.connect() as conn:
        model = conn.execute(text("SELECT proj_median FROM projection_sources WHERE slate_id = :s AND source = 'model' "
                                  "AND player_id = 'F_W'"), {"s": SLATE}).scalar()
    assert float(model) == 12


def test_accuracy_compares_sources_on_the_same_players(engine, showdown_slate):
    apply_sabersim(SLATE, showdown_slate, engine=engine, write_projections=False)
    with engine.begin() as conn:
        conn.execute(text("INSERT INTO contest_ownership (contest_id, slate_id, player_id, name, position, pct_drafted, fpts_contest) "
                          "VALUES ('TESTC_SS', :s, 'F_W', 'Jaylen Warren', 'RB', 70, 20.0)"), {"s": SLATE})
    acc = projection_accuracy(SLATE, engine=engine)
    assert acc["players"] == 1
    assert acc["by_source"]["model"]["mae"] == pytest.approx(8.0)
    assert acc["by_source"]["sabersim"]["mae"] == pytest.approx(20.0 - 16.64, abs=0.01)
