import json

import pytest
from sqlalchemy import text

from models.absence_redistribution import RB_SHARES, redistribute_absence

SLATE = "test_absence_redistribution"
SOURCE = "dk_thu_mon_2026_10_01"
NAMES = ("Travis Etienne Jr.", "Alvin Kamara", "Kendre Miller", "Chris Olave", "Devaughn Vele")


@pytest.fixture
def slate(engine):
    # A throwaway copy of real New Orleans players (real ids resolve through the crosswalk).
    with engine.begin() as conn:
        if conn.execute(text("SELECT count(*) FROM slate_player_pool WHERE slate_id = :s AND name = ANY(:n)"),
                        {"s": SOURCE, "n": list(NAMES)}).scalar() < len(NAMES):
            pytest.skip(f"{SOURCE} not loaded")
        conn.execute(text("""INSERT INTO slate_player_pool (slate_id, player_id, name, position, salary, team, opponent, game_time)
            SELECT :t, player_id, name, position, salary, team, opponent, game_time FROM slate_player_pool WHERE slate_id = :s AND name = ANY(:n)"""),
            {"t": SLATE, "s": SOURCE, "n": list(NAMES)})
        conn.execute(text("""INSERT INTO projections (slate_id, player_id, proj_floor, proj_median, proj_ceiling, proj_percentiles)
            SELECT :t, p.player_id, p.proj_floor, p.proj_median, p.proj_ceiling, p.proj_percentiles FROM projections p
            JOIN slate_player_pool sp USING (slate_id, player_id) WHERE p.slate_id = :s AND sp.name = ANY(:n)"""),
            {"t": SLATE, "s": SOURCE, "n": list(NAMES)})
    try:
        yield SLATE
    finally:
        with engine.begin() as conn:
            conn.execute(text("DELETE FROM projections WHERE slate_id = :t"), {"t": SLATE})
            conn.execute(text("DELETE FROM slate_player_pool WHERE slate_id = :t"), {"t": SLATE})


def _medians(engine):
    with engine.connect() as conn:
        return {r.name: float(r.proj_median) for r in conn.execute(text(
            "SELECT sp.name, p.proj_median FROM projections p JOIN slate_player_pool sp USING (slate_id, player_id) WHERE p.slate_id = :t"),
            {"t": SLATE})}


def test_lead_back_out_moves_measured_shares_to_his_backups(slate, engine):
    before = _medians(engine)
    changes = redistribute_absence(SLATE, "Travis Etienne Jr.", engine=engine)
    after = _medians(engine)
    assert [c[0] for c in changes] == ["Alvin Kamara", "Kendre Miller"]
    out = before["Travis Etienne Jr."]
    assert after["Alvin Kamara"] == pytest.approx(before["Alvin Kamara"] + RB_SHARES[2] * out, abs=0.05)
    assert after["Kendre Miller"] == pytest.approx(before["Kendre Miller"] + RB_SHARES[3] * out, abs=0.05)
    assert after["Chris Olave"] == before["Chris Olave"]


def test_receiver_out_moves_nothing(slate, engine):
    before = _medians(engine)
    assert redistribute_absence(SLATE, "Chris Olave", engine=engine) == []
    assert _medians(engine) == before


def test_unknown_player_is_an_error(slate, engine):
    with pytest.raises(ValueError):
        redistribute_absence(SLATE, "Nobody Here", engine=engine)
