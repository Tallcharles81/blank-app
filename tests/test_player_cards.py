from sqlalchemy import text

from models.player_cards import format_cards, player_cards

SLATE = "dk_sunday_2026_10_11"


def _ids(engine, names):
    with engine.connect() as conn:
        return [r[0] for r in conn.execute(text(
            "SELECT player_id FROM slate_player_pool WHERE slate_id = :s AND name = ANY(:n)"), {"s": SLATE, "n": names})]


def test_cards_show_averages_usage_and_matchup(engine):
    cards = player_cards(SLATE, _ids(engine, ["Jahmyr Gibbs", "Raiders"]), engine=engine).set_index("player")
    gibbs, raiders = cards.loc["Jahmyr Gibbs"], cards.loc["Raiders"]
    # Only games before the slate's week count (week 5 -> weeks 1-4).
    assert 1 <= gibbs.games <= 4
    assert gibbs.season_avg > 0 and gibbs.last3_avg > 0
    assert gibbs.car_pg > 0 and gibbs.tgt_pg >= 0
    assert 1 <= gibbs.opp_rank_vs_pos <= 32
    assert gibbs.proj_median < gibbs.proj_ceiling
    # A DST's matchup is the offense it faces, with both implied totals shown.
    assert raiders.pos == "DST" and raiders.opp == "NE"
    assert raiders.opp_rank_vs_pos is not None
    table = format_cards(cards.reset_index())
    assert "Jahmyr Gibbs" in table and "Raiders" in table
