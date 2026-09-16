from types import SimpleNamespace

from data.nflverse_fetch import _skill_player_fantasy_points

# Real DK Classic scoring rules (pasted directly by the user, verified line
# by line against data/nflverse_fetch.py's _dst_fantasy_points/
# _points_allowed_bonus for DST, and against these real stat lines for
# skill players - see _skill_player_fantasy_points' module-level comment for
# the full story of why nflverse's own fantasy_points/fantasy_points_ppr
# columns were confirmed NOT to match DK's real formula).

_STAT_FIELDS = [
    "passing_yards", "passing_tds", "passing_interceptions",
    "rushing_yards", "rushing_tds",
    "receiving_yards", "receiving_tds", "receptions",
    "special_teams_tds", "fumble_recovery_tds", "fumbles_lost_total",
    "passing_2pt_conversions", "rushing_2pt_conversions", "receiving_2pt_conversions",
]


def _row(**overrides):
    base = {field: 0 for field in _STAT_FIELDS}
    base.update(overrides)
    return SimpleNamespace(**base)


def test_puka_nacua_real_225_yard_2_td_game():
    # Real 2025 week 16 stat line: 12 rec, 225 receiving yards, 2 TD, no
    # fumbles/2pt. nflverse's own fantasy_points_ppr stores 46.5 (a generic
    # full-PPR formula, confirmed by hand); real DK Classic scoring adds the
    # 100+ receiving yard bonus this generic formula doesn't have, for 49.5.
    row = _row(receiving_yards=225, receiving_tds=2, receptions=12)
    assert _skill_player_fantasy_points(row) == 49.5


def test_joe_flacco_real_470_yard_4_td_2_int_game():
    # Real 2025 week 9 stat line: 470 pass yards, 4 pass TD, 2 INT, 1 fumble
    # lost (on a sack), 1 passing 2pt conversion, -1 rushing yard. nflverse's
    # own fantasy_points_ppr stores 30.7 (matches -2/INT, -2/fumble, no 300-
    # yard bonus); real DK Classic (-1/INT, -1/fumble, +3 for 300+ passing)
    # is 36.7.
    row = _row(
        passing_yards=470, passing_tds=4, passing_interceptions=2,
        rushing_yards=-1, fumbles_lost_total=1, passing_2pt_conversions=1,
    )
    assert _skill_player_fantasy_points(row) == 36.7


def test_offensive_fumble_recovery_td_is_scored():
    # Real 2024 week 4 case: Patrick Ricard recovered his own team's fumble
    # for a TD with zero other stats. nflverse's own fantasy_points/
    # fantasy_points_ppr scores this 0.0 - that scoring category is entirely
    # missing from nflverse's formula, not just discounted. Real DK Classic
    # scores "Offensive Fumble Recovery TD" at +6.
    row = _row(fumble_recovery_tds=1)
    assert _skill_player_fantasy_points(row) == 6.0


def test_special_teams_return_td_is_scored():
    row = _row(special_teams_tds=1)
    assert _skill_player_fantasy_points(row) == 6.0


def test_hundred_yard_rushing_bonus():
    row = _row(rushing_yards=100, rushing_tds=0)
    # 100 * 0.1 = 10.0, + the 100+ yard bonus (+3) = 13.0
    assert _skill_player_fantasy_points(row) == 13.0


def test_no_bonus_at_ninety_nine_rushing_yards():
    row = _row(rushing_yards=99)
    assert _skill_player_fantasy_points(row) == 9.9  # no bonus - real DK threshold is 100+, not 99


def test_fumble_lost_on_a_return_play_is_still_penalized():
    # Real, confirmed pattern across 113 real 2023-2025 player-weeks:
    # fumbles_lost_total=1 with sack_fumbles_lost/rushing_fumbles_lost/
    # receiving_fumbles_lost all 0 (a punt/kickoff return fumble) - nflverse's
    # own fantasy_points_ppr does not penalize these at all. DK's real rule
    # ("Fumble Lost -1 Pt") has no play-type qualifier, so fumbles_lost_total
    # (not the sum of the three sub-categories) is used deliberately.
    row = _row(fumbles_lost_total=1)
    assert _skill_player_fantasy_points(row) == -1.0


def test_two_point_conversion_scores_regardless_of_type():
    for field in ("passing_2pt_conversions", "rushing_2pt_conversions", "receiving_2pt_conversions"):
        row = _row(**{field: 1})
        assert _skill_player_fantasy_points(row) == 2.0, f"{field} should score +2"


def test_interception_thrown_is_penalized_one_point_not_two():
    row = _row(passing_interceptions=1)
    assert _skill_player_fantasy_points(row) == -1.0
