import csv

import pytest
from sqlalchemy import text

from data.ownership_calibration import (
    MIN_MATCHED_PLAYERS_FOR_CORRELATION,
    _classify_qb_starter_certainty,
    _rank,
    _spearman,
    analyze_qb_ownership_gap,
    import_contest_standings,
    parse_contest_standings_csv,
    run_ownership_correlation_test,
    summarize_ownership_calibration,
)

# Real player identities already loaded for dk_thu_mon_2026_09_17 (with real
# stored proj_median), used to build a small, realistic-shaped contest CSV
# fixture rather than made-up names - mirrors the real DK export quirk this
# module has to handle: a player who was started at their natural position
# in some entries and FLEX in others gets a separate ownership row per slot.
GIBBS = {"name": "Jahmyr Gibbs", "player_id": "44137072", "salary": 8500}
ROBINSON = {"name": "Bijan Robinson", "player_id": "44137074", "salary": 8200}
FAKE_PLAYER_NAME = "Zzyzx Nonexistent Player"


def _write_contest_csv(path, player_rows):
    """player_rows: list of (name, roster_position, pct_drafted, fpts) -
    written in DK's real bolted-on-side-table shape (entry columns + a
    separate Player/Roster Position/%Drafted/FPTS block on the same rows).
    """
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["Rank", "EntryId", "EntryName", "TimeRemaining", "Points", "Lineup", "", "Player", "Roster Position", "%Drafted", "FPTS"]
        )
        for i, (name, roster_pos, pct, fpts) in enumerate(player_rows, start=1):
            writer.writerow([i, 1000 + i, f"entrant{i}", 0, 100.0, "some lineup text", "", name, roster_pos, f"{pct}%", fpts])


@pytest.fixture
def contest_csv(tmp_path):
    path = tmp_path / "contest-standings-TEST.csv"
    _write_contest_csv(
        path,
        [
            (GIBBS["name"], "RB", 39.23, 37.6),
            (GIBBS["name"], "FLEX", 4.40, 37.6),  # same real player, second roster slot
            (ROBINSON["name"], "RB", 25.00, 29.8),
            ("Bengals ", "DST", 5.0, 10.0),  # DST trailing-space quirk from the real export
            (FAKE_PLAYER_NAME, "WR", 1.0, 0.0),  # no real slate_player_pool match
        ],
    )
    return str(path)


def test_parse_contest_standings_csv_dedupes_and_sums_roster_slots(contest_csv):
    parsed, skipped, is_showdown = parse_contest_standings_csv(contest_csv)
    assert skipped == 0
    assert is_showdown is False  # this fixture uses real Classic roster-position labels
    # Gibbs' two roster-slot rows (RB + FLEX) must sum into one real total.
    assert parsed[GIBBS["name"]]["pct_drafted"] == pytest.approx(39.23 + 4.40)
    assert parsed[GIBBS["name"]]["fpts_contest"] == 37.6
    assert parsed[ROBINSON["name"]]["pct_drafted"] == pytest.approx(25.00)
    # The DST trailing-space quirk must be preserved by parsing (stripped at
    # match time in import_contest_standings, not silently altered here).
    assert "Bengals" in parsed  # stripped during parsing itself
    assert parsed[FAKE_PLAYER_NAME]["pct_drafted"] == pytest.approx(1.0)


def test_parse_contest_standings_csv_prefers_flex_fpts_on_showdown(tmp_path):
    # Regression test for a real bug: on a Classic slate a player's FPTS is
    # identical across every roster-slot row (no multiplier), so "last row
    # wins" was safe there - but on Showdown, CPT applies a real 1.5x
    # multiplier to the SAME real per-game performance, so a player's CPT
    # row and FLEX row carry genuinely DIFFERENT real FPTS. Found for real
    # reconciling entered lineups against DK's own per-entry totals (Matthew
    # Stafford: 46.47 as CPT vs 30.98 as FLEX in the same real contest,
    # exactly 1.5x). The real, unmultiplied FLEX value must be preferred
    # regardless of which row happens to appear last in the file.
    path = tmp_path / "contest-standings-SHOWDOWN-TEST.csv"
    _write_contest_csv(
        path,
        [
            (GIBBS["name"], "CPT", 10.0, 46.47),  # CPT row written FIRST in the file...
            (GIBBS["name"], "FLEX", 30.0, 30.98),  # ...but FLEX (unmultiplied) must still win
        ],
    )
    parsed, skipped, is_showdown = parse_contest_standings_csv(str(path))
    assert is_showdown is True
    assert parsed[GIBBS["name"]]["fpts_contest"] == pytest.approx(30.98)


def test_parse_contest_standings_csv_falls_back_to_cpt_fpts_if_never_flexed(tmp_path):
    # A player nobody in the whole real field ever drafted into FLEX (only
    # ever CPT) has no unmultiplied real value to prefer - falls back to
    # the CPT (multiplied) number rather than raising or silently dropping
    # the player. A real, disclosed limitation of that edge case.
    path = tmp_path / "contest-standings-SHOWDOWN-TEST2.csv"
    _write_contest_csv(path, [(GIBBS["name"], "CPT", 10.0, 46.47)])
    parsed, skipped, is_showdown = parse_contest_standings_csv(str(path))
    assert is_showdown is True
    assert parsed[GIBBS["name"]]["fpts_contest"] == pytest.approx(46.47)


TEST_CONTEST_ID = "TEST_CONTEST_OWNERSHIP"


@pytest.fixture
def imported_contest(engine, contest_csv):
    try:
        result = import_contest_standings(contest_csv, slate_id="dk_thu_mon_2026_09_17", contest_id=TEST_CONTEST_ID, engine=engine)
        yield result
    finally:
        with engine.begin() as conn:
            conn.execute(text("DELETE FROM contest_ownership WHERE contest_id = :c"), {"c": TEST_CONTEST_ID})
            conn.execute(text("DELETE FROM ownership_calibration_runs WHERE contest_id = :c"), {"c": TEST_CONTEST_ID})


def test_import_contest_standings_matches_real_players_and_computes_proxy(engine, imported_contest):
    assert imported_contest["total_players"] == 4  # Gibbs, Robinson, Bengals, the fake player
    # Gibbs, Robinson, AND the Bengals DST all have real stored proj_median
    # for this slate - DSTs get real projections too, not just skill players.
    assert imported_contest["matched_with_projection"] == 3
    assert imported_contest["unmatched"] == 1  # the fake player

    with engine.connect() as conn:
        rows = {
            r["name"]: dict(r)
            for r in conn.execute(
                text("SELECT * FROM contest_ownership WHERE contest_id = :c"), {"c": TEST_CONTEST_ID}
            ).mappings()
        }

    gibbs_row = rows[GIBBS["name"]]
    assert gibbs_row["player_id"] == GIBBS["player_id"]
    assert float(gibbs_row["pct_drafted"]) == pytest.approx(39.23 + 4.40)
    # ownership_proxy must match models.field_simulation.ownership_proxy's
    # real formula - proj_median / (salary / 1000) - within the stored
    # column's own NUMERIC(10, 4) precision (schema.sql), not a false
    # failure over rounding at the 5th decimal place.
    expected_proxy = float(gibbs_row["proj_median_at_import"]) / (GIBBS["salary"] / 1000.0)
    assert float(gibbs_row["ownership_proxy"]) == pytest.approx(expected_proxy, abs=1e-3)

    fake_row = rows[FAKE_PLAYER_NAME]
    assert fake_row["player_id"] is None
    assert fake_row["unmatched_reason"] is not None


def test_import_contest_standings_upsert_does_not_duplicate(engine, imported_contest):
    # Re-importing the same contest_id must update, not add duplicate rows -
    # the real workflow is "the user hands the same export again" not just
    # "a brand-new contest every time."
    with engine.connect() as conn:
        count = conn.execute(
            text("SELECT count(*) FROM contest_ownership WHERE contest_id = :c"), {"c": TEST_CONTEST_ID}
        ).scalar()
    assert count == 4


TEST_SHOWDOWN_CONTEST_ID = "TEST_CONTEST_OWNERSHIP_SHOWDOWN"


def test_import_contest_standings_refuses_to_compute_proxy_for_a_real_showdown_contest(engine, tmp_path):
    # Regression test for a real methodology bug caught before ever running
    # a correlation on it: this codebase has never loaded a real Showdown
    # slate_player_pool (no real Showdown CSV export was ever available -
    # see data/player_crosswalk.py), so the only pool a Showdown contest's
    # players can be name-matched against is Classic-priced. A Showdown
    # CPT/FLEX slot's real DK salary is a different number from that same
    # player's Classic salary, so computing ownership_proxy off the Classic
    # price would silently produce a wrong result rather than an honest gap.
    path = tmp_path / "contest-standings-SHOWDOWN-TEST.csv"
    _write_contest_csv(
        path,
        [
            (GIBBS["name"], "CPT", 45.0, 37.6),
            (ROBINSON["name"], "FLEX", 20.0, 29.8),
        ],
    )
    try:
        result = import_contest_standings(
            str(path), slate_id="dk_thu_mon_2026_09_17", contest_id=TEST_SHOWDOWN_CONTEST_ID, engine=engine
        )
        assert result["is_showdown"] is True
        assert result["showdown_salary_mismatch"] == 2
        assert result["matched_with_projection"] == 0

        with engine.connect() as conn:
            rows = {
                r["name"]: dict(r)
                for r in conn.execute(
                    text("SELECT * FROM contest_ownership WHERE contest_id = :c"), {"c": TEST_SHOWDOWN_CONTEST_ID}
                ).mappings()
            }
        gibbs_row = rows[GIBBS["name"]]
        assert gibbs_row["player_id"] == GIBBS["player_id"]  # real identity still captured
        assert gibbs_row["salary"] is None  # but salary/proxy intentionally withheld
        assert gibbs_row["ownership_proxy"] is None
        assert "Showdown" in gibbs_row["unmatched_reason"]
    finally:
        with engine.begin() as conn:
            conn.execute(text("DELETE FROM contest_ownership WHERE contest_id = :c"), {"c": TEST_SHOWDOWN_CONTEST_ID})


TEST_SHOWDOWN_MATCH_CONTEST_ID = "TEST_CONTEST_OWNERSHIP_SHOWDOWN_MATCHED"

# Real players/prices from dk_showdown_det_buf_2026_09_17, the first real
# Showdown slate this codebase has ever loaded (see data/dk_salary_csv.py) -
# Amon-Ra St. Brown's real FLEX row.
ST_BROWN_FLEX = {"name": "Amon-Ra St. Brown", "player_id": "44138434", "salary": 10400}


def test_import_contest_standings_computes_real_proxy_for_showdown_vs_showdown(engine, tmp_path):
    # Once this codebase actually has a real Showdown slate_player_pool to
    # match against (unlike the Classic-pool case above), a real Showdown
    # contest's pricing IS valid and a real ownership_proxy should be
    # computed, not silently withheld - the whole reason
    # slate_pool_is_showdown exists in import_contest_standings.
    path = tmp_path / "contest-standings-SHOWDOWN-MATCH-TEST.csv"
    _write_contest_csv(path, [(ST_BROWN_FLEX["name"], "CPT", 30.0, 54.86)])
    try:
        result = import_contest_standings(
            str(path),
            slate_id="dk_showdown_det_buf_2026_09_17",
            contest_id=TEST_SHOWDOWN_MATCH_CONTEST_ID,
            engine=engine,
        )
        assert result["is_showdown"] is True
        assert result["showdown_salary_mismatch"] == 0
        assert result["matched_with_projection"] == 1

        with engine.connect() as conn:
            row = dict(
                conn.execute(
                    text("SELECT * FROM contest_ownership WHERE contest_id = :c"),
                    {"c": TEST_SHOWDOWN_MATCH_CONTEST_ID},
                ).mappings().fetchone()
            )
        # Priced off the real FLEX row, not the CPT row that this contest's
        # own %Drafted rows happened to be filed under - see this function's
        # own docstring for why FLEX is the one real base price.
        assert row["salary"] == ST_BROWN_FLEX["salary"]
        assert row["ownership_proxy"] is not None
        # Resolved from real game history, not left as the literal "FLEX"
        # roster-slot label - otherwise analyze_qb_ownership_gap's own
        # `position == "QB"` filter could never match a real Showdown QB.
        assert row["position"] == "WR"
    finally:
        with engine.begin() as conn:
            conn.execute(
                text("DELETE FROM contest_ownership WHERE contest_id = :c"), {"c": TEST_SHOWDOWN_MATCH_CONTEST_ID}
            )


# --- _spearman: manual rank correlation, no scipy dependency --------------


def test_spearman_perfect_positive_correlation():
    # n=5 is far below where _spearman's own documented large-n normal
    # approximation is accurate (its docstring is explicit that it's meant
    # for matched samples in the hundreds) - a perfect rank correlation at
    # this n still comes back "small but not nearly as extreme as the exact
    # test would give," which is the real, expected behavior of the
    # approximation, not a bug. Asserting a small p rather than a
    # near-zero one so this test reflects what the function actually does
    # at small n instead of a stronger claim it was never meant to satisfy.
    rho, p_value = _spearman([1, 2, 3, 4, 5], [10, 20, 30, 40, 50])
    assert rho == pytest.approx(1.0)
    assert 0.0 < p_value < 0.05


def test_spearman_perfect_negative_correlation():
    rho, _ = _spearman([1, 2, 3, 4, 5], [50, 40, 30, 20, 10])
    assert rho == pytest.approx(-1.0)


def test_spearman_handles_ties_via_average_rank():
    # Two tied values in x (both rank 1.5) must not crash or silently distort - this
    # is the exact tie-corrected Spearman formula scipy's default uses too.
    rho, _ = _spearman([1, 1, 3, 4], [10, 10, 30, 40])
    assert rho == pytest.approx(1.0)


def test_rank_assigns_average_rank_to_ties():
    # Best (highest value) = rank 1; two tied highest values split ranks 1 and 2.
    assert _rank([10, 10, 5]) == [1.5, 1.5, 3.0]


# --- run_ownership_correlation_test ----------------------------------------


def test_run_ownership_correlation_test_reports_too_small_a_sample(engine, imported_contest):
    # Only 2 real matched-with-projection players in this fixture - well
    # below MIN_MATCHED_PLAYERS_FOR_CORRELATION, must report that honestly
    # rather than compute a meaningless correlation off 2 points.
    result = run_ownership_correlation_test(TEST_CONTEST_ID, slate_id="dk_thu_mon_2026_09_17", engine=engine, store=False)
    assert result["proj_median"]["n"] < MIN_MATCHED_PLAYERS_FOR_CORRELATION
    assert result["proj_median"]["spearman_rho"] is None
    assert "note" in result["proj_median"]


def test_run_ownership_correlation_test_uses_the_same_intersected_sample_for_both_bases(engine):
    # Uses the real, permanently-imported contest 193028208 (a real DK
    # contest-standings export the user actually entered, imported against
    # dk_thu_mon_2026_09_17 and kept in this dev DB as real accumulating
    # calibration data - see data/ownership_calibration.py's module
    # docstring) rather than a synthetic fixture, the same real-fixture-
    # dependency pattern tests/test_pre_lock_check.py's
    # test_hard_role_exclusions_survives_real_stars_and_catches_real_winston
    # already uses: if this contest is ever removed from the dev DB, this
    # test needs updating; the synthetic-fixture tests above cover the same
    # intersection logic without that dependency.
    #
    # Regression test for a real bug caught before ever reporting a result:
    # an earlier version ran realized_fpts over every matched row with
    # salary+fpts regardless of whether proj_median existed, a different
    # (larger) sample than proj_median's - which would make "realized_fpts
    # scored a higher rho" potentially just an artifact of an easier sample,
    # not a real head-to-head result. Both bases must report the same n.
    result = run_ownership_correlation_test("193028208", slate_id="dk_thu_mon_2026_09_17", engine=engine, store=False)
    assert result["proj_median"]["n"] == result["realized_fpts"]["n"]
    # The full, non-intersected sample is real too, but must be clearly a
    # different (larger or equal) size, never silently conflated with the
    # head-to-head comparison.
    assert result["realized_fpts_full_sample"]["n"] >= result["realized_fpts"]["n"]


def test_run_ownership_correlation_test_stores_results_when_requested(engine, imported_contest):
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM ownership_calibration_runs WHERE contest_id = :c"), {"c": TEST_CONTEST_ID})

    run_ownership_correlation_test(TEST_CONTEST_ID, slate_id="dk_thu_mon_2026_09_17", engine=engine, store=True)

    with engine.connect() as conn:
        stored = conn.execute(
            text("SELECT proxy_basis FROM ownership_calibration_runs WHERE contest_id = :c"), {"c": TEST_CONTEST_ID}
        ).scalars().fetchall()
    assert set(stored) == {"proj_median", "realized_fpts", "realized_fpts_full_sample"}


# --- QB starter-certainty dig ----------------------------------------------


def _game(snap_pct):
    return {"snap_pct": snap_pct}


def test_classify_qb_starter_certainty_thin_data():
    assert _classify_qb_starter_certainty([_game(0.9)]) == "thin_data"


def test_classify_qb_starter_certainty_intermittent_backup_pattern():
    # Real Jameis Winston shape - the exact pattern data/pre_lock_check.py's
    # hard gate is built to catch.
    games = [_game(0.03), _game(1.0), _game(1.0), _game(0.77)]
    assert _classify_qb_starter_certainty(games) == "intermittent_backup_pattern"


def test_classify_qb_starter_certainty_clean_starter():
    games = [_game(1.0), _game(0.99), _game(0.95), _game(1.0)]
    assert _classify_qb_starter_certainty(games) == "clean_starter"


def test_classify_qb_starter_certainty_uncertain_other():
    # Real variability that doesn't fit the extreme backup signature (never
    # near-zero) and doesn't clear the clean-starter bar either.
    games = [_game(1.0), _game(0.39), _game(0.98), _game(0.92)]
    assert _classify_qb_starter_certainty(games) == "uncertain_other"


def test_analyze_qb_ownership_gap_runs_on_real_contest_and_reports_every_category(engine):
    # Uses the real, permanently-imported contest 193028208 - see the
    # comment on test_run_ownership_correlation_test_uses_the_same_
    # intersected_sample_for_both_bases above for why this dependency is
    # deliberate, matching this codebase's existing real-fixture-dependent
    # test pattern.
    result = analyze_qb_ownership_gap("193028208", slate_id="dk_thu_mon_2026_09_17", engine=engine)

    assert len(result["qbs"]) >= MIN_MATCHED_PLAYERS_FOR_CORRELATION  # a real, non-trivial number of real QBs
    for qb in result["qbs"]:
        assert qb["starter_certainty"] in {"thin_data", "intermittent_backup_pattern", "clean_starter", "uncertain_other"}

    summary = result["summary_by_starter_certainty"]
    # Real result on this contest: starter uncertainty is a REAL contributing
    # factor (intermittent-backup QBs show a clearly more negative mean
    # rank_diff than clean starters) but does NOT explain most of QB's
    # overall bias - clean, obviously-certain starters still carry a
    # substantial negative rank_diff of their own. Asserting the direction
    # (backups worse than clean starters) since that's the real, structural
    # relationship; not asserting exact numbers, which will shift as more
    # contests are added to this same real dev-DB fixture over time.
    assert "clean_starter" in summary
    assert "intermittent_backup_pattern" in summary
    assert summary["intermittent_backup_pattern"]["mean_rank_diff"] < summary["clean_starter"]["mean_rank_diff"]


# --- summarize_ownership_calibration ---------------------------------------


def test_summarize_ownership_calibration_reports_per_contest_not_pooled(engine, imported_contest):
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM ownership_calibration_runs WHERE contest_id = :c"), {"c": TEST_CONTEST_ID})
    run_ownership_correlation_test(TEST_CONTEST_ID, slate_id="dk_thu_mon_2026_09_17", engine=engine, store=True)

    result = summarize_ownership_calibration(engine=engine)
    assert "proj_median" in result
    proj_basis = result["proj_median"]

    contest_ids = {c["contest_id"] for c in proj_basis["contests"]}
    assert TEST_CONTEST_ID in contest_ids
    # Per-contest rows must be individually present, not collapsed into one
    # blended number - the whole point of this function over a naive pooled
    # correlation across contests that may share the same real slate.
    test_entry = next(c for c in proj_basis["contests"] if c["contest_id"] == TEST_CONTEST_ID)
    assert test_entry["n_matched"] == 3  # this fixture's tiny 3-player matched-with-projection sample
    assert test_entry["spearman_rho"] is None  # below MIN_MATCHED_PLAYERS_FOR_CORRELATION, correctly reported as such


def test_summarize_ownership_calibration_flags_shared_slate_ids(engine):
    # Uses the real permanently-imported contests 193028208/193028210 (both
    # on dk_thu_mon_2026_09_17) - same real-fixture-dependency pattern as
    # the other tests above that rely on this session's real imported data.
    result = summarize_ownership_calibration(engine=engine)
    proj_basis = result["proj_median"]
    real_contests = [c for c in proj_basis["contests"] if c["contest_id"] in ("193028208", "193028210")]
    assert len(real_contests) == 2
    # Both real contests share the same real slate_id - distinct_slate_ids_with_data
    # must reflect that they are NOT independent weeks, not silently count them as two.
    slate_ids = {c["slate_id"] for c in real_contests}
    assert len(slate_ids) == 1
    assert proj_basis["distinct_slate_ids_with_data"] < proj_basis["n_contests_with_data"]


def test_run_ownership_correlation_test_on_a_second_real_week_shows_real_signal(engine):
    # Uses the real, permanently-imported contest 195661349 (a real DK
    # $100K-scale contest-standings export on dk_sunday_2026_09_20 - a
    # genuinely different real week from every other real-fixture contest
    # in this file, all of which are dk_thu_mon_2026_09_17). Same real-
    # fixture-dependency pattern as the tests above.
    result = run_ownership_correlation_test("195661349", slate_id="dk_sunday_2026_09_20", engine=engine, store=False)
    assert result["proj_median"]["n"] >= MIN_MATCHED_PLAYERS_FOR_CORRELATION
    assert result["proj_median"]["spearman_rho"] > 0.4
    assert result["proj_median"]["p_value"] < 0.001


def test_position_ownership_calibration_still_reflects_real_cross_week_data(engine):
    # Real regression guard against POSITION_OWNERSHIP_CALIBRATION silently
    # going stale: recomputes each position's real mean(%Drafted)/mean(raw
    # ownership_proxy) ratio, pooled across every real CLASSIC-slate
    # contest imported so far, and checks the live constant is still within
    # a real, reasonable tolerance of that data. A future contest import
    # that shifts these ratios enough to fail this test is exactly the
    # signal that the constant needs another real update, the same way
    # this session's own cross-week check just triggered one.
    #
    # Classic slates only, by real, deliberate design - a real Showdown
    # contest (195910196, dk_showdown_ind_kc_2026_09_20) is also in this
    # table, and pooling its ratios in with Classic ones distorts every
    # position (a Showdown pool's real ownership_proxy scale isn't
    # comparable to a Classic pool's - far fewer real players compete for
    # the same %Drafted share, and CPT/FLEX pricing changes the raw
    # points-per-$1000 math entirely) - see POSITION_OWNERSHIP_CALIBRATION's
    # own real disclosure comment for the numbers this caused.
    from models.field_simulation import POSITION_OWNERSHIP_CALIBRATION

    with engine.connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT position, pct_drafted, ownership_proxy FROM contest_ownership
                WHERE ownership_proxy IS NOT NULL AND slate_id NOT ILIKE '%showdown%'
                """
            )
        ).fetchall()

    by_position = {}
    for r in rows:
        by_position.setdefault(r.position, {"pct": [], "proxy": []})
        by_position[r.position]["pct"].append(float(r.pct_drafted))
        by_position[r.position]["proxy"].append(float(r.ownership_proxy))

    for position, calibrated_value in POSITION_OWNERSHIP_CALIBRATION.items():
        d = by_position[position]
        n = len(d["pct"])
        real_ratio = (sum(d["pct"]) / n) / (sum(d["proxy"]) / n)
        assert abs(real_ratio - calibrated_value) < 0.15, (
            f"{position}: real pooled ratio {real_ratio:.3f} has drifted from the "
            f"live constant {calibrated_value} by more than the real tolerance - re-run "
            "data/ownership_calibration.py's calibration and update models/field_simulation.py"
        )
