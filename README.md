# NFL DFS Optimizer

A DraftKings NFL daily-fantasy pipeline: real player projections, a MILP
lineup optimizer, a correlated-outcome contest simulator, and real-data-backed
validation for all of it. There is currently **no application layer** — every
workflow below is a Python function you call directly (from a script, a
REPL, or a test). `app/`, `main.py`, and `streamlit_app.py` are unused
scaffolding left over from the original project template; nothing in this
codebase runs through them yet.

## Layout

- `db/` — `schema.sql` (the real source of truth for every table) and
  `migrate.py::run_migrations()` / `get_engine()`, used everywhere else.
- `data/` — everything that gets real data INTO the database: DK salary CSV
  parsing (`dk_salary_csv.py`), nflverse historical stats (`nflverse_fetch.py`,
  `depth_charts.py`), DK↔nflverse player identity resolution
  (`player_crosswalk.py`), pre-lock availability checks (`pre_lock_check.py`,
  `player_availability.py`), contest-standings/ownership import
  (`ownership_calibration.py`), and a pre-slate news scanner (`news_scanner.py`).
- `models/` — the actual engine: projections (`projections.py`), the MILP
  lineup builder (`optimizer.py`), the playing-time/role gate
  (`playing_time_engine.py`), the correlated-outcome simulator
  (`simulation.py`), synthetic-field/ownership-leverage modeling
  (`field_simulation.py`), real payout-curve math (`payout.py`), historical
  backtesting (`backtest.py`, `calibration.py`), and a post-hoc lineup sanity
  reviewer (`lineup_anomaly_review.py`).
- `tests/` — 188 tests, almost all run against a real, already-populated
  Postgres database rather than mocks or fixtures (see Testing / CI below).

## Setup

```
pip install -r requirements.txt
```

Set `PGHOST` / `PGPORT` / `PGDATABASE` / `PGUSER` / `PGPASSWORD` (a `.env`
file works, via `python-dotenv`), then:

```
python db/migrate.py
```

That only creates the schema. This project has no seed/bootstrap script —
real data gets in via the loaders in `data/` (see below), accumulated
organically over real weeks of use. There is currently no reproducible way
to go from an empty database to one the full test suite passes against.

## Core workflows

**Load a slate.** Export a DK salary CSV for a slate, then:

```python
from data.dk_salary_csv import load_slate_player_pool
load_slate_player_pool("dk_sunday_2026_09_27", "/path/to/DKSalaries.csv")
```

Projections come from `models/projections.py` (built from real recent
`player_weekly_stats` history, not hardcoded) and are written per-slate.

**Build a lineup.**

```python
from models.optimizer import generate_lineups
lineups, availability_excluded, injury_report_available = generate_lineups(
    "dk_sunday_2026_09_27",
    num_lineups=5,
    projection_field="proj_ceiling",   # GPP; use "proj_median" for cash
    max_exposure=0.6,
    min_uniques=3,
)
```

Every returned lineup has already passed `validate_lineup` (real DK
roster/salary/slot rules) and the hard availability/role gates in
`data/pre_lock_check.py` and `models/playing_time_engine.py`.

**Rank candidates by simulated win rate** (not just raw projection) via
`models.simulation.select_best_by_simulation` /
`models.optimizer.generate_simulation_selected_lineup` — see
`models/simulation.py`'s module docstring for the real, disclosed limitation
of this simulator (a pairwise Gaussian-copula correlation model, not
play-by-play) and the real null backtest result it's honest about.

**Import real contest ownership/results** after a contest closes, one file:

```python
from data.ownership_calibration import import_contest_standings
import_contest_standings("/path/to/contest-standings-<id>.csv", slate_id, contest_id)
```

or a whole folder at once (the ToS-safe alternative to scraping DK — see
that module's docstring for why an automated scraper was scoped and
rejected):

```python
from data.ownership_calibration import bulk_import_contest_standings
bulk_import_contest_standings("/path/to/folder", slate_id="dk_sunday_2026_09_27")
```

**Real payout curves**, keyed by `contest_id` (`models/payout.py`), feed
into `run_field_simulation(..., contest_id=...)` and
`backtest_slate(..., contest_id=...)` for `cash_pct` / expected-payout
estimates — see that module's docstring for why most real contests only
have a partial curve on record, and why that's tracked explicitly rather
than guessed at.

## Testing / CI

```
python -m pytest tests/
```

`.github/workflows/tests.yml` runs on every push: a `lint` job (ruff, real
gate) and a `pytest` job against a freshly migrated, **empty** CI Postgres
container. That pytest job is deliberately **not** a merge gate yet
(`continue-on-error: true`) — most of this suite exercises real historical
NFL data, real DK slate pools, and real imported contests that only exist
in a database built up through actual use, so a clean CI checkout currently
fails ~74 of 188 tests for lack of that data, not because anything is
broken. The job is still real signal (a pass count that drops below the
current ~112 means something genuinely regressed); making it a hard gate
would need a real seed-data strategy, which doesn't exist yet.

## Known limitations (the honest state, not a wish list)

- **No live ownership feed.** Every "ownership" number in this codebase is
  `ownership_proxy` (points-per-$1000 salary, position-calibrated against
  real contest data) standing in for it — see `models/field_simulation.py`.
- **Field-simulation concentration is uncalibrated** (`DEFAULT_CONCENTRATION`
  in `models/field_simulation.py`) — there isn't enough real ownership data
  yet to know what value matches a real DK field's actual duplication.
- **`backtest_slate` reuses the current salary grid against past weeks'
  real outcomes** (no historical DK salary archive exists) — a disclosed
  approximation, not a true historical reconstruction.
- **No application layer.** Everything above is called directly; there is
  no CLI, API, or UI on top of it yet.
