# NBA DraftKings Optimizer — Starter Scaffold

A three-stage pipeline: pull data → build projections → solve for the optimal lineup.

## Setup (run this in Claude Code, in your project folder)

```bash
pip install -r requirements.txt
```

## Step 1 — Pull player data and team defense stats

```bash
python fetch_data.py --season 2025-26 --last-n-games 15
```

This saves:
- `data/player_form_summary.csv` — season avg vs. last-15-game avg DK points per player
- `data/team_defense_ranks.csv` — team defensive ratings

## Step 2 — Get today's DraftKings salary export

Download the salary CSV from the DraftKings contest lobby ("Export to CSV" button
next to the player pool). Save it as `data/dk_salaries.csv`.

> DK's export column names shift occasionally — if `projections.py` errors on
> missing columns, open the CSV and check the header row, then adjust the
> column names in `load_slate()`.

## Step 3 — Build projections

```bash
python projections.py --slate data/dk_salaries.csv --recent-weight 0.65
```

`--recent-weight` controls how much recent form (vs. season average) drives
the projection. 0.65 means 65% recent form / 35% season average — a
reasonable starting point. Push it toward 0.8+ if you want to chase hot
streaks harder, or toward 0.4 if you want more stability.

## Step 4 — Generate optimal lineup(s)

```bash
python optimizer.py --projections data/projections.csv --lineups 3
```

Generates 3 unique lineups (useful for GPP tournament entries where you want
lineup diversity). Use `--lineups 1` for cash games / 50-50s.

## Where to take this next

This scaffold is intentionally simple so you have a working end-to-end
pipeline fast. The highest-leverage upgrades, roughly in order of effort vs.
payoff:

1. **Backtest first.** Before trusting this with real money, run it against
   5-10 past slates where you already know the results. Compare projected
   vs. actual DK points to see how far off the model runs.
2. **Defense-vs-position (DVP) data** instead of overall team defensive
   rating — how a team defends point guards specifically matters more than
   its overall defensive rating.
3. **Injury/news feed** — a projection is worthless if a starter is ruled
   out an hour before tip-off. This isn't automated in the scaffold yet.
4. **Ownership projections** — for GPPs, being right when the field is
   wrong is what wins big; a low-owned, high-upside player is often better
   than the "obvious" chalk play even at a similar projection.
5. **Regression model** — once you've got a season of backtested data,
   swap the simple weighted-average projection for a proper regression
   (scikit-learn) using minutes, usage rate, pace, and matchup as features.

## Honest expectation

This gets you a statistically-grounded lineup, not a guaranteed winner.
Cash games (beat the median) are the more realistic target for a model
like this; GPP tournaments still carry heavy variance no matter how good
the projections are.
