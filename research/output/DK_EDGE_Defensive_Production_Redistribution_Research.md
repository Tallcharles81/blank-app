# DK EDGE — DEFENSIVE PRODUCTION REDISTRIBUTION RESEARCH

Generated from `research/output/redistribution_results.json` by `scripts/run_redistribution_research.py`. Train [2019, 2020, 2021, 2022, 2023], validation [2024], untouched test [2025].

## A. Executive summary

- Position-group effects exist but are small. A defense's effect on the WR / TE / RB share of targets clears a permutation test (p = 0.003 / 0.017 / 0.010), with a true between-defense SD of 1.7 / 1.2 / 1.3 share points. A full 17-game sample recovers only 0.31 / 0.25 / 0.28 of that (reliability); year-over-year r = 0.20 / 0.05 / 0.20.
- Role-level effects inside a group are not distinguishable from noise: WR1 NO RELIABLE SIGNAL (perm p 0.153, YoY r -0.04), WR2 NO RELIABLE SIGNAL (perm p 0.123, YoY r 0.08), TE1 NO RELIABLE SIGNAL (perm p 0.186, YoY r -0.01), RB1 WEAK SIGNAL (perm p 0.096, YoY r 0.13), RB2 NO RELIABLE SIGNAL (perm p 0.419, YoY r 0.05). Within-group splits (a role's share of its own group) show the same: WR1 NO RELIABLE SIGNAL, WR2 NO RELIABLE SIGNAL, WR3+ WEAK SIGNAL, TE1 NO RELIABLE SIGNAL, RB1 NO RELIABLE SIGNAL, RB2 NO RELIABLE SIGNAL.
- Redistribution tests: 36 tests of whether share taken from WR1, WR2, TE1 or RB1 goes to a specific other role more than proportionally; 0 had p < 0.05 (about 1.8 expected by chance) and 0 survived FDR (q < 0.10). RB rushing suppression → RB receiving: cross-half r = -0.047. RB1 → RB2 carry split: cross-half r = 0.009.
- Out-of-sample target shares (mean abs error per role): no defense 0.04374, generic defense-vs-position 0.04375, role redistribution 0.04373 (2025 test). Role vs generic: NO OUT-OF-SAMPLE IMPROVEMENT; role vs none: NO OUT-OF-SAMPLE IMPROVEMENT. Role model minus no-defense in the 2025 test: -0.00016 total share points per team-game (≈ 0.01 targets per team-game at 35 targets).
- Carry shares: generic group model vs no-defense: NO OUT-OF-SAMPLE IMPROVEMENT (2025 diff -0.00062, p 0.162); RB1/RB2 role split vs generic in 2025: diff 0.00069 (p 0.149; positive = role split worse).
- DFS points (WR/TE/RB, 2025 test, n = 5435), all models league-calibrated: MAE no-defense (A) 4.106, generic 4.106, role (B) 4.106, B+matchup (C) 4.105, Vegas terms only, no defense 4.104, C+Vegas (D) 4.103. Paired vs A in test: group 0.000 (p 0.797); role 0.000 (p 0.892); role_matchup -0.001 (p 0.775); script_none -0.001 (p 0.380); script_role_matchup -0.002 (p 0.518). In validation: group -0.002 (p 0.112); role 0.003 (p 0.007); role_matchup 0.010 (p <0.001); script_none -0.001 (p 0.494); script_role_matchup 0.009 (p 0.007).
- Calibration artifact: without the shared league calibration, the role model's DK MAE edge over no-defense is -0.007 (2025) / -0.008 (2024); with it, 0.000 / 0.003. Any gap between those pairs came from defenses' rolling estimates absorbing a league-wide bias in the expectations, not from defense-specific information.
- Pressure changes the target distribution within a game: on pressured dropbacks WR1 share is -3.1 pts, RB1 +1.6 pts, RB2 +1.7 pts, TE1 +0.3 pts; aDOT +1.86 yds and deep-target rate +4.9 pts (deeper, not shallower). A defense's PRE-GAME pressure rate does not forecast any role's target share out of sample (Section I).
- QB: generic QB defense-vs-position vs pressure/coverage model, 2025 test MAE — trailing only 7.117, generic 7.090, pressure/coverage 7.136, both 7.111. Paired vs trailing in test: generic -0.027 (p 0.219), pressure 0.019 (p 0.241); validation: generic 0.023 (p 0.286), pressure 0.011 (p 0.508).
- Positive controls: the same method detects clear, repeatable defense effects on pressure rate, sack rate, aDOT, deep rate, completion rate and coverage shell (Section L), so the null results for role redistribution are not a failure to detect anything.

*Specialized vs generic, judged on 2024 validation and the untouched 2025 test*

| Most important test | Result |
|---|---|
| WR/TE: role redistribution vs generic defense-vs-position (target shares) | NO OUT-OF-SAMPLE IMPROVEMENT |
| WR/TE/RB: DK points, role model minus generic model (2025 test, MAE) | -0.000 DK pts per player-game |
| WR/TE/RB: DK points, role model (B) vs no defense | NO OUT-OF-SAMPLE IMPROVEMENT |
| WR/TE/RB: DK points, generic defense-vs-position vs no defense | NO OUT-OF-SAMPLE IMPROVEMENT |
| RB: RB1/RB2 rushing split vs generic (carry shares) | NO OUT-OF-SAMPLE IMPROVEMENT |
| QB: pressure/coverage model vs trailing baseline (DK points) | NO OUT-OF-SAMPLE IMPROVEMENT |
| QB: generic QB defense-vs-position vs trailing baseline | NO OUT-OF-SAMPLE IMPROVEMENT |

## B. Data sources

- nflverse play-by-play (regular season): targets, receiver, air yards, pass location/length, completions, yards, TDs, red zone, sacks, QB hits, scrambles, rushes.
- nflverse participation: on-field offensive player ids for every play (snap and dropback participation, the route proxy), was_pressure, defense man/zone type, defense coverage type (charted for 89–100% of dropbacks each season).
- nflverse player weekly stats, scored with DK Edge's own DK formula (data/nflverse_fetch.py::_skill_player_fantasy_points).
- nflverse players (position), schedules (closing spread and total, roof, weather).
- PFR per-defender coverage charting (targets/yards allowed per defender per week) via nflverse pfr_advstats.
- FTN charting (2022+) was downloaded (blitzers, pass rushers) but not used in the final models; participation pressure covers every season.

| Season | Games | Split |
|---|---|---|
| 2018 | 256 | warmup |
| 2019 | 256 | train |
| 2020 | 256 | train |
| 2021 | 272 | train |
| 2022 | 271 | train |
| 2023 | 272 | train |
| 2024 | 272 | validation |
| 2025 | 272 | test |

4254 team-games and 78994 player-games. Nothing was read from or written to the DK Edge database.

## C. Role classification methodology

Roles are assigned per game, among players who were active that day, using only each player's usage in his previous 8 games (across seasons). Nothing from the game itself is used except who was active, which is known at lock.

| Role | Rule |
|---|---|
| WR1 / WR2 / WR3+ | Active WRs ranked by trailing share of team targets |
| TE1 / TE2+ | Active TEs ranked by trailing share of offensive snaps |
| RB1 / RB2 / RB3+ | Active RBs/FBs ranked by trailing share of team carries + targets |
| QB | Active QB with the most trailing dropbacks |
| Stratifiers | TE1 targets per dropback snap (receiving vs blocking-heavy), RB committee (top RB < 60% of RB carries), receiving-RB identity, QB designed runs per game |

Expected production for a role = the trailing target (or carry) share of the players filling it, with thin histories pulled toward a position prior, normalized across every active player so a team-game's expected shares sum to exactly 1. Actual shares also sum to 1, so every model respects the team's real pass volume.

Slot vs outside alignment, routes run, and CB-to-receiver assignments are not in any public data source (see Section R). Dropback participation (on the field for a dropback) stands in for routes; pass location (middle vs outside) is a target-location measure, not alignment.

## D. Defense-by-defense fingerprints (end of 2025, last 17 games)

Each cell: shrunk effect vs expectation (role cells: % of the league-average share for that role; aDOT: yards) and its evidence label. Shrinkage for each metric uses that metric's own estimated between-defense variance (k = σ²/τ²), so a metric with little real between-defense variance is shrunk hard toward zero. Labels: HIGH = |effect| ≥ 3 posterior SD and reliability ≥ 0.5; MEDIUM ≥ 2 SD; LOW otherwise.

Role-cell labels across all 32 defenses: LOW: 192

| Defense | Games | WR1 tgt | WR2 tgt | WR3+ tgt | TE1 tgt | RB1 tgt | RB2 tgt | Pressure | Sack rate | aDOT |
|---|---|---|---|---|---|---|---|---|---|---|
| ARI | 17 | +0.9% (LOW) | -0.6% (LOW) | -4.5% (LOW) | +2.4% (LOW) | +1.0% (LOW) | -0.2% (LOW) | -3.8% (LOW) | -0.9% (LOW) | -0.32 (LOW) |
| ATL | 17 | +0.8% (LOW) | +0.2% (LOW) | +1.7% (LOW) | -0.7% (LOW) | +2.3% (LOW) | +0.0% (LOW) | +2.9% (LOW) | +14.3% (LOW) | +0.40 (LOW) |
| BAL | 17 | +1.2% (LOW) | -0.7% (LOW) | +3.4% (LOW) | -1.4% (LOW) | +1.0% (LOW) | -0.2% (LOW) | -3.5% (LOW) | -10.4% (LOW) | +0.43 (LOW) |
| BUF | 17 | +2.4% (LOW) | +0.0% (LOW) | -4.0% (LOW) | -0.4% (LOW) | +2.5% (LOW) | +0.0% (LOW) | +4.7% (LOW) | +4.4% (LOW) | -0.05 (LOW) |
| CAR | 17 | +0.9% (LOW) | +0.7% (LOW) | -4.9% (LOW) | -0.6% (LOW) | -1.5% (LOW) | +0.3% (LOW) | -9.1% (LOW) | -4.2% (LOW) | -0.01 (LOW) |
| CHI | 17 | +0.6% (LOW) | +0.3% (LOW) | -3.3% (LOW) | +2.1% (LOW) | +0.3% (LOW) | -0.1% (LOW) | -3.6% (LOW) | -3.5% (LOW) | +0.11 (LOW) |
| CIN | 17 | -0.0% (LOW) | -3.2% (LOW) | -1.8% (LOW) | +1.7% (LOW) | -2.0% (LOW) | +0.3% (LOW) | -0.2% (LOW) | -0.3% (LOW) | +0.00 (LOW) |
| CLE | 17 | +0.4% (LOW) | +2.2% (LOW) | -0.7% (LOW) | -0.9% (LOW) | +0.1% (LOW) | -0.0% (LOW) | +6.4% (LOW) | +16.3% (LOW) | +0.03 (LOW) |
| DAL | 17 | +0.0% (LOW) | -0.2% (LOW) | -1.3% (LOW) | -0.5% (LOW) | +5.1% (LOW) | -0.1% (LOW) | +0.3% (LOW) | -13.7% (LOW) | +0.22 (LOW) |
| DEN | 17 | +0.9% (LOW) | +1.5% (LOW) | -0.4% (LOW) | -0.2% (LOW) | -3.0% (LOW) | -0.1% (LOW) | +5.5% (LOW) | +22.3% (MEDIUM) | +0.29 (LOW) |
| DET | 17 | -1.0% (LOW) | -1.4% (LOW) | +3.5% (LOW) | +0.6% (LOW) | -0.1% (LOW) | -0.1% (LOW) | -0.0% (LOW) | +9.1% (LOW) | +0.93 (MEDIUM) |
| GB | 17 | -0.6% (LOW) | +0.5% (LOW) | +0.5% (LOW) | +0.7% (LOW) | -1.4% (LOW) | +0.1% (LOW) | -0.1% (LOW) | -6.4% (LOW) | -0.24 (LOW) |
| HOU | 17 | -0.3% (LOW) | +1.0% (LOW) | -1.5% (LOW) | +0.2% (LOW) | +0.7% (LOW) | +0.0% (LOW) | +1.0% (LOW) | +6.2% (LOW) | +0.35 (LOW) |
| IND | 17 | -0.7% (LOW) | -0.0% (LOW) | +2.3% (LOW) | -0.2% (LOW) | +0.1% (LOW) | -0.1% (LOW) | +1.8% (LOW) | -0.6% (LOW) | +0.11 (LOW) |
| JAX | 17 | +1.8% (LOW) | -1.9% (LOW) | -0.6% (LOW) | +0.6% (LOW) | +0.6% (LOW) | -0.0% (LOW) | -5.5% (LOW) | -15.0% (LOW) | +0.14 (LOW) |
| KC | 17 | +1.3% (LOW) | +1.2% (LOW) | -4.3% (LOW) | +1.0% (LOW) | -1.1% (LOW) | +0.1% (LOW) | +3.1% (LOW) | -1.0% (LOW) | -0.54 (LOW) |
| LA | 17 | +0.5% (LOW) | -1.7% (LOW) | +0.5% (LOW) | +0.8% (LOW) | +2.0% (LOW) | -0.1% (LOW) | +2.6% (LOW) | +5.0% (LOW) | -0.08 (LOW) |
| LAC | 17 | +1.0% (LOW) | +0.9% (LOW) | +1.7% (LOW) | -0.9% (LOW) | +1.7% (LOW) | -0.0% (LOW) | +1.1% (LOW) | +10.3% (LOW) | -0.06 (LOW) |
| LV | 17 | -0.9% (LOW) | +0.6% (LOW) | +0.5% (LOW) | +0.9% (LOW) | +1.3% (LOW) | -0.1% (LOW) | -4.6% (LOW) | -3.2% (LOW) | -0.32 (LOW) |
| MIA | 17 | -0.6% (LOW) | -3.1% (LOW) | +0.5% (LOW) | +1.7% (LOW) | +2.3% (LOW) | +0.0% (LOW) | -4.4% (LOW) | +1.8% (LOW) | -0.47 (LOW) |
| MIN | 17 | -1.1% (LOW) | +0.9% (LOW) | -1.4% (LOW) | +0.4% (LOW) | +2.3% (LOW) | +0.0% (LOW) | +1.9% (LOW) | +24.9% (MEDIUM) | -0.71 (LOW) |
| NE | 17 | +0.9% (LOW) | +0.5% (LOW) | -3.5% (LOW) | +0.3% (LOW) | +2.8% (LOW) | -0.1% (LOW) | -2.8% (LOW) | -7.4% (LOW) | -0.06 (LOW) |
| NO | 17 | +0.2% (LOW) | +1.0% (LOW) | -0.4% (LOW) | +0.2% (LOW) | -3.7% (LOW) | -0.1% (LOW) | -1.1% (LOW) | +11.4% (LOW) | +0.30 (LOW) |
| NYG | 17 | +0.9% (LOW) | -0.8% (LOW) | -0.6% (LOW) | +1.2% (LOW) | +2.6% (LOW) | -0.1% (LOW) | -1.9% (LOW) | -0.7% (LOW) | +0.08 (LOW) |
| NYJ | 17 | +1.4% (LOW) | +0.3% (LOW) | +0.8% (LOW) | -1.1% (LOW) | -2.6% (LOW) | -0.1% (LOW) | -8.3% (LOW) | -14.2% (LOW) | -0.29 (LOW) |
| PHI | 17 | -0.1% (LOW) | +2.5% (LOW) | -1.5% (LOW) | +1.3% (LOW) | -0.1% (LOW) | -0.1% (LOW) | +2.8% (LOW) | +4.0% (LOW) | +0.54 (LOW) |
| PIT | 17 | +0.7% (LOW) | -1.2% (LOW) | -0.7% (LOW) | -0.1% (LOW) | +0.3% (LOW) | -0.0% (LOW) | -2.3% (LOW) | -0.7% (LOW) | -0.34 (LOW) |
| SEA | 17 | -1.1% (LOW) | +1.1% (LOW) | -1.6% (LOW) | +0.3% (LOW) | +0.8% (LOW) | +0.3% (LOW) | +6.4% (LOW) | +2.4% (LOW) | -0.70 (LOW) |
| SF | 17 | -0.6% (LOW) | -1.0% (LOW) | +0.4% (LOW) | +0.6% (LOW) | +3.0% (LOW) | -0.0% (LOW) | -7.5% (LOW) | -17.6% (LOW) | -0.01 (LOW) |
| TB | 17 | -0.1% (LOW) | -0.5% (LOW) | +1.4% (LOW) | +0.5% (LOW) | +0.1% (LOW) | -0.1% (LOW) | +5.9% (LOW) | -4.7% (LOW) | +0.14 (LOW) |
| TEN | 17 | -0.2% (LOW) | +0.3% (LOW) | +0.2% (LOW) | +1.7% (LOW) | -1.2% (LOW) | +0.1% (LOW) | -0.0% (LOW) | +7.1% (LOW) | -0.06 (LOW) |
| WAS | 17 | -0.4% (LOW) | +1.8% (LOW) | -2.9% (LOW) | +0.5% (LOW) | -0.1% (LOW) | -0.1% (LOW) | +0.7% (LOW) | +10.5% (LOW) | -0.05 (LOW) |

## E. WR1 / WR2 / WR3+ findings

*Defense effect on each WR role's share of team targets (train 2019–2023)*

| Role | Avg share | True defense SD (share pts) | Rel. 17 games | Perm p | Split-half r | YoY r | Verdict |
|---|---|---|---|---|---|---|---|
| WR1 | 0.233 | 0.73 | 0.10 | 0.153 | 0.142 | -0.037 | NO RELIABLE SIGNAL |
| WR2 | 0.167 | 0.71 | 0.11 | 0.123 | 0.113 | 0.082 | NO RELIABLE SIGNAL |
| WR3+ | 0.190 | 1.05 | 0.17 | 0.086 | 0.213 | 0.035 | WEAK SIGNAL |

*Defense effect on each WR role's share of WR targets only*

| Role | Avg share | True defense SD (share pts) | Rel. 17 games | Perm p | Split-half r | YoY r | Verdict |
|---|---|---|---|---|---|---|---|
| WR1 | — | 0.40 | 0.01 | 0.412 | 0.099 | -0.002 | NO RELIABLE SIGNAL |
| WR2 | — | 1.30 | 0.14 | 0.116 | 0.114 | 0.050 | NO RELIABLE SIGNAL |
| WR3+ | — | 0.87 | 0.05 | 0.478 | 0.178 | 0.001 | WEAK SIGNAL |

*Where WR1's lost share goes: cross-half correlation between a defense's WR1 residual and each other role's share of the remaining targets (0 = proportional reallocation)*

| Receiving role | Cross-half r | p | Defense-seasons |
|---|---|---|---|
| WR2 | +0.030 | 0.710 | 160 |
| WR3+ | +0.006 | 0.945 | 160 |
| TE1 | -0.003 | 0.966 | 160 |
| TE2+ | -0.061 | 0.445 | 160 |
| RB1 | -0.013 | 0.866 | 160 |
| RB2 | +0.009 | 0.906 | 160 |
| RB3+ | +0.002 | 0.982 | 160 |

*Same test for WR2*

| Receiving role | Cross-half r | p | Defense-seasons |
|---|---|---|---|
| WR1 | +0.026 | 0.742 | 160 |
| WR3+ | -0.003 | 0.973 | 160 |
| TE1 | +0.055 | 0.490 | 160 |
| TE2+ | -0.041 | 0.605 | 160 |
| RB1 | -0.024 | 0.766 | 160 |
| RB2 | +0.011 | 0.888 | 160 |
| RB3+ | -0.046 | 0.561 | 160 |

| Predictor → outcome | Train | p | q | 2024 | p | 2025 | p | Category |
|---|---|---|---|---|---|---|---|---|
| def_pressure_pre_z -> WR1 target share | +0.0008 | 0.666 | 0.825 | +0.0045 | 0.252 | +0.0041 | 0.306 | NO RELIABLE SIGNAL |
| def_pressure_pre_z -> WR2 target share | -0.0012 | 0.476 | 0.740 | +0.0026 | 0.476 | +0.0036 | 0.302 | NO RELIABLE SIGNAL |
| def_pressure_pre_z -> WR3+ target share | -0.0022 | 0.235 | 0.510 | -0.0023 | 0.587 | -0.0004 | 0.927 | NO RELIABLE SIGNAL |
| def_man_pre_z -> WR1 target share | +0.0009 | 0.634 | 0.815 | +0.0047 | 0.239 | +0.0081 | 0.046 | WEAK SIGNAL |
| def_man_pre_z -> WR2 target share | +0.0017 | 0.289 | 0.569 | +0.0039 | 0.273 | +0.0034 | 0.336 | NO RELIABLE SIGNAL |
| def_man_pre_z -> WR3+ target share | +0.0048 | 0.010 | 0.065 | +0.0039 | 0.366 | -0.0042 | 0.266 | WEAK SIGNAL |
| def_two_high_pre_z -> WR1 target share | -0.0016 | 0.367 | 0.660 | +0.0001 | 0.983 | -0.0049 | 0.223 | NO RELIABLE SIGNAL |
| def_two_high_pre_z -> WR2 target share | -0.0030 | 0.069 | 0.274 | -0.0102 | 0.004 | -0.0008 | 0.824 | WEAK SIGNAL |
| def_two_high_pre_z -> WR3+ target share | +0.0011 | 0.569 | 0.774 | +0.0013 | 0.752 | -0.0033 | 0.384 | NO RELIABLE SIGNAL |
| off_favored_by -> WR1 target share | +0.0005 | 0.063 | 0.274 | -0.0008 | 0.229 | +0.0001 | 0.840 | NO RELIABLE SIGNAL |
| off_favored_by -> WR2 target share | -0.0002 | 0.500 | 0.744 | +0.0004 | 0.535 | -0.0008 | 0.160 | NO RELIABLE SIGNAL |
| off_favored_by -> WR3+ target share | -0.0003 | 0.310 | 0.591 | -0.0002 | 0.766 | -0.0010 | 0.087 | NO RELIABLE SIGNAL |
| total_line -> WR1 target share | +0.0007 | 0.090 | 0.334 | +0.0006 | 0.570 | +0.0009 | 0.361 | NO RELIABLE SIGNAL |
| total_line -> WR2 target share | -0.0005 | 0.138 | 0.394 | -0.0004 | 0.659 | -0.0007 | 0.428 | NO RELIABLE SIGNAL |
| total_line -> WR3+ target share | -0.0000 | 0.936 | 0.987 | -0.0013 | 0.239 | -0.0019 | 0.032 | WEAK SIGNAL |

## F. TE findings

*Defense effect on TE roles' share of team targets (train)*

| Role | Avg share | True defense SD (share pts) | Rel. 17 games | Perm p | Split-half r | YoY r | Verdict |
|---|---|---|---|---|---|---|---|
| TE1 | 0.137 | 0.50 | 0.08 | 0.186 | 0.025 | -0.007 | NO RELIABLE SIGNAL |
| TE2+ | 0.077 | 0.68 | 0.18 | 0.043 | 0.151 | 0.098 | WEAK SIGNAL |

*Defense effect on the TE group's share (generic)*

| Role | Avg share | True defense SD (share pts) | Rel. 17 games | Perm p | Split-half r | YoY r | Verdict |
|---|---|---|---|---|---|---|---|
| TE | 0.213 | 1.21 | 0.25 | 0.017 | 0.221 | 0.053 | MODERATE SIGNAL |

*Where TE1's lost share goes (cross-half r)*

| Receiving role | Cross-half r | p | Defense-seasons |
|---|---|---|---|
| WR1 | -0.030 | 0.711 | 160 |
| WR2 | +0.038 | 0.630 | 160 |
| WR3+ | -0.098 | 0.220 | 160 |
| TE2+ | +0.127 | 0.109 | 160 |
| RB1 | +0.040 | 0.620 | 160 |
| RB2 | +0.001 | 0.989 | 160 |
| RB3+ | -0.034 | 0.667 | 160 |

| Predictor → outcome | Train | p | q | 2024 | p | 2025 | p | Category |
|---|---|---|---|---|---|---|---|---|
| def_pressure_pre_z -> TE1 target share | +0.0022 | 0.108 | 0.356 | -0.0019 | 0.552 | +0.0005 | 0.864 | NO RELIABLE SIGNAL |
| def_pressure_pre_z -> TE2+ target share | +0.0001 | 0.920 | 0.987 | -0.0006 | 0.838 | -0.0037 | 0.202 | NO RELIABLE SIGNAL |
| def_man_pre_z -> TE1 target share | -0.0018 | 0.200 | 0.483 | +0.0018 | 0.582 | -0.0045 | 0.145 | NO RELIABLE SIGNAL |
| def_man_pre_z -> TE2+ target share | +0.0016 | 0.180 | 0.465 | -0.0021 | 0.452 | -0.0034 | 0.250 | NO RELIABLE SIGNAL |
| def_two_high_pre_z -> TE1 target share | +0.0010 | 0.481 | 0.740 | -0.0008 | 0.818 | +0.0035 | 0.263 | NO RELIABLE SIGNAL |
| def_two_high_pre_z -> TE2+ target share | +0.0007 | 0.581 | 0.774 | -0.0001 | 0.975 | +0.0032 | 0.270 | NO RELIABLE SIGNAL |
| off_favored_by -> TE1 target share | +0.0002 | 0.260 | 0.547 | +0.0003 | 0.651 | +0.0009 | 0.067 | NO RELIABLE SIGNAL |
| off_favored_by -> TE2+ target share | -0.0002 | 0.351 | 0.650 | +0.0004 | 0.424 | +0.0008 | 0.077 | NO RELIABLE SIGNAL |
| total_line -> TE1 target share | +0.0001 | 0.865 | 0.973 | +0.0001 | 0.864 | +0.0007 | 0.368 | NO RELIABLE SIGNAL |
| total_line -> TE2+ target share | -0.0002 | 0.543 | 0.774 | -0.0001 | 0.874 | +0.0009 | 0.225 | NO RELIABLE SIGNAL |

## G. RB1 / RB2 findings (rushing)

*Defense effect on share of team carries (train)*

| Role | Avg share | True defense SD (share pts) | Rel. 17 games | Perm p | Split-half r | YoY r | Verdict |
|---|---|---|---|---|---|---|---|
| RB1 | 0.578 | 0.86 | 0.04 | 0.422 | -0.030 | -0.031 | NO RELIABLE SIGNAL |
| RB2 | 0.235 | 0.71 | 0.03 | 0.379 | 0.049 | 0.100 | NO RELIABLE SIGNAL |
| QB | 0.056 | 0.59 | 0.14 | 0.140 | 0.089 | 0.083 | NO RELIABLE SIGNAL |

RB1 → RB2 carry redistribution (does a defense that cuts RB1's carry share hand them to RB2?): cross-half r = 0.009 (n = 160 defense-seasons).

*Carry-share prediction, mean abs error per role*

| Split | No defense (A) | Generic DvP | Role (B) | Vegas terms, no defense | Role + Vegas (D) |
|---|---|---|---|---|---|
| train | 0.04429 | 0.04426 | 0.04428 | 0.04426 | 0.04425 |
| validation | 0.04105 | 0.04101 | 0.04107 | 0.04094 | 0.04097 |
| test | 0.04123 | 0.04117 | 0.04124 | 0.04127 | 0.04127 |

*Carry-share paired differences (negative = second model better)*

| Split | Comparison | Mean diff | 95% CI | p | Team-games |
|---|---|---|---|---|---|
| train | generic vs no-defense | -0.00027 | -0.00072 to +0.00017 | 0.229 | 2654 |
| train | role vs no-defense | -0.00007 | -0.00032 to +0.00017 | 0.613 | 2654 |
| train | role vs generic | +0.00020 | -0.00027 to +0.00069 | 0.388 | 2654 |
| validation | generic vs no-defense | -0.00039 | -0.00134 to +0.00048 | 0.404 | 544 |
| validation | role vs no-defense | +0.00025 | -0.00024 to +0.00072 | 0.325 | 544 |
| validation | role vs generic | +0.00064 | -0.00022 to +0.00152 | 0.181 | 544 |
| test | generic vs no-defense | -0.00062 | -0.00150 to +0.00019 | 0.162 | 544 |
| test | role vs no-defense | +0.00008 | -0.00048 to +0.00064 | 0.787 | 544 |
| test | role vs generic | +0.00069 | -0.00026 to +0.00169 | 0.149 | 544 |

## H. RB receiving findings

*Defense effect on RB roles' share of team targets (train)*

| Role | Avg share | True defense SD (share pts) | Rel. 17 games | Perm p | Split-half r | YoY r | Verdict |
|---|---|---|---|---|---|---|---|
| RB1 | 0.099 | 0.61 | 0.13 | 0.096 | 0.111 | 0.126 | WEAK SIGNAL |
| RB2 | 0.064 | 0.11 | 0.01 | 0.419 | -0.002 | 0.049 | NO RELIABLE SIGNAL |
| RB3+ | 0.032 | 0.48 | 0.18 | 0.037 | 0.038 | -0.015 | WEAK SIGNAL |

*Defense effect on the RB group's target share (generic)*

| Role | Avg share | True defense SD (share pts) | Rel. 17 games | Perm p | Split-half r | YoY r | Verdict |
|---|---|---|---|---|---|---|---|
| RB | 0.195 | 1.30 | 0.28 | 0.010 | 0.221 | 0.203 | MODERATE SIGNAL |

*Where RB1's lost target share goes (cross-half r)*

| Receiving role | Cross-half r | p | Defense-seasons |
|---|---|---|---|
| WR1 | -0.014 | 0.863 | 160 |
| WR2 | -0.025 | 0.750 | 160 |
| WR3+ | -0.092 | 0.247 | 160 |
| TE1 | +0.042 | 0.596 | 160 |
| TE2+ | -0.010 | 0.899 | 160 |
| RB2 | +0.072 | 0.369 | 160 |
| RB3+ | +0.136 | 0.087 | 160 |

RB rushing suppression → RB receiving increase: cross-half r between a defense's RB carry-share residual and its RB target-share residual = -0.047 (half-season p values 0.713, 0.123; n = 160).

| Predictor → outcome | Train | p | q | 2024 | p | 2025 | p | Category |
|---|---|---|---|---|---|---|---|---|
| def_pressure_pre_z -> RB1 target share | -0.0005 | 0.668 | 0.825 | -0.0016 | 0.553 | -0.0025 | 0.364 | NO RELIABLE SIGNAL |
| def_pressure_pre_z -> RB2 target share | +0.0000 | 0.995 | 0.995 | -0.0004 | 0.867 | +0.0005 | 0.803 | NO RELIABLE SIGNAL |
| def_man_pre_z -> RB1 target share | -0.0045 | <0.001 | 0.004 | -0.0103 | <0.001 | +0.0034 | 0.220 | MODERATE SIGNAL |
| def_man_pre_z -> RB2 target share | -0.0020 | 0.056 | 0.273 | -0.0017 | 0.428 | -0.0022 | 0.294 | NO RELIABLE SIGNAL |
| def_two_high_pre_z -> RB1 target share | -0.0001 | 0.928 | 0.987 | +0.0079 | 0.004 | -0.0018 | 0.523 | NO RELIABLE SIGNAL |
| def_two_high_pre_z -> RB2 target share | +0.0012 | 0.230 | 0.510 | +0.0018 | 0.394 | +0.0034 | 0.105 | NO RELIABLE SIGNAL |
| off_favored_by -> RB1 target share | +0.0000 | 0.808 | 0.925 | +0.0005 | 0.315 | +0.0001 | 0.798 | NO RELIABLE SIGNAL |
| off_favored_by -> RB2 target share | -0.0000 | 0.940 | 0.987 | +0.0000 | 0.939 | +0.0001 | 0.865 | NO RELIABLE SIGNAL |
| total_line -> RB1 target share | +0.0001 | 0.802 | 0.925 | +0.0005 | 0.477 | +0.0009 | 0.193 | NO RELIABLE SIGNAL |
| total_line -> RB2 target share | -0.0004 | 0.122 | 0.365 | +0.0004 | 0.422 | -0.0005 | 0.356 | NO RELIABLE SIGNAL |

## I. QB pressure findings

Play level, within the same game (132905 charted dropbacks, 2019, 2020, 2021, 2022, 2023, 2024, 2025; pressured on 29.5% of dropbacks). Differences are pressured minus clean, averaged within games; 95% CIs resample games.

| Metric | Pressured | Clean | Within-game diff | 95% CI | p |
|---|---|---|---|---|---|
| sack rate | 0.102 | 0.000 | +0.0994 | +0.0931 to +0.1048 | <0.001 |
| scramble rate | 0.035 | 0.019 | +0.0143 | +0.0118 to +0.0168 | <0.001 |
| int rate per attempt | 0.032 | 0.019 | +0.0125 | +0.0103 to +0.0148 | <0.001 |
| completion rate | 0.493 | 0.705 | -0.2170 | -0.2239 to -0.2099 | <0.001 |
| adot | 9.316 | 7.422 | +1.8610 | +1.6894 to +2.0261 | <0.001 |
| deep target rate | 0.153 | 0.105 | +0.0487 | +0.0441 to +0.0539 | <0.001 |

| Target share | Pressured | Clean | Within-game diff | 95% CI | p |
|---|---|---|---|---|---|
| WR1 | 0.212 | 0.242 | -0.0307 | -0.0365 to -0.0249 | <0.001 |
| WR2 | 0.158 | 0.169 | -0.0124 | -0.0168 to -0.0071 | <0.001 |
| WR3+ | 0.188 | 0.191 | -0.0031 | -0.0085 to +0.0023 | 0.275 |
| TE1 | 0.141 | 0.139 | +0.0034 | -0.0016 to +0.0080 | 0.188 |
| TE2+ | 0.081 | 0.077 | +0.0051 | +0.0014 to +0.0085 | 0.007 |
| RB1 | 0.114 | 0.097 | +0.0162 | +0.0115 to +0.0204 | <0.001 |
| RB2 | 0.072 | 0.055 | +0.0168 | +0.0131 to +0.0205 | <0.001 |
| RB3+ | 0.033 | 0.028 | +0.0049 | +0.0025 to +0.0073 | <0.001 |

Predictive (pre-game): a defense's trailing pressure rate, z-scored within season, against the offense's residual in that game. Slope per 1 SD.

| Predictor → outcome | Train | p | q | 2024 | p | 2025 | p | Category |
|---|---|---|---|---|---|---|---|---|
| def_pressure_pre_z -> adot_resid | -0.0467 | 0.274 | 0.556 | +0.1203 | 0.202 | +0.0538 | 0.584 | NO RELIABLE SIGNAL |
| def_pressure_pre_z -> sack_rate_resid | +0.0034 | <0.001 | 0.002 | +0.0022 | 0.277 | +0.0069 | 0.001 | MODERATE SIGNAL |
| def_pressure_pre_z -> deep_rate_resid | -0.0016 | 0.217 | 0.507 | +0.0080 | 0.006 | +0.0053 | 0.070 | NO RELIABLE SIGNAL |
| def_pressure_pre_z -> cmp_rate_resid | -0.0048 | 0.011 | 0.065 | -0.0035 | 0.404 | -0.0137 | <0.001 | WEAK SIGNAL |
| def_pressure_pre_z -> int_rate_resid | +0.0009 | 0.101 | 0.353 | +0.0014 | 0.246 | +0.0002 | 0.889 | NO RELIABLE SIGNAL |
| def_pressure_pre_z -> scramble_rate_resid | +0.0002 | 0.757 | 0.900 | -0.0045 | 0.019 | +0.0010 | 0.603 | NO RELIABLE SIGNAL |
| def_pressure_pre_z -> WR1 target share | +0.0008 | 0.666 | 0.825 | +0.0045 | 0.252 | +0.0041 | 0.306 | NO RELIABLE SIGNAL |
| def_pressure_pre_z -> WR2 target share | -0.0012 | 0.476 | 0.740 | +0.0026 | 0.476 | +0.0036 | 0.302 | NO RELIABLE SIGNAL |
| def_pressure_pre_z -> WR3+ target share | -0.0022 | 0.235 | 0.510 | -0.0023 | 0.587 | -0.0004 | 0.927 | NO RELIABLE SIGNAL |
| def_pressure_pre_z -> TE1 target share | +0.0022 | 0.108 | 0.356 | -0.0019 | 0.552 | +0.0005 | 0.864 | NO RELIABLE SIGNAL |
| def_pressure_pre_z -> TE2+ target share | +0.0001 | 0.920 | 0.987 | -0.0006 | 0.838 | -0.0037 | 0.202 | NO RELIABLE SIGNAL |
| def_pressure_pre_z -> RB1 target share | -0.0005 | 0.668 | 0.825 | -0.0016 | 0.553 | -0.0025 | 0.364 | NO RELIABLE SIGNAL |
| def_pressure_pre_z -> RB2 target share | +0.0000 | 0.995 | 0.995 | -0.0004 | 0.867 | +0.0005 | 0.803 | NO RELIABLE SIGNAL |

*QB DK points (baseline = trailing DK/game + train league calibration +0.452; generic shrinkage k = 64; pressure/coverage terms fit on train, DK pts per 1 SD: pressure -0.367, man +0.080, two_high -0.098)*

| Split | QB model | n | MAE | RMSE | Bias | Corr |
|---|---|---|---|---|---|---|
| train | Trailing DK/game | 2506 | 6.979 | 8.753 | -0.000 | 0.311 |
| train | Generic QB DvP | 2506 | 6.964 | 8.726 | -0.036 | 0.318 |
| train | Pressure + coverage | 2506 | 6.964 | 8.745 | +0.002 | 0.313 |
| train | Both | 2506 | 6.956 | 8.722 | -0.035 | 0.319 |
| validation | Trailing DK/game | 522 | 6.928 | 8.774 | -0.335 | 0.281 |
| validation | Generic QB DvP | 522 | 6.951 | 8.789 | -0.323 | 0.278 |
| validation | Pressure + coverage | 522 | 6.939 | 8.772 | -0.337 | 0.281 |
| validation | Both | 522 | 6.960 | 8.786 | -0.325 | 0.279 |
| test | Trailing DK/game | 521 | 7.117 | 8.922 | +0.497 | 0.220 |
| test | Generic QB DvP | 521 | 7.090 | 8.874 | +0.530 | 0.232 |
| test | Pressure + coverage | 521 | 7.136 | 8.940 | +0.495 | 0.217 |
| test | Both | 521 | 7.111 | 8.894 | +0.529 | 0.230 |

## J. CB-WR interaction findings

No public data assigns a specific CB to a specific receiver on a play (models/matchups.py documents the same check). The closest real signals: the defense's best CB by trailing PFR-charted yards/target allowed (min 10 trailing targets), the whole secondary's trailing yards/target allowed, and trailing man-coverage and two-high rates. The existing matchups.py WR adjustment built on the team-level version already backtested as harmful.

| Predictor → outcome | Train | p | q | 2024 | p | 2025 | p | Category |
|---|---|---|---|---|---|---|---|---|
| def_man_pre_z -> adot_resid | +0.2939 | <0.001 | <0.001 | +0.3034 | 0.001 | +0.3214 | <0.001 | STRONG REPEATABLE SIGNAL |
| def_man_pre_z -> sack_rate_resid | -0.0004 | 0.697 | 0.844 | -0.0018 | 0.373 | +0.0024 | 0.251 | NO RELIABLE SIGNAL |
| def_man_pre_z -> deep_rate_resid | +0.0078 | <0.001 | <0.001 | +0.0127 | <0.001 | +0.0077 | 0.009 | STRONG REPEATABLE SIGNAL |
| def_man_pre_z -> cmp_rate_resid | -0.0090 | <0.001 | <0.001 | -0.0095 | 0.023 | -0.0070 | 0.078 | MODERATE SIGNAL |
| def_man_pre_z -> int_rate_resid | +0.0009 | 0.113 | 0.356 | -0.0004 | 0.725 | -0.0006 | 0.619 | NO RELIABLE SIGNAL |
| def_man_pre_z -> scramble_rate_resid | -0.0006 | 0.438 | 0.727 | +0.0049 | 0.010 | +0.0011 | 0.560 | NO RELIABLE SIGNAL |
| def_man_pre_z -> WR1 target share | +0.0009 | 0.634 | 0.815 | +0.0047 | 0.239 | +0.0081 | 0.046 | WEAK SIGNAL |
| def_man_pre_z -> WR2 target share | +0.0017 | 0.289 | 0.569 | +0.0039 | 0.273 | +0.0034 | 0.336 | NO RELIABLE SIGNAL |
| def_man_pre_z -> WR3+ target share | +0.0048 | 0.010 | 0.065 | +0.0039 | 0.366 | -0.0042 | 0.266 | WEAK SIGNAL |
| def_man_pre_z -> TE1 target share | -0.0018 | 0.200 | 0.483 | +0.0018 | 0.582 | -0.0045 | 0.145 | NO RELIABLE SIGNAL |
| def_man_pre_z -> TE2+ target share | +0.0016 | 0.180 | 0.465 | -0.0021 | 0.452 | -0.0034 | 0.250 | NO RELIABLE SIGNAL |
| def_man_pre_z -> RB1 target share | -0.0045 | <0.001 | 0.004 | -0.0103 | <0.001 | +0.0034 | 0.220 | MODERATE SIGNAL |
| def_man_pre_z -> RB2 target share | -0.0020 | 0.056 | 0.273 | -0.0017 | 0.428 | -0.0022 | 0.294 | NO RELIABLE SIGNAL |
| def_two_high_pre_z -> adot_resid | -0.1547 | <0.001 | 0.003 | -0.1862 | 0.048 | -0.3334 | <0.001 | STRONG REPEATABLE SIGNAL |
| def_two_high_pre_z -> sack_rate_resid | +0.0006 | 0.508 | 0.744 | +0.0017 | 0.401 | -0.0003 | 0.880 | NO RELIABLE SIGNAL |
| def_two_high_pre_z -> deep_rate_resid | -0.0046 | <0.001 | 0.004 | -0.0035 | 0.229 | -0.0076 | 0.009 | MODERATE SIGNAL |
| def_two_high_pre_z -> cmp_rate_resid | +0.0000 | 0.982 | 0.995 | +0.0048 | 0.249 | +0.0019 | 0.627 | NO RELIABLE SIGNAL |
| def_two_high_pre_z -> int_rate_resid | -0.0003 | 0.590 | 0.774 | +0.0018 | 0.132 | +0.0011 | 0.352 | NO RELIABLE SIGNAL |
| def_two_high_pre_z -> scramble_rate_resid | +0.0006 | 0.432 | 0.727 | +0.0016 | 0.402 | +0.0012 | 0.550 | NO RELIABLE SIGNAL |
| def_two_high_pre_z -> WR1 target share | -0.0016 | 0.367 | 0.660 | +0.0001 | 0.983 | -0.0049 | 0.223 | NO RELIABLE SIGNAL |
| def_two_high_pre_z -> WR2 target share | -0.0030 | 0.069 | 0.274 | -0.0102 | 0.004 | -0.0008 | 0.824 | WEAK SIGNAL |
| def_two_high_pre_z -> WR3+ target share | +0.0011 | 0.569 | 0.774 | +0.0013 | 0.752 | -0.0033 | 0.384 | NO RELIABLE SIGNAL |
| def_two_high_pre_z -> TE1 target share | +0.0010 | 0.481 | 0.740 | -0.0008 | 0.818 | +0.0035 | 0.263 | NO RELIABLE SIGNAL |
| def_two_high_pre_z -> TE2+ target share | +0.0007 | 0.581 | 0.774 | -0.0001 | 0.975 | +0.0032 | 0.270 | NO RELIABLE SIGNAL |
| def_two_high_pre_z -> RB1 target share | -0.0001 | 0.928 | 0.987 | +0.0079 | 0.004 | -0.0018 | 0.523 | NO RELIABLE SIGNAL |
| def_two_high_pre_z -> RB2 target share | +0.0012 | 0.230 | 0.510 | +0.0018 | 0.394 | +0.0034 | 0.105 | NO RELIABLE SIGNAL |
| top_cb_z -> WR1 tgt_resid | +0.0052 | 0.004 | 0.031 | -0.0034 | 0.400 | +0.0060 | 0.132 | WEAK SIGNAL |
| top_cb_z -> WR1 yds_resid_per_tgt | +0.0653 | 0.477 | 0.740 | -0.0323 | 0.860 | +0.1052 | 0.580 | NO RELIABLE SIGNAL |
| team_cov_z -> WR1 tgt_resid | +0.0011 | 0.557 | 0.774 | -0.0059 | 0.139 | -0.0075 | 0.062 | NO RELIABLE SIGNAL |
| team_cov_z -> WR1 yds_resid_per_tgt | +0.2388 | 0.009 | 0.065 | -0.0971 | 0.602 | +0.0375 | 0.844 | WEAK SIGNAL |

All-signals-agree subgroup: defense's prior-games WR1 estimate is suppressive AND best CB ≥ 0.5 SD better than average AND pressure ≥ 0.5 SD above AND man rate ≥ 0.5 SD above.

| Split | Games | WR1 tgt-share resid (agree) | (other) | WR1 yds/tgt resid (agree) | (other) | WR1 DK (agree) | (other) |
|---|---|---|---|---|---|---|---|
| train | 51 | -0.0112 | +0.0065 | -0.22 | +0.07 | 13.09 | 14.40 |
| validation | 4 | +0.0978 | +0.0153 | +0.82 | -0.09 | 19.25 | 14.18 |
| test | 9 | -0.0132 | +0.0071 | -1.69 | -0.07 | 15.14 | 13.25 |

## K. Vegas / game-script interaction findings

| Predictor → outcome | Train | p | q | 2024 | p | 2025 | p | Category |
|---|---|---|---|---|---|---|---|---|
| off_favored_by -> WR1 target share | +0.0005 | 0.063 | 0.274 | -0.0008 | 0.229 | +0.0001 | 0.840 | NO RELIABLE SIGNAL |
| off_favored_by -> WR2 target share | -0.0002 | 0.500 | 0.744 | +0.0004 | 0.535 | -0.0008 | 0.160 | NO RELIABLE SIGNAL |
| off_favored_by -> WR3+ target share | -0.0003 | 0.310 | 0.591 | -0.0002 | 0.766 | -0.0010 | 0.087 | NO RELIABLE SIGNAL |
| off_favored_by -> TE1 target share | +0.0002 | 0.260 | 0.547 | +0.0003 | 0.651 | +0.0009 | 0.067 | NO RELIABLE SIGNAL |
| off_favored_by -> TE2+ target share | -0.0002 | 0.351 | 0.650 | +0.0004 | 0.424 | +0.0008 | 0.077 | NO RELIABLE SIGNAL |
| off_favored_by -> RB1 target share | +0.0000 | 0.808 | 0.925 | +0.0005 | 0.315 | +0.0001 | 0.798 | NO RELIABLE SIGNAL |
| off_favored_by -> RB2 target share | -0.0000 | 0.940 | 0.987 | +0.0000 | 0.939 | +0.0001 | 0.865 | NO RELIABLE SIGNAL |
| off_favored_by -> RB1 carry share | -0.0000 | 0.988 | 0.995 | +0.0018 | 0.173 | +0.0031 | 0.010 | NO RELIABLE SIGNAL |
| off_favored_by -> RB2 carry share | +0.0009 | 0.070 | 0.274 | -0.0009 | 0.445 | -0.0020 | 0.071 | NO RELIABLE SIGNAL |
| off_favored_by -> QB carry share | -0.0005 | 0.013 | 0.068 | -0.0012 | 0.014 | -0.0011 | 0.003 | WEAK SIGNAL |
| total_line -> WR1 target share | +0.0007 | 0.090 | 0.334 | +0.0006 | 0.570 | +0.0009 | 0.361 | NO RELIABLE SIGNAL |
| total_line -> WR2 target share | -0.0005 | 0.138 | 0.394 | -0.0004 | 0.659 | -0.0007 | 0.428 | NO RELIABLE SIGNAL |
| total_line -> WR3+ target share | -0.0000 | 0.936 | 0.987 | -0.0013 | 0.239 | -0.0019 | 0.032 | WEAK SIGNAL |
| total_line -> TE1 target share | +0.0001 | 0.865 | 0.973 | +0.0001 | 0.864 | +0.0007 | 0.368 | NO RELIABLE SIGNAL |
| total_line -> TE2+ target share | -0.0002 | 0.543 | 0.774 | -0.0001 | 0.874 | +0.0009 | 0.225 | NO RELIABLE SIGNAL |
| total_line -> RB1 target share | +0.0001 | 0.802 | 0.925 | +0.0005 | 0.477 | +0.0009 | 0.193 | NO RELIABLE SIGNAL |
| total_line -> RB2 target share | -0.0004 | 0.122 | 0.365 | +0.0004 | 0.422 | -0.0005 | 0.356 | NO RELIABLE SIGNAL |
| total_line -> RB1 carry share | +0.0011 | 0.185 | 0.465 | +0.0053 | 0.004 | +0.0017 | 0.368 | WEAK SIGNAL |
| total_line -> RB2 carry share | -0.0010 | 0.160 | 0.439 | -0.0060 | <0.001 | -0.0004 | 0.792 | WEAK SIGNAL |
| total_line -> QB carry share | -0.0002 | 0.394 | 0.690 | -0.0001 | 0.874 | -0.0000 | 0.950 | NO RELIABLE SIGNAL |

Consistency of a defense's group-level effect across conditions (same defense-season, split by condition; r across defense-seasons):

| Condition | WR r | TE r | RB r |
|---|---|---|---|
| favored vs underdog | 0.198 | 0.125 | 0.073 |
| home vs away | 0.079 | 0.081 | 0.103 |
| high vs low total | 0.065 | 0.070 | 0.134 |

Per-role spread/total terms fit on 2019–2023, 2025 test target-share MAE: no defense 0.04374, Vegas terms only 0.04371, role model 0.04373, role + Vegas (D) 0.04370. Carry shares: no defense 0.04123, Vegas only 0.04127, role + Vegas 0.04127.

## L. Statistical results

Train seasons [2019, 2020, 2021, 2022, 2023]. τ = true between-defense SD beyond sampling noise (method of moments over defense-seasons); σ = game-to-game SD; permutation p shuffles which defense faced which game within a season (300 permutations, minimum p ≈ 0.003); reliability = share of a 17-game average that is real signal; optimal k = σ²/τ² games of shrinkage; split-half = odd vs even weeks; YoY = season N vs N+1.

| Family | Metric | τ | σ | Perm p | Rel. 17g | Opt. k | Split-half r | p | YoY r | p | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Role target share | WR1 | 0.0073 | 0.092 | 0.153 | 0.10 | 157 | 0.142 | 0.073 | -0.037 | 0.675 | NO RELIABLE SIGNAL |
| Role target share | WR2 | 0.0071 | 0.085 | 0.123 | 0.11 | 140 | 0.113 | 0.157 | 0.082 | 0.357 | NO RELIABLE SIGNAL |
| Role target share | WR3+ | 0.0105 | 0.096 | 0.086 | 0.17 | 83 | 0.213 | 0.007 | 0.035 | 0.699 | WEAK SIGNAL |
| Role target share | TE1 | 0.0050 | 0.072 | 0.186 | 0.08 | 208 | 0.025 | 0.753 | -0.007 | 0.941 | NO RELIABLE SIGNAL |
| Role target share | TE2+ | 0.0068 | 0.061 | 0.043 | 0.18 | 79 | 0.151 | 0.056 | 0.098 | 0.270 | WEAK SIGNAL |
| Role target share | RB1 | 0.0061 | 0.066 | 0.096 | 0.13 | 115 | 0.111 | 0.162 | 0.126 | 0.157 | WEAK SIGNAL |
| Role target share | RB2 | 0.0011 | 0.054 | 0.419 | 0.01 | 2496 | -0.002 | 0.978 | 0.049 | 0.587 | NO RELIABLE SIGNAL |
| Role target share | RB3+ | 0.0048 | 0.042 | 0.037 | 0.18 | 75 | 0.038 | 0.637 | -0.015 | 0.869 | WEAK SIGNAL |
| Group target share | WR | 0.0173 | 0.107 | 0.003 | 0.31 | 38 | 0.210 | 0.008 | 0.196 | 0.027 | STRONG REPEATABLE SIGNAL |
| Group target share | TE | 0.0121 | 0.088 | 0.017 | 0.25 | 52 | 0.221 | 0.005 | 0.053 | 0.554 | MODERATE SIGNAL |
| Group target share | RB | 0.0130 | 0.087 | 0.010 | 0.28 | 45 | 0.221 | 0.005 | 0.203 | 0.022 | MODERATE SIGNAL |
| Within-group share | WR1 | 0.0040 | 0.146 | 0.412 | 0.01 | 1309 | 0.099 | 0.213 | -0.002 | 0.982 | NO RELIABLE SIGNAL |
| Within-group share | WR2 | 0.0130 | 0.135 | 0.116 | 0.14 | 109 | 0.114 | 0.151 | 0.050 | 0.573 | NO RELIABLE SIGNAL |
| Within-group share | WR3+ | 0.0087 | 0.150 | 0.478 | 0.05 | 297 | 0.178 | 0.024 | 0.001 | 0.987 | WEAK SIGNAL |
| Within-group share | TE1 | 0.0229 | 0.238 | 0.150 | 0.14 | 109 | 0.006 | 0.937 | 0.008 | 0.928 | NO RELIABLE SIGNAL |
| Within-group share | RB1 | 0.0184 | 0.256 | 0.146 | 0.08 | 194 | 0.033 | 0.680 | -0.116 | 0.191 | NO RELIABLE SIGNAL |
| Within-group share | RB2 | 0.0000 | 0.248 | 0.595 | 0.00 | ∞ | -0.023 | 0.776 | -0.100 | 0.260 | NO RELIABLE SIGNAL |
| Carry share | RB1 | 0.0086 | 0.185 | 0.422 | 0.04 | 457 | -0.030 | 0.704 | -0.031 | 0.730 | NO RELIABLE SIGNAL |
| Carry share | RB2 | 0.0071 | 0.165 | 0.379 | 0.03 | 532 | 0.049 | 0.540 | 0.100 | 0.262 | NO RELIABLE SIGNAL |
| Carry share | QB | 0.0059 | 0.061 | 0.140 | 0.14 | 108 | 0.089 | 0.261 | 0.083 | 0.355 | NO RELIABLE SIGNAL |
| Passing environment (positive controls) | pressure_rate | 0.0180 | 0.100 | 0.003 | 0.36 | 31 | 0.260 | <0.001 | 0.085 | 0.343 | MODERATE SIGNAL |
| Passing environment (positive controls) | sack_rate | 0.0089 | 0.046 | 0.003 | 0.39 | 26 | 0.326 | <0.001 | 0.063 | 0.478 | MODERATE SIGNAL |
| Passing environment (positive controls) | adot | 0.5125 | 2.133 | 0.003 | 0.50 | 17 | 0.428 | <0.001 | 0.175 | 0.049 | STRONG REPEATABLE SIGNAL |
| Passing environment (positive controls) | deep_rate | 0.0122 | 0.067 | 0.003 | 0.36 | 30 | 0.256 | 0.001 | 0.161 | 0.069 | MODERATE SIGNAL |
| Passing environment (positive controls) | cmp_rate | 0.0220 | 0.095 | 0.003 | 0.48 | 19 | 0.337 | <0.001 | 0.225 | 0.011 | STRONG REPEATABLE SIGNAL |
| Passing environment (positive controls) | int_rate | 0.0041 | 0.029 | 0.007 | 0.25 | 50 | 0.201 | 0.011 | 0.252 | 0.004 | MODERATE SIGNAL |
| Passing environment (positive controls) | scramble_rate | 0.0000 | 0.040 | 0.415 | 0.00 | ∞ | -0.076 | 0.338 | 0.199 | 0.024 | NO RELIABLE SIGNAL |
| Passing environment (positive controls) | man_rate | 0.0871 | 0.135 | 0.003 | 0.88 | 2 | 0.812 | <0.001 | 0.464 | <0.001 | STRONG REPEATABLE SIGNAL |
| Passing environment (positive controls) | two_high_rate | 0.0813 | 0.135 | 0.003 | 0.86 | 3 | 0.772 | <0.001 | 0.462 | <0.001 | STRONG REPEATABLE SIGNAL |

## M. Out-of-sample results

Shrinkage and window chosen on train only. Target-share models chose window 8 / k 128 (role) and window 34 / k 128 (generic); carry-share models {'window': 8, 'k': 128} / {'window': 8, 'k': 16}. Grid: windows [8, 17, 34], k [0, 2, 4, 8, 16, 32, 64, 128].

*Target-share prediction, calibrated models, mean abs error per role (lower is better)*

| Split | No defense (A) | Generic DvP | Role (B) | Vegas terms, no defense | Role + Vegas (D) |
|---|---|---|---|---|---|
| train | 0.04398 | 0.04397 | 0.04397 | 0.04395 | 0.04395 |
| validation | 0.04422 | 0.04418 | 0.04419 | 0.04423 | 0.04420 |
| test | 0.04374 | 0.04375 | 0.04373 | 0.04371 | 0.04370 |

*Same, WITHOUT league calibration (defense estimates absorb league-wide expectation bias)*

| Split | No defense (A) | Generic DvP | Role (B) | Vegas terms, no defense | Role + Vegas (D) |
|---|---|---|---|---|---|
| train | 0.04443 | 0.04440 | 0.04429 | — | — |
| validation | 0.04478 | 0.04472 | 0.04457 | — | — |
| test | 0.04401 | 0.04404 | 0.04392 | — | — |

*Uncalibrated paired differences, for comparison*

| Split | Comparison | Mean diff | 95% CI | p | Team-games |
|---|---|---|---|---|---|
| train | generic vs no-defense | -0.00027 | -0.00057 to +0.00004 | 0.081 | 2654 |
| train | role vs no-defense | -0.00135 | -0.00195 to -0.00083 | <0.001 | 2654 |
| train | role vs generic | -0.00108 | -0.00159 to -0.00061 | <0.001 | 2654 |
| validation | generic vs no-defense | -0.00063 | -0.00156 to +0.00019 | 0.149 | 544 |
| validation | role vs no-defense | -0.00207 | -0.00339 to -0.00062 | 0.005 | 544 |
| validation | role vs generic | -0.00144 | -0.00257 to -0.00016 | 0.020 | 544 |
| test | generic vs no-defense | +0.00031 | -0.00044 to +0.00104 | 0.432 | 544 |
| test | role vs no-defense | -0.00088 | -0.00217 to +0.00049 | 0.193 | 544 |
| test | role vs generic | -0.00119 | -0.00241 to -0.00015 | 0.056 | 544 |

*Target-share paired differences per team-game (sum over roles; negative = second model better; CI resamples team-games)*

| Split | Comparison | Mean diff | 95% CI | p | Team-games |
|---|---|---|---|---|---|
| train | generic vs no-defense | -0.00015 | -0.00033 to +0.00002 | 0.098 | 2654 |
| train | role vs no-defense | -0.00008 | -0.00029 to +0.00010 | 0.368 | 2654 |
| train | role vs generic | +0.00007 | -0.00014 to +0.00028 | 0.514 | 2654 |
| validation | generic vs no-defense | -0.00039 | -0.00090 to +0.00011 | 0.151 | 544 |
| validation | role vs no-defense | -0.00029 | -0.00066 to +0.00010 | 0.166 | 544 |
| validation | role vs generic | +0.00010 | -0.00047 to +0.00069 | 0.733 | 544 |
| test | generic vs no-defense | +0.00007 | -0.00042 to +0.00056 | 0.773 | 544 |
| test | role vs no-defense | -0.00016 | -0.00051 to +0.00023 | 0.407 | 544 |
| test | role vs generic | -0.00023 | -0.00071 to +0.00027 | 0.389 | 544 |

*Target-share mean abs error by role*

| Role | 2024 none | 2024 generic | 2024 role | 2025 none | 2025 generic | 2025 role |
|---|---|---|---|---|---|---|
| WR1 | 0.0733 | 0.0732 | 0.0733 | 0.0755 | 0.0757 | 0.0755 |
| WR2 | 0.0674 | 0.0673 | 0.0674 | 0.0667 | 0.0667 | 0.0666 |
| WR3+ | 0.0776 | 0.0775 | 0.0776 | 0.0710 | 0.0709 | 0.0710 |
| TE1 | 0.0593 | 0.0593 | 0.0592 | 0.0569 | 0.0569 | 0.0568 |
| TE2+ | 0.0481 | 0.0481 | 0.0482 | 0.0527 | 0.0525 | 0.0526 |
| RB1 | 0.0517 | 0.0516 | 0.0516 | 0.0504 | 0.0503 | 0.0504 |
| RB2 | 0.0384 | 0.0384 | 0.0383 | 0.0369 | 0.0369 | 0.0368 |
| RB3+ | 0.0223 | 0.0223 | 0.0223 | 0.0221 | 0.0221 | 0.0221 |

## N. DFS projection comparison

Player level (WR/TE/RB with ≥ 4 prior games). Every model shares the same team-volume and player-efficiency inputs; only the defense terms differ. Model key: trailing: reference: player's trailing DK points per game (no opportunity model); none_uncal: reference: opportunity model without league calibration; role_uncal: reference: role model without league calibration (shows the calibration artifact); none: MODEL A - calibrated opportunity model, offense/player expectation only, no defense; group: generic defense-vs-position (group-level target/carry share adjustment); role: MODEL B - role-level redistribution; role_matchup: MODEL C - B + defense receiving-efficiency (DK per target allowed); script_none: Vegas/game-script role terms only, no defense (isolates D's Vegas part); script_role_matchup: MODEL D - C + Vegas/game-script role terms.

*DK points, validation*

| Pos | Model | n | MAE | RMSE | Bias | Corr | Median AE |
|---|---|---|---|---|---|---|---|
| ALL | trailing | 5460 | 4.267 | 6.217 | +0.014 | 0.630 | 2.782 |
| ALL | none_uncal | 5460 | 4.196 | 6.072 | -0.284 | 0.646 | 2.792 |
| ALL | role_uncal | 5460 | 4.188 | 6.070 | -0.278 | 0.646 | 2.798 |
| ALL | none | 5460 | 4.147 | 6.063 | -0.405 | 0.648 | 2.712 |
| ALL | group | 5460 | 4.145 | 6.062 | -0.407 | 0.648 | 2.714 |
| ALL | role | 5460 | 4.150 | 6.066 | -0.405 | 0.648 | 2.706 |
| ALL | role_matchup | 5460 | 4.157 | 6.066 | -0.373 | 0.647 | 2.734 |
| ALL | script_none | 5460 | 4.146 | 6.059 | -0.405 | 0.649 | 2.708 |
| ALL | script_role_matchup | 5460 | 4.156 | 6.062 | -0.373 | 0.648 | 2.714 |
| WR | trailing | 2441 | 4.722 | 6.814 | +0.001 | 0.585 | 3.117 |
| WR | none_uncal | 2441 | 4.682 | 6.711 | -0.415 | 0.594 | 3.128 |
| WR | role_uncal | 2441 | 4.678 | 6.710 | -0.404 | 0.594 | 3.128 |
| WR | none | 2441 | 4.645 | 6.702 | -0.505 | 0.596 | 3.051 |
| WR | group | 2441 | 4.642 | 6.701 | -0.510 | 0.597 | 3.032 |
| WR | role | 2441 | 4.649 | 6.703 | -0.502 | 0.596 | 3.046 |
| WR | role_matchup | 2441 | 4.664 | 6.711 | -0.457 | 0.594 | 3.063 |
| WR | script_none | 2441 | 4.646 | 6.699 | -0.503 | 0.597 | 3.032 |
| WR | script_role_matchup | 2441 | 4.664 | 6.708 | -0.455 | 0.595 | 3.049 |
| TE | trailing | 1528 | 3.154 | 4.702 | -0.008 | 0.617 | 2.000 |
| TE | none_uncal | 1528 | 3.139 | 4.617 | -0.097 | 0.631 | 2.006 |
| TE | role_uncal | 1528 | 3.124 | 4.614 | -0.096 | 0.631 | 2.016 |
| TE | none | 1528 | 3.078 | 4.602 | -0.190 | 0.634 | 1.925 |
| TE | group | 1528 | 3.079 | 4.602 | -0.186 | 0.634 | 1.925 |
| TE | role | 1528 | 3.080 | 4.603 | -0.187 | 0.634 | 1.930 |
| TE | role_matchup | 1528 | 3.083 | 4.598 | -0.164 | 0.635 | 1.903 |
| TE | script_none | 1528 | 3.076 | 4.601 | -0.193 | 0.634 | 1.925 |
| TE | script_role_matchup | 1528 | 3.081 | 4.597 | -0.167 | 0.635 | 1.902 |
| RB | trailing | 1491 | 4.661 | 6.549 | +0.056 | 0.649 | 3.213 |
| RB | none_uncal | 1491 | 4.483 | 6.280 | -0.261 | 0.678 | 3.164 |
| RB | role_uncal | 1491 | 4.475 | 6.277 | -0.260 | 0.678 | 3.137 |
| RB | none | 1491 | 4.428 | 6.276 | -0.463 | 0.680 | 3.063 |
| RB | group | 1491 | 4.424 | 6.272 | -0.464 | 0.680 | 3.034 |
| RB | role | 1491 | 4.429 | 6.281 | -0.470 | 0.679 | 3.052 |
| RB | role_matchup | 1491 | 4.429 | 6.273 | -0.450 | 0.680 | 3.042 |
| RB | script_none | 1491 | 4.426 | 6.267 | -0.463 | 0.681 | 3.083 |
| RB | script_role_matchup | 1491 | 4.426 | 6.264 | -0.450 | 0.681 | 3.084 |

*DK points, test*

| Pos | Model | n | MAE | RMSE | Bias | Corr | Median AE |
|---|---|---|---|---|---|---|---|
| ALL | trailing | 5435 | 4.251 | 6.170 | +0.220 | 0.619 | 2.838 |
| ALL | none_uncal | 5435 | 4.147 | 5.963 | -0.017 | 0.637 | 2.815 |
| ALL | role_uncal | 5435 | 4.140 | 5.963 | -0.013 | 0.638 | 2.791 |
| ALL | none | 5435 | 4.106 | 5.958 | -0.141 | 0.639 | 2.726 |
| ALL | group | 5435 | 4.106 | 5.959 | -0.140 | 0.638 | 2.727 |
| ALL | role | 5435 | 4.106 | 5.958 | -0.140 | 0.639 | 2.713 |
| ALL | role_matchup | 5435 | 4.105 | 5.948 | -0.096 | 0.640 | 2.728 |
| ALL | script_none | 5435 | 4.104 | 5.956 | -0.140 | 0.639 | 2.723 |
| ALL | script_role_matchup | 5435 | 4.103 | 5.946 | -0.096 | 0.640 | 2.713 |
| WR | trailing | 2422 | 4.612 | 6.463 | +0.398 | 0.582 | 3.150 |
| WR | none_uncal | 2422 | 4.511 | 6.257 | +0.061 | 0.596 | 3.134 |
| WR | role_uncal | 2422 | 4.506 | 6.261 | +0.083 | 0.596 | 3.117 |
| WR | none | 2422 | 4.479 | 6.252 | -0.030 | 0.597 | 3.066 |
| WR | group | 2422 | 4.478 | 6.254 | -0.029 | 0.597 | 3.051 |
| WR | role | 2422 | 4.479 | 6.253 | -0.028 | 0.597 | 3.034 |
| WR | role_matchup | 2422 | 4.480 | 6.243 | +0.030 | 0.599 | 3.037 |
| WR | script_none | 2422 | 4.477 | 6.249 | -0.027 | 0.597 | 3.050 |
| WR | script_role_matchup | 2422 | 4.478 | 6.241 | +0.033 | 0.599 | 3.040 |
| TE | trailing | 1553 | 3.251 | 4.972 | +0.039 | 0.596 | 1.925 |
| TE | none_uncal | 1553 | 3.233 | 4.875 | -0.052 | 0.604 | 1.995 |
| TE | role_uncal | 1553 | 3.221 | 4.867 | -0.056 | 0.606 | 1.994 |
| TE | none | 1553 | 3.185 | 4.865 | -0.150 | 0.607 | 1.903 |
| TE | group | 1553 | 3.183 | 4.859 | -0.149 | 0.609 | 1.907 |
| TE | role | 1553 | 3.184 | 4.863 | -0.152 | 0.608 | 1.908 |
| TE | role_matchup | 1553 | 3.181 | 4.859 | -0.116 | 0.609 | 1.911 |
| TE | script_none | 1553 | 3.183 | 4.864 | -0.153 | 0.608 | 1.915 |
| TE | script_role_matchup | 1553 | 3.179 | 4.859 | -0.119 | 0.609 | 1.916 |
| RB | trailing | 1460 | 4.715 | 6.791 | +0.116 | 0.637 | 3.250 |
| RB | none_uncal | 1460 | 4.516 | 6.492 | -0.109 | 0.666 | 3.133 |
| RB | role_uncal | 1460 | 4.510 | 6.491 | -0.126 | 0.666 | 3.124 |
| RB | none | 1460 | 4.466 | 6.492 | -0.315 | 0.667 | 3.052 |
| RB | group | 1460 | 4.470 | 6.497 | -0.316 | 0.667 | 3.053 |
| RB | role | 1460 | 4.468 | 6.491 | -0.313 | 0.667 | 3.070 |
| RB | role_matchup | 1460 | 4.465 | 6.475 | -0.284 | 0.669 | 3.083 |
| RB | script_none | 1460 | 4.465 | 6.490 | -0.315 | 0.667 | 3.048 |
| RB | script_role_matchup | 1460 | 4.464 | 6.473 | -0.284 | 0.669 | 3.070 |

*Component errors, all positions, test*

| Model | Targets MAE | Receptions MAE | Rec yards MAE | Carries MAE | TDs MAE |
|---|---|---|---|---|---|
| trailing | 1.591 | 1.220 | 15.65 | 1.003 | 0.296 |
| none_uncal | 1.572 | 1.208 | 15.58 | 0.950 | 0.297 |
| role_uncal | 1.569 | 1.206 | 15.54 | 0.949 | 0.297 |
| none | 1.566 | 1.204 | 15.49 | 0.949 | 0.296 |
| group | 1.566 | 1.203 | 15.49 | 0.948 | 0.296 |
| role | 1.566 | 1.203 | 15.49 | 0.949 | 0.296 |
| role_matchup | 1.566 | 1.203 | 15.49 | 0.949 | 0.296 |
| script_none | 1.565 | 1.203 | 15.49 | 0.951 | 0.296 |
| script_role_matchup | 1.565 | 1.203 | 15.49 | 0.951 | 0.296 |

| Split | Model vs no-defense | Mean abs-error diff (DK pts) | 95% CI (games resampled) | p | n |
|---|---|---|---|---|---|
| train | trailing | +0.1507 | +0.1277 to +0.1720 | <0.001 | 26517 |
| train | none_uncal | +0.0437 | +0.0403 to +0.0473 | <0.001 | 26517 |
| train | role_uncal | +0.0359 | +0.0320 to +0.0396 | <0.001 | 26517 |
| train | group | -0.0006 | -0.0017 to +0.0004 | 0.228 | 26517 |
| train | role | -0.0006 | -0.0015 to +0.0004 | 0.240 | 26517 |
| train | role_matchup | -0.0042 | -0.0073 to -0.0008 | 0.009 | 26517 |
| train | script_none | -0.0027 | -0.0042 to -0.0013 | <0.001 | 26517 |
| train | script_role_matchup | -0.0066 | -0.0103 to -0.0030 | <0.001 | 26517 |
| validation | trailing | +0.1192 | +0.0674 to +0.1677 | <0.001 | 5460 |
| validation | none_uncal | +0.0487 | +0.0420 to +0.0559 | <0.001 | 5460 |
| validation | role_uncal | +0.0403 | +0.0331 to +0.0477 | <0.001 | 5460 |
| validation | group | -0.0022 | -0.0052 to +0.0005 | 0.112 | 5460 |
| validation | role | +0.0026 | +0.0007 to +0.0046 | 0.007 | 5460 |
| validation | role_matchup | +0.0101 | +0.0031 to +0.0172 | <0.001 | 5460 |
| validation | script_none | -0.0010 | -0.0034 to +0.0021 | 0.494 | 5460 |
| validation | script_role_matchup | +0.0087 | +0.0018 to +0.0161 | 0.007 | 5460 |
| test | trailing | +0.1452 | +0.1010 to +0.1872 | <0.001 | 5435 |
| test | none_uncal | +0.0418 | +0.0352 to +0.0489 | <0.001 | 5435 |
| test | role_uncal | +0.0344 | +0.0281 to +0.0412 | <0.001 | 5435 |
| test | group | +0.0003 | -0.0020 to +0.0028 | 0.797 | 5435 |
| test | role | +0.0001 | -0.0019 to +0.0021 | 0.892 | 5435 |
| test | role_matchup | -0.0009 | -0.0076 to +0.0062 | 0.775 | 5435 |
| test | script_none | -0.0013 | -0.0042 to +0.0016 | 0.380 | 5435 |
| test | script_role_matchup | -0.0024 | -0.0100 to +0.0054 | 0.518 | 5435 |

*Quantile coverage / pinball loss, test (coverage should equal the quantile; lower pinball is better)*

| Model | P10 | P25 | P50 | P75 | P90 | P95 | P99 |
|---|---|---|---|---|---|---|---|
| trailing | 0.251 / 0.654 | 0.252 / 1.612 | 0.505 / 2.039 | 0.752 / 2.002 | 0.898 / 1.539 | 0.947 / 1.202 | 0.989 / 0.614 |
| none_uncal | 0.251 / 0.654 | 0.252 / 1.613 | 0.503 / 2.050 | 0.752 / 1.891 | 0.906 / 1.301 | 0.953 / 0.916 | 0.988 / 0.378 |
| role_uncal | 0.251 / 0.654 | 0.252 / 1.613 | 0.505 / 2.044 | 0.753 / 1.891 | 0.905 / 1.309 | 0.953 / 0.932 | 0.988 / 0.394 |
| none | 0.251 / 0.654 | 0.252 / 1.612 | 0.506 / 2.033 | 0.752 / 1.893 | 0.904 / 1.333 | 0.952 / 0.964 | 0.988 / 0.422 |
| group | 0.251 / 0.654 | 0.252 / 1.612 | 0.507 / 2.034 | 0.750 / 1.893 | 0.904 / 1.333 | 0.953 / 0.963 | 0.989 / 0.425 |
| role | 0.251 / 0.654 | 0.252 / 1.612 | 0.506 / 2.033 | 0.751 / 1.893 | 0.904 / 1.333 | 0.953 / 0.964 | 0.988 / 0.423 |
| role_matchup | 0.251 / 0.654 | 0.252 / 1.612 | 0.508 / 2.029 | 0.755 / 1.891 | 0.905 / 1.334 | 0.953 / 0.966 | 0.989 / 0.424 |
| script_none | 0.251 / 0.654 | 0.252 / 1.612 | 0.507 / 2.033 | 0.751 / 1.892 | 0.905 / 1.335 | 0.952 / 0.964 | 0.989 / 0.424 |
| script_role_matchup | 0.251 / 0.654 | 0.252 / 1.611 | 0.508 / 2.028 | 0.754 / 1.890 | 0.905 / 1.334 | 0.954 / 0.968 | 0.990 / 0.427 |

## O. False-discovery controls

| Test family | Tests | p < 0.05 | Expected by chance | BH q < 0.10 | BH q < 0.05 |
|---|---|---|---|---|---|
| group target share (per defense-season) | 480 | 55 | 24 | 12 | 8 |
| role target share (per defense-season) | 1280 | 159 | 64 | 35 | 24 |
| Redistribution (suppressed role → other role) | 36 | 0 | 1.8 | 0 | — |
| Predictive slopes (train period) | 63 | 12 | 3.2 | — | 8 |

Evidence categories for predictive patterns: STRONG = train BH q < 0.05 and same-sign p < 0.05 in both 2024 and 2025; MODERATE = train q < 0.05 and one holdout; WEAK = train q < 0.10 or one holdout only; otherwise NO RELIABLE SIGNAL. Out-of-sample model gains are judged on the untouched 2025 test and 2024 validation, not on significance in training.

## P. Recommended candidate features

- Football pattern (not yet a DFS feature): def_pressure_pre_z -> sack_rate_resid — MODERATE SIGNAL; train +0.0034, 2024 +0.0022, 2025 +0.0069
- Football pattern (not yet a DFS feature): def_man_pre_z -> adot_resid — STRONG REPEATABLE SIGNAL; train +0.2939, 2024 +0.3034, 2025 +0.3214
- Football pattern (not yet a DFS feature): def_man_pre_z -> deep_rate_resid — STRONG REPEATABLE SIGNAL; train +0.0078, 2024 +0.0127, 2025 +0.0077
- Football pattern (not yet a DFS feature): def_man_pre_z -> cmp_rate_resid — MODERATE SIGNAL; train -0.0090, 2024 -0.0095, 2025 -0.0070
- Football pattern (not yet a DFS feature): def_two_high_pre_z -> adot_resid — STRONG REPEATABLE SIGNAL; train -0.1547, 2024 -0.1862, 2025 -0.3334
- Football pattern (not yet a DFS feature): def_two_high_pre_z -> deep_rate_resid — MODERATE SIGNAL; train -0.0046, 2024 -0.0035, 2025 -0.0076

## Q. Rejected features

- Role-level target redistribution vs generic defense-vs-position: 2024 +0.0001 (p 0.733), 2025 -0.0002 (p 0.389) share pts per team-game
- Role-level target model vs no defense: 2024 -0.0003 (p 0.166), 2025 -0.0002 (p 0.407) share pts per team-game
- Generic defense-vs-position target shares vs no defense: 2024 -0.0004 (p 0.151), 2025 +0.0001 (p 0.773) share pts per team-game
- Generic defense-vs-position carry shares vs no defense: 2024 -0.0004 (p 0.404), 2025 -0.0006 (p 0.162) share pts per team-game
- RB1/RB2 carry-split role model vs generic: 2024 +0.0006 (p 0.181), 2025 +0.0007 (p 0.149) share pts per team-game
- Generic DvP (DK points): 2024 -0.0022 (p 0.112), 2025 +0.0003 (p 0.797) DK pts per player-game
- Role redistribution, model B (DK points): 2024 +0.0026 (p 0.007), 2025 +0.0001 (p 0.892) DK pts per player-game
- B + matchup efficiency, model C (DK points): 2024 +0.0101 (p <0.001), 2025 -0.0009 (p 0.775) DK pts per player-game
- Vegas/game-script role terms only, no defense (DK points): 2024 -0.0010 (p 0.494), 2025 -0.0013 (p 0.380) DK pts per player-game
- C + Vegas/game script, model D (DK points): 2024 +0.0087 (p 0.007), 2025 -0.0024 (p 0.518) DK pts per player-game
- Generic QB defense-vs-position: 2024 +0.0231 (p 0.286), 2025 -0.0265 (p 0.219) DK pts per QB-game
- QB pressure/coverage model: 2024 +0.0113 (p 0.508), 2025 +0.0190 (p 0.241) DK pts per QB-game
- QB generic + pressure/coverage: 2024 +0.0323 (p 0.243), 2025 -0.0060 (p 0.831) DK pts per QB-game
- def_man_pre_z -> RB1 target share: MODERATE in train + 2024 but reversed sign in 2025 (+0.0034) - not repeatable

## R. Data limitations

- No slot/outside alignment, route counts, or CB-to-receiver assignments exist in public nflverse data. Dropback participation is a route proxy; pass location is not alignment.
- Time snapshots (OPEN / T-24 / T-6 / T-1 / LOCK) cannot be reconstructed: nflverse keeps closing lines and final injury reports only. Every pre-game input here is either strictly prior-game data or closing Vegas lines plus game-day actives, i.e. a LOCK snapshot.
- Man/zone charting changes level between 2022 (≈29% man) and 2023–2024 (42–49%), consistent with a change in charting source; coverage rates are z-scored within season for that reason.
- PFR per-defender coverage is a charter's judgment of the covering defender, not a verified assignment.
- Seven seasons of training data give ~160 defense-seasons; a true role-level defense SD below ~0.5 share points would be hard to detect even if it exists.
- Expected production uses trailing player usage; it does not model mid-week role news. Residuals therefore include role changes the offense made that no defense caused, which adds noise equally to every model.

What would make this stronger: charted alignment (slot/wide) and route data, per-play coverage assignments, timestamped injury and line history, and more seasons of participation charting.

## S. Full list of tested patterns

Slope columns: effect per unit of the predictor (per 1 SD for z-scored predictors, per point for spread/total) in train / 2024 validation / 2025 test. Variance-component rows show τ and permutation p; redistribution rows show the cross-half correlation.

| Pattern | Train effect | p | q | 2024 effect | p | 2025 effect | p | n | Category |
|---|---|---|---|---|---|---|---|---|---|
| def_pressure_pre_z -> adot_resid | -0.04667 | 0.274 | 0.556 | +0.12028 | 0.202 | +0.05379 | 0.584 | 2654 | NO RELIABLE SIGNAL |
| def_pressure_pre_z -> sack_rate_resid | +0.00342 | <0.001 | 0.002 | +0.00222 | 0.277 | +0.00688 | 0.001 | 2654 | MODERATE SIGNAL |
| def_pressure_pre_z -> deep_rate_resid | -0.00163 | 0.217 | 0.507 | +0.00802 | 0.006 | +0.00531 | 0.070 | 2654 | NO RELIABLE SIGNAL |
| def_pressure_pre_z -> cmp_rate_resid | -0.00479 | 0.011 | 0.065 | -0.00351 | 0.404 | -0.01369 | <0.001 | 2654 | WEAK SIGNAL |
| def_pressure_pre_z -> int_rate_resid | +0.00093 | 0.101 | 0.353 | +0.00137 | 0.246 | +0.00017 | 0.889 | 2654 | NO RELIABLE SIGNAL |
| def_pressure_pre_z -> scramble_rate_resid | +0.00024 | 0.757 | 0.900 | -0.00451 | 0.019 | +0.00102 | 0.603 | 2654 | NO RELIABLE SIGNAL |
| def_pressure_pre_z -> WR1 target share | +0.00077 | 0.666 | 0.825 | +0.00453 | 0.252 | +0.00414 | 0.306 | 2654 | NO RELIABLE SIGNAL |
| def_pressure_pre_z -> WR2 target share | -0.00118 | 0.476 | 0.740 | +0.00256 | 0.476 | +0.00361 | 0.302 | 2654 | NO RELIABLE SIGNAL |
| def_pressure_pre_z -> WR3+ target share | -0.00223 | 0.235 | 0.510 | -0.00232 | 0.587 | -0.00035 | 0.927 | 2654 | NO RELIABLE SIGNAL |
| def_pressure_pre_z -> TE1 target share | +0.00224 | 0.108 | 0.356 | -0.00194 | 0.552 | +0.00053 | 0.864 | 2654 | NO RELIABLE SIGNAL |
| def_pressure_pre_z -> TE2+ target share | +0.00012 | 0.920 | 0.987 | -0.00057 | 0.838 | -0.00374 | 0.202 | 2654 | NO RELIABLE SIGNAL |
| def_pressure_pre_z -> RB1 target share | -0.00055 | 0.668 | 0.825 | -0.00165 | 0.553 | -0.00252 | 0.364 | 2654 | NO RELIABLE SIGNAL |
| def_pressure_pre_z -> RB2 target share | +0.00001 | 0.995 | 0.995 | -0.00035 | 0.867 | +0.00052 | 0.803 | 2654 | NO RELIABLE SIGNAL |
| def_man_pre_z -> adot_resid | +0.29389 | <0.001 | <0.001 | +0.30336 | 0.001 | +0.32141 | <0.001 | 2654 | STRONG REPEATABLE SIGNAL |
| def_man_pre_z -> sack_rate_resid | -0.00035 | 0.697 | 0.844 | -0.00182 | 0.373 | +0.00244 | 0.251 | 2654 | NO RELIABLE SIGNAL |
| def_man_pre_z -> deep_rate_resid | +0.00777 | <0.001 | <0.001 | +0.01273 | <0.001 | +0.00766 | 0.009 | 2654 | STRONG REPEATABLE SIGNAL |
| def_man_pre_z -> cmp_rate_resid | -0.00899 | <0.001 | <0.001 | -0.00950 | 0.023 | -0.00700 | 0.078 | 2654 | MODERATE SIGNAL |
| def_man_pre_z -> int_rate_resid | +0.00090 | 0.113 | 0.356 | -0.00042 | 0.725 | -0.00061 | 0.619 | 2654 | NO RELIABLE SIGNAL |
| def_man_pre_z -> scramble_rate_resid | -0.00061 | 0.438 | 0.727 | +0.00494 | 0.010 | +0.00114 | 0.560 | 2654 | NO RELIABLE SIGNAL |
| def_man_pre_z -> WR1 target share | +0.00085 | 0.634 | 0.815 | +0.00467 | 0.239 | +0.00806 | 0.046 | 2654 | WEAK SIGNAL |
| def_man_pre_z -> WR2 target share | +0.00175 | 0.289 | 0.569 | +0.00395 | 0.273 | +0.00337 | 0.336 | 2654 | NO RELIABLE SIGNAL |
| def_man_pre_z -> WR3+ target share | +0.00480 | 0.010 | 0.065 | +0.00387 | 0.366 | -0.00423 | 0.266 | 2654 | WEAK SIGNAL |
| def_man_pre_z -> TE1 target share | -0.00179 | 0.200 | 0.483 | +0.00180 | 0.582 | -0.00453 | 0.145 | 2654 | NO RELIABLE SIGNAL |
| def_man_pre_z -> TE2+ target share | +0.00159 | 0.180 | 0.465 | -0.00210 | 0.452 | -0.00337 | 0.250 | 2654 | NO RELIABLE SIGNAL |
| def_man_pre_z -> RB1 target share | -0.00451 | <0.001 | 0.004 | -0.01031 | <0.001 | +0.00340 | 0.220 | 2654 | MODERATE SIGNAL |
| def_man_pre_z -> RB2 target share | -0.00199 | 0.056 | 0.273 | -0.00167 | 0.428 | -0.00220 | 0.294 | 2654 | NO RELIABLE SIGNAL |
| def_two_high_pre_z -> adot_resid | -0.15472 | <0.001 | 0.003 | -0.18618 | 0.048 | -0.33336 | <0.001 | 2654 | STRONG REPEATABLE SIGNAL |
| def_two_high_pre_z -> sack_rate_resid | +0.00060 | 0.508 | 0.744 | +0.00172 | 0.401 | -0.00032 | 0.880 | 2654 | NO RELIABLE SIGNAL |
| def_two_high_pre_z -> deep_rate_resid | -0.00460 | <0.001 | 0.004 | -0.00350 | 0.229 | -0.00763 | 0.009 | 2654 | MODERATE SIGNAL |
| def_two_high_pre_z -> cmp_rate_resid | +0.00004 | 0.982 | 0.995 | +0.00485 | 0.249 | +0.00194 | 0.627 | 2654 | NO RELIABLE SIGNAL |
| def_two_high_pre_z -> int_rate_resid | -0.00031 | 0.590 | 0.774 | +0.00177 | 0.132 | +0.00114 | 0.352 | 2654 | NO RELIABLE SIGNAL |
| def_two_high_pre_z -> scramble_rate_resid | +0.00062 | 0.432 | 0.727 | +0.00163 | 0.402 | +0.00117 | 0.550 | 2654 | NO RELIABLE SIGNAL |
| def_two_high_pre_z -> WR1 target share | -0.00161 | 0.367 | 0.660 | +0.00008 | 0.983 | -0.00492 | 0.223 | 2654 | NO RELIABLE SIGNAL |
| def_two_high_pre_z -> WR2 target share | -0.00299 | 0.069 | 0.274 | -0.01016 | 0.004 | -0.00078 | 0.824 | 2654 | WEAK SIGNAL |
| def_two_high_pre_z -> WR3+ target share | +0.00107 | 0.569 | 0.774 | +0.00135 | 0.752 | -0.00332 | 0.384 | 2654 | NO RELIABLE SIGNAL |
| def_two_high_pre_z -> TE1 target share | +0.00098 | 0.481 | 0.740 | -0.00075 | 0.818 | +0.00348 | 0.263 | 2654 | NO RELIABLE SIGNAL |
| def_two_high_pre_z -> TE2+ target share | +0.00065 | 0.581 | 0.774 | -0.00009 | 0.975 | +0.00323 | 0.270 | 2654 | NO RELIABLE SIGNAL |
| def_two_high_pre_z -> RB1 target share | -0.00012 | 0.928 | 0.987 | +0.00790 | 0.004 | -0.00177 | 0.523 | 2654 | NO RELIABLE SIGNAL |
| def_two_high_pre_z -> RB2 target share | +0.00125 | 0.230 | 0.510 | +0.00179 | 0.394 | +0.00339 | 0.105 | 2654 | NO RELIABLE SIGNAL |
| top_cb_z -> WR1 tgt_resid | +0.00518 | 0.004 | 0.031 | -0.00336 | 0.400 | +0.00604 | 0.132 | 2631 | WEAK SIGNAL |
| top_cb_z -> WR1 yds_resid_per_tgt | +0.06529 | 0.477 | 0.740 | -0.03227 | 0.860 | +0.10517 | 0.580 | 2502 | NO RELIABLE SIGNAL |
| team_cov_z -> WR1 tgt_resid | +0.00106 | 0.557 | 0.774 | -0.00587 | 0.139 | -0.00754 | 0.062 | 2635 | NO RELIABLE SIGNAL |
| team_cov_z -> WR1 yds_resid_per_tgt | +0.23884 | 0.009 | 0.065 | -0.09710 | 0.602 | +0.03752 | 0.844 | 2506 | WEAK SIGNAL |
| off_favored_by -> WR1 target share | +0.00051 | 0.063 | 0.274 | -0.00083 | 0.229 | +0.00013 | 0.840 | 2638 | NO RELIABLE SIGNAL |
| off_favored_by -> WR2 target share | -0.00017 | 0.500 | 0.744 | +0.00039 | 0.535 | -0.00077 | 0.160 | 2638 | NO RELIABLE SIGNAL |
| off_favored_by -> WR3+ target share | -0.00029 | 0.310 | 0.591 | -0.00022 | 0.766 | -0.00101 | 0.087 | 2638 | NO RELIABLE SIGNAL |
| off_favored_by -> TE1 target share | +0.00024 | 0.260 | 0.547 | +0.00026 | 0.651 | +0.00089 | 0.067 | 2638 | NO RELIABLE SIGNAL |
| off_favored_by -> TE2+ target share | -0.00017 | 0.351 | 0.650 | +0.00039 | 0.424 | +0.00081 | 0.077 | 2638 | NO RELIABLE SIGNAL |
| off_favored_by -> RB1 target share | +0.00005 | 0.808 | 0.925 | +0.00048 | 0.315 | +0.00011 | 0.798 | 2638 | NO RELIABLE SIGNAL |
| off_favored_by -> RB2 target share | -0.00001 | 0.940 | 0.987 | +0.00003 | 0.939 | +0.00006 | 0.865 | 2638 | NO RELIABLE SIGNAL |
| off_favored_by -> RB1 carry share | -0.00001 | 0.988 | 0.995 | +0.00176 | 0.173 | +0.00311 | 0.010 | 2638 | NO RELIABLE SIGNAL |
| off_favored_by -> RB2 carry share | +0.00089 | 0.070 | 0.274 | -0.00090 | 0.445 | -0.00196 | 0.071 | 2638 | NO RELIABLE SIGNAL |
| off_favored_by -> QB carry share | -0.00046 | 0.013 | 0.068 | -0.00117 | 0.014 | -0.00107 | 0.003 | 2638 | WEAK SIGNAL |
| total_line -> WR1 target share | +0.00067 | 0.090 | 0.334 | +0.00056 | 0.570 | +0.00089 | 0.361 | 2638 | NO RELIABLE SIGNAL |
| total_line -> WR2 target share | -0.00054 | 0.138 | 0.394 | -0.00040 | 0.659 | -0.00067 | 0.428 | 2638 | NO RELIABLE SIGNAL |
| total_line -> WR3+ target share | -0.00003 | 0.936 | 0.987 | -0.00125 | 0.239 | -0.00195 | 0.032 | 2638 | WEAK SIGNAL |
| total_line -> TE1 target share | +0.00005 | 0.865 | 0.973 | +0.00014 | 0.864 | +0.00067 | 0.368 | 2638 | NO RELIABLE SIGNAL |
| total_line -> TE2+ target share | -0.00016 | 0.543 | 0.774 | -0.00011 | 0.874 | +0.00085 | 0.225 | 2638 | NO RELIABLE SIGNAL |
| total_line -> RB1 target share | +0.00007 | 0.802 | 0.925 | +0.00049 | 0.477 | +0.00086 | 0.193 | 2638 | NO RELIABLE SIGNAL |
| total_line -> RB2 target share | -0.00036 | 0.122 | 0.365 | +0.00042 | 0.422 | -0.00046 | 0.356 | 2638 | NO RELIABLE SIGNAL |
| total_line -> RB1 carry share | +0.00106 | 0.185 | 0.465 | +0.00529 | 0.004 | +0.00167 | 0.368 | 2638 | WEAK SIGNAL |
| total_line -> RB2 carry share | -0.00100 | 0.160 | 0.439 | -0.00601 | <0.001 | -0.00044 | 0.792 | 2638 | WEAK SIGNAL |
| total_line -> QB carry share | -0.00023 | 0.394 | 0.690 | -0.00011 | 0.874 | -0.00004 | 0.950 | 2638 | NO RELIABLE SIGNAL |
| defense → WR1 target share (variance component) | 0.0073 | 0.153 | — | — | — | — | — | 160 | NO RELIABLE SIGNAL |
| defense → WR2 target share (variance component) | 0.0071 | 0.123 | — | — | — | — | — | 160 | NO RELIABLE SIGNAL |
| defense → WR3+ target share (variance component) | 0.0105 | 0.086 | — | — | — | — | — | 160 | WEAK SIGNAL |
| defense → TE1 target share (variance component) | 0.0050 | 0.186 | — | — | — | — | — | 160 | NO RELIABLE SIGNAL |
| defense → TE2+ target share (variance component) | 0.0068 | 0.043 | — | — | — | — | — | 160 | WEAK SIGNAL |
| defense → RB1 target share (variance component) | 0.0061 | 0.096 | — | — | — | — | — | 160 | WEAK SIGNAL |
| defense → RB2 target share (variance component) | 0.0011 | 0.419 | — | — | — | — | — | 160 | NO RELIABLE SIGNAL |
| defense → RB3+ target share (variance component) | 0.0048 | 0.037 | — | — | — | — | — | 160 | WEAK SIGNAL |
| defense → WR group target share (variance component) | 0.0173 | 0.003 | — | — | — | — | — | 160 | STRONG REPEATABLE SIGNAL |
| defense → TE group target share (variance component) | 0.0121 | 0.017 | — | — | — | — | — | 160 | MODERATE SIGNAL |
| defense → RB group target share (variance component) | 0.0130 | 0.010 | — | — | — | — | — | 160 | MODERATE SIGNAL |
| WR1 suppression → WR2 (conditional share) | +0.030 | 0.710 | — | — | — | — | — | 160 | NO RELIABLE SIGNAL |
| WR1 suppression → WR3+ (conditional share) | +0.006 | 0.945 | — | — | — | — | — | 160 | NO RELIABLE SIGNAL |
| WR1 suppression → TE1 (conditional share) | -0.003 | 0.966 | — | — | — | — | — | 160 | NO RELIABLE SIGNAL |
| WR1 suppression → TE2+ (conditional share) | -0.061 | 0.445 | — | — | — | — | — | 160 | NO RELIABLE SIGNAL |
| WR1 suppression → RB1 (conditional share) | -0.013 | 0.866 | — | — | — | — | — | 160 | NO RELIABLE SIGNAL |
| WR1 suppression → RB2 (conditional share) | +0.009 | 0.906 | — | — | — | — | — | 160 | NO RELIABLE SIGNAL |
| WR1 suppression → RB3+ (conditional share) | +0.002 | 0.982 | — | — | — | — | — | 160 | NO RELIABLE SIGNAL |
| WR1 suppression → QB (conditional share) | +0.108 | 0.173 | — | — | — | — | — | 160 | NO RELIABLE SIGNAL |
| WR1 suppression → OTHER (conditional share) | +0.081 | 0.309 | — | — | — | — | — | 160 | NO RELIABLE SIGNAL |
| WR2 suppression → WR1 (conditional share) | +0.026 | 0.742 | — | — | — | — | — | 160 | NO RELIABLE SIGNAL |
| WR2 suppression → WR3+ (conditional share) | -0.003 | 0.973 | — | — | — | — | — | 160 | NO RELIABLE SIGNAL |
| WR2 suppression → TE1 (conditional share) | +0.055 | 0.490 | — | — | — | — | — | 160 | NO RELIABLE SIGNAL |
| WR2 suppression → TE2+ (conditional share) | -0.041 | 0.605 | — | — | — | — | — | 160 | NO RELIABLE SIGNAL |
| WR2 suppression → RB1 (conditional share) | -0.024 | 0.766 | — | — | — | — | — | 160 | NO RELIABLE SIGNAL |
| WR2 suppression → RB2 (conditional share) | +0.011 | 0.888 | — | — | — | — | — | 160 | NO RELIABLE SIGNAL |
| WR2 suppression → RB3+ (conditional share) | -0.046 | 0.561 | — | — | — | — | — | 160 | NO RELIABLE SIGNAL |
| WR2 suppression → QB (conditional share) | -0.013 | 0.867 | — | — | — | — | — | 160 | NO RELIABLE SIGNAL |
| WR2 suppression → OTHER (conditional share) | -0.034 | 0.673 | — | — | — | — | — | 160 | NO RELIABLE SIGNAL |
| TE1 suppression → WR1 (conditional share) | -0.030 | 0.711 | — | — | — | — | — | 160 | NO RELIABLE SIGNAL |
| TE1 suppression → WR2 (conditional share) | +0.038 | 0.630 | — | — | — | — | — | 160 | NO RELIABLE SIGNAL |
| TE1 suppression → WR3+ (conditional share) | -0.098 | 0.220 | — | — | — | — | — | 160 | NO RELIABLE SIGNAL |
| TE1 suppression → TE2+ (conditional share) | +0.127 | 0.109 | — | — | — | — | — | 160 | NO RELIABLE SIGNAL |
| TE1 suppression → RB1 (conditional share) | +0.040 | 0.620 | — | — | — | — | — | 160 | NO RELIABLE SIGNAL |
| TE1 suppression → RB2 (conditional share) | +0.001 | 0.989 | — | — | — | — | — | 160 | NO RELIABLE SIGNAL |
| TE1 suppression → RB3+ (conditional share) | -0.034 | 0.667 | — | — | — | — | — | 160 | NO RELIABLE SIGNAL |
| TE1 suppression → QB (conditional share) | +0.045 | 0.569 | — | — | — | — | — | 160 | NO RELIABLE SIGNAL |
| TE1 suppression → OTHER (conditional share) | -0.036 | 0.652 | — | — | — | — | — | 160 | NO RELIABLE SIGNAL |
| RB1 suppression → WR1 (conditional share) | -0.014 | 0.863 | — | — | — | — | — | 160 | NO RELIABLE SIGNAL |
| RB1 suppression → WR2 (conditional share) | -0.025 | 0.750 | — | — | — | — | — | 160 | NO RELIABLE SIGNAL |
| RB1 suppression → WR3+ (conditional share) | -0.092 | 0.247 | — | — | — | — | — | 160 | NO RELIABLE SIGNAL |
| RB1 suppression → TE1 (conditional share) | +0.042 | 0.596 | — | — | — | — | — | 160 | NO RELIABLE SIGNAL |
| RB1 suppression → TE2+ (conditional share) | -0.010 | 0.899 | — | — | — | — | — | 160 | NO RELIABLE SIGNAL |
| RB1 suppression → RB2 (conditional share) | +0.072 | 0.369 | — | — | — | — | — | 160 | NO RELIABLE SIGNAL |
| RB1 suppression → RB3+ (conditional share) | +0.136 | 0.087 | — | — | — | — | — | 160 | NO RELIABLE SIGNAL |
| RB1 suppression → QB (conditional share) | -0.023 | 0.770 | — | — | — | — | — | 160 | NO RELIABLE SIGNAL |
| RB1 suppression → OTHER (conditional share) | -0.075 | 0.349 | — | — | — | — | — | 160 | NO RELIABLE SIGNAL |
