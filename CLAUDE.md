# Project conventions

- Never hardcode API keys, database passwords, or secrets in code or committed
  files (including `.env.example`) — use environment variables, with only
  placeholder values in anything checked into git.
- Wrap anything that fetches external data (HTTP requests, API calls) in
  try/except so one failed request doesn't crash the app.
- Add a short comment explaining *why*, not what, for anything non-obvious
  (a workaround, a hidden constraint, a format assumption).
- If unsure about an exact file format or API response shape, check it first
  rather than guessing.
- Commit in small chunks with clear messages so changes are easy to track and
  roll back.

## Standing lineup-generation default

When building a lineup for the user (not a backtest, not a diagnostic
comparison), default to `models.optimizer.generate_simulation_selected_lineup`
(ranked[0], the real simulated-win-rate pick) instead of calling
`generate_lineups` directly for its raw ceiling-sum pick. This is a real,
standing preference the user set deliberately, aware of `models/simulation.py`'s
own disclosed null backtest result (55 weeks, t=0.20 - simulation-based
selection did not beat the deterministic ceiling pick on that sample). Keep
using it as the default regardless; do not revert to the deterministic path
or go quiet about the tradeoff on your own initiative.

What to keep doing, standing: if a future, larger batch of real contest data
(via a real backtest re-run, not a guess) shows this choice is actually
costing real results - not just producing a different lineup - say so
plainly and let the user decide whether to change the default. Never switch
back to the deterministic path silently, and never suppress a negative
finding to avoid revisiting this decision.

## Standing reporting convention: name the file

Whenever reporting exposure/duplicate/verification numbers (or any other
per-export analysis) for a generated file, state the exact filename in the
same message as the numbers - e.g. "these numbers are for
DKUpload_20lineups_v2.csv" - rather than a bare table the reader has to
guess the source of. A real incident this caused: reporting a table without
naming which of two successive exports (19-lineup vs. 20-lineup) it
described, which the user reasonably read as contradicting a later,
correct report on the newer file. This applies every time a new version of
a file is generated, not just DK lineup exports.

## Standing workflow: tag every build by method (optimizer vs simulator)

Every lineup file built for the user is exported through
`data.lineup_tracking.export_grouped_lineups(slate_id, build_id, groups, path)`
with groups named `"optimizer"` (raw `generate_lineups` picks) and
`"simulator"` (`generate_simulation_selected_lineup` picks), never
`write_dk_upload_csv` directly - so each lineup is registered in
`built_lineups` with its method. Tell the user which file holds which group.

When contest standings are uploaded, import them with
`bulk_import_contest_standings` (it scores every registered build on that
slate automatically), then report `compare_build_groups()` - name the
contest and file, give each group's average points and finish percentile,
and say which of the user's entries came from which group (late swaps are
reported as edited). The verdict stays INSUFFICIENT SAMPLE until
`MIN_SLATES_FOR_VERDICT` slates; until then, don't present either method as
better. Once it isn't, this is the "larger batch of real contest data" the
simulator-default section above refers to: report it plainly either way.
Seeded with dk_showdown_nyg_lar_2026_09_21 and dk_sunday_2026_09_27.

Showdown builds (set by the user after PHI@CHI 2026-09-28, where the
2.9%-captained backup QB was the winning captain): pass
`min_captain_per_qb=1` per 10-lineup group so every starting QB captains at
least one lineup in each group, and pick the simulator group's lineups with
`select_with_captain_coverage(ranked, 10, availability_report["captain_coverage"])`
so the simulation ranking can't drop them. Report it in the captain
exposure table; tracking decides whether it helps.

## Standing workflow: spread the portfolio across games and players

Set by the user after 2026-10-04, when 20 lineups used only 3 QBs (Purdy 9,
Lawrence 6, Allen 5), put up to 12 of 20 on the same players, and gave the
#2-ranked game (DAL@HOU, 34-30) 4 of 180 player slots while the contest
leader stacked it. Every multi-lineup build:

1. Rank the slate's games with `models.game_environment.rank_slate_games`
   and show the user the table.
2. Build each 10-lineup group with `models.portfolio.qb_plan` +
   `build_spread_portfolio`: both QBs of each of the top 5 games get one
   stacked lineup (10 different QBs), every player capped at 40% and every
   DST at 30%. Optimizer group: `choose=None`; simulator group: 8
   candidates per QB, keep the best simulated one.
3. Pass the news exclusions as `excluded_player_ids` as before.

Retro test (projections stored at the time + the injury-return fix):
9/27 cashed 6 of 20 vs 4 built, best rank 141 of 89,179; 10/04 cashed 7
vs 1. Spreading alone, without the fix, was roughly neutral (5 and 0) -
the projections matter as much as the spread.
4. Single-entry lineups built the same day are counted with the
   multi-entry file: report the combined exposure, and don't let one
   player's bust sink every entry - flag any player in more than 40% of all
   the user's entries for the day.

Report the per-game slot counts and QB list with the exposure table. Track
it like the other rules: compare_build_groups and finish percentiles
decide whether it helps.

## Standing workflow: single-entry lineups and file naming

Updated 2026-10-05 (the user: "do whatever you think is best"). For a
single-entry contest (or any contest the user enters with one lineup), pick
`models.contest_selection.generate_contest_selected_lineup(slate_id,
paid_share, ...)`'s ranked[0], with the same news exclusions and
(Showdown) `min_captain_per_qb=1` as the main build. `paid_share` is that
contest's paid places / entries. It replaced simulator-01 (win rate among
our own candidates). Simulator-01 had averaged 8 points worse than its
pool across 6 slates; in the 9-slate retro study (stored projections, 60
candidates) its pick finished top 70% in Classic vs 51% for the average
candidate. The cash-rate pick averaged top 43% over all 9 and was never
the worst. No selection score predicted much (Spearman within +/-0.06), so
report it as the least-bad pick, not an edge. The multi-entry simulator
group (best simulated p90 per QB) is unchanged. A different single-entry
pick needs a stated, specific reason (e.g. news after the build), told to
the user. Keep tracking it with `compare_top_pick_to_pool()` and say so
plainly if it underperforms.

Deliver exactly one file per contest, named for the contest, e.g.
`DKUpload_PIT_CLE_PlayAction_FINAL.csv`. When a lineup changes, overwrite
that file instead of adding v2/v3, and delete superseded versions: on
PIT@CLE, three versions per contest led to the Pylon v2 lineup going into
Play-Action, where the Play-Action file would have cashed by 41 places.

After standings are imported, report `compare_top_pick_to_pool()` next to
`compare_build_groups()`; it stays INSUFFICIENT SAMPLE until
`MIN_SLATES_FOR_VERDICT` slates.

## Standing workflow: news research before building and before lock

Set by the user after PHI@CHI 2026-09-28: the data feeds still had Tyson
Bagent as Chicago's QB, while Case Keenum's start had been reported the day
before. It was only caught after the lineups were built, when 8 of 20
lineups plus the booster lineup had the backup QB.

1. Before building any lineup, search the web for the slate's news:
   injuries and designations (Out/Doubtful/Questionable), starting QBs,
   suspensions, role changes, and the Vegas spread/total. Wherever news
   contradicts the feeds (the availability gate, depth chart, projections),
   correct the build for it - excluded players, promoted starters,
   projections - and tell the user each correction and its source.
2. After building, check again once inactives are posted (about 90 minutes
   before kickoff). Confirm every rostered player in every lineup is active
   and in his expected role, and give the user the exact lineup rows and the
   swaps needed. If the user asks before inactives are out, say when to
   check back instead of implying the lineups are final.

Both checks run on every slate whether or not the user asks a follow-up.
Cite the sources. If a search is blocked or comes back empty, say so rather
than assuming no news.

If the inactive list can't be confirmed from here (ATL@NO 2026-10-05: ESPN,
CBS, SI, Yahoo, FantasyPros and the DK/Sleeper APIs were all blocked by the
network policy, and Questionable Noah Fant turned out inactive and scored 0
in 6 of 20 lineups), don't leave it at "probably fine": at build time keep
any Questionable player who didn't practice fully to at most 2 of 20
lineups, and before lock tell the user plainly that inactives are
unconfirmed, give the swap rows, and ask them to check DraftKings' red
"O"/"INACT" tag before kickoff.

When news rules out a player, move his production with
`models.absence_redistribution.redistribute_absence(slate_id, name)`, not a
hand-made split (added 2026-10-06). Measured on 2023-25: a lead RB's backup
gains a median 34% of his points and the third back 15%; a lead WR's absence
lifts the other WRs/TEs by about nothing, so it moves nothing. The two
hand-made splits it replaced both missed: Coker (60% to pass catchers,
DET@CAR) and Etienne (Kamara got 33%, then scored 22.8 vs our 11.9).

## SaberSim: one-time reference, our model stays the base

The user had a SaberSim subscription for one slate only (cancelled
2026-10-01) and sent one export, NFL_2026-09-28-815pm_DK_Mon-Thu.csv
(PHI@CHI + PIT@CLE). Our own model remains the base projection for every
slate going forward; do not wait for or ask for SaberSim files. If another
export ever arrives, `data.sabersim_import.apply_sabersim(slate_id, path)`
still works (it snapshots our projections as source `model` first).

What the files were used for: PIT@CLE 2026-10-01 and the Thu-Mon
dk_thu_mon_2026_10_01 slate projections were rebased on them (kickers kept
ours on Showdown), and both sources are stored in projection_sources.
After standings are imported for those two slates, report
`projection_accuracy(slate_id)` (model vs sabersim, same players). PHI@CHI:
model MAE 4.24 vs SaberSim 4.50 on 19 players; PIT@CLE (2026-10-01):
model 4.23 vs SaberSim 5.00 on 18 players (both too low, bias -3.8 / -4.1).
Shape comparison on 36
players: medians within about a point at every position; SaberSim's RB
downside is much narrower (p10 at ~31% below median vs our ~97%) and its TE
upside wider. Checked against history 2026-10-02
(`run_quantile_coverage_backtest`, 2023-26): our RB floor is right - 11% of
real scores for RBs projected 8+ fell below our p10 (target 10%) vs 34%
below a SaberSim-style floor - so keep ours. Their wider TE upside is
supported: 17% of real TE scores beat our p90 (target 10%).

## Standing workflow: weekly player baseline update

Set by the user 2026-10-01: SaberSim exports seeded a running baseline per
player (player_baselines). The full-week export
NFL_2026-10-01-815pm_DK_Thu-Mon.csv (all 32 teams, week 4) seeded 422 of
its 428 projected players, replacing the earlier 4-team seed, with each
player's stat projections and week context (salary, projected ownership,
injury status, team/game implied points, 95th/99th percentile) stored in
usage. The 6 without a baseline are rookies with no NFL games (Tanner Arkin,
Gavin Bartholomew, Max Klare, Mark Redman, Patrick Herbert) and Drew
Ogletree (Andrew in nflverse). From then on each player's numbers are
updated from real results. If another SaberSim export arrives, seed it with
`seed_from_sabersim(..., replace_older=True)` and apply it to its slate.
Every week, after games are played and before building the next slate:

1. Refresh 2026 stats (`refresh_player_weekly_stats` / `refresh_dst_weekly_stats`).
2. `models.player_baselines.update_baselines((season, week))` - each played
   game moves a baseline 20% toward the real score (games a player missed
   change nothing).
   Safe to run after each game day (Thursday, then Sunday/Monday): each
   baseline is marked only through its own last applied game.
3. `generate_projections` then blends each baseline 50/50 with the model's
   projection automatically and records model / baseline / blend in
   projection_sources; report `projection_accuracy` for the slate after its
   standings are imported, so we can see whether the blend beats the model.

Players without a baseline use the model, which already rebuilds from
recent games each week. The 20% and 50% weights are starting
values; revisit them once several weeks of model-vs-blend accuracy exist.

## Top-5 winner study (2026-10-05): what held up and what didn't

Studied the top 5 lineups of all 27 uploaded contests (5 Classic Sundays
incl. week 1, 7 Showdowns) against the whole field and our builds.
- Real and already in the rules: winners' QB came from a top-5 Vegas-total
  game 71% of the time vs 46% for the field (the spread portfolio covers it).
- Not different from the field: total ownership (112 vs 112), QB ownership
  (~7%), stack size (QB + 1.35 vs 1.33 receivers), kicker/DST use in
  Showdown, captain ownership (11% vs 14%). Don't present these as edges.
- Looked like patterns but failed a retro test (stored projections, same
  pool, only the rule changed) - do not adopt without new evidence:
  no RB in Classic FLEX (winners 14% vs field 40% 3-RB): avg finish
  55/44/36% -> 54/47/39%, cashes 18 -> 15 over 9/20, 9/27, 10/04.
  Showdown captain's team 4+ of 6 (winners 84% vs field 66%): avg finish
  43% -> 44%, cashes 35 -> 33, worse on 3 of 6 slates.
- What actually separated us from the winners on 9/27 and 10/04 was the
  players, not construction: 0 exposure to the cores (Geno/G. Wilson/Sadiq/
  JSN; Stroud/Lamb/Collins/Hockenson), from projection misses (Geno $4,900
  ranked QB29, rookie Sadiq projected 3.7, injury-returner Collins 11.1 at
  $7,200) plus the pre-spread concentration.
Scripts: scratchpad top5_patterns.py / top5_agg.py / retro_patterns.py.

Game-script Showdown builds (tested 2026-10-06 after ATL@NO, where every
top-100 lineup was Kamara/Bijan/B. Robinson and none of ours was): simulate
the game 1,000 times, build the best lineup for each simulated game, and
pick the 20 that cash in the most different outcomes. On all 7 Showdowns
with standings (stored projections, same exclusions/caps/captain rule) it
finished better on average (top 44% vs 48%, 5 of 7 slates) and cashed more
(40 vs 35 of ~140), but hit fewer top-1% finishes (2 vs 5), and on a typical
GPP payout curve it returned $1.27 per $1 vs $1.80 for the current build
(also behind at flatter and steeper curves). 10 optimizer + 10 game-script
was worse than both. Not adopted; code kept in the scratchpad
(scenario_builds_rejected.py). Revisit only with more slates, or for
cash-style contests where finishing in the money matters more than the top.
