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
upside wider - candidates to verify against our own history before
changing anything, not changes to make on one file.

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
