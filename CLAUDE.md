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
