"""How much should the playing-time floor lean on last season?

For every RB/WR/TE player-week 2024-2026 (the player played that week and
has 1+ earlier games this season), rebuild the gate's blended snap share and
touches from strictly earlier games at several K values
(weight_on_current = n/(n+K); K=0 = this season only), then check:
  1. snap-share prediction error vs the real snap share that week
  2. the floor's exclusions: how many, their real average DK points, and how
     many real contributors (>= threshold points) each K would have thrown out.
"""
import json

import numpy as np
import pandas as pd

from db.migrate import get_engine
from models import playing_time_engine as pte
from models.calibration import MEANINGFUL_SCORE_THRESHOLD

e = get_engine()
df = pd.read_sql("""select player_id, player_name, position, season, week, snap_pct::float snap,
    coalesce(carries,0)+coalesce(receptions,0) touches, fantasy_points_ppr::float dk
    from player_weekly_stats where season between 2023 and 2026 and position in ('RB','WR','TE')""", e)
df = df.sort_values(['player_id', 'season', 'week'])
by_player = {pid: g for pid, g in df.groupby('player_id')}
KS = [0, 1, 2, 4, 8]
floor = {'WR': pte.WR_MIN_SNAP_PCT, 'TE': pte.TE_MIN_SNAP_PCT}

def blend(cur, pri, k):
    if not cur:
        return np.mean(pri) if pri else None
    if not pri or k == 0:
        return np.mean(cur)
    w = len(cur) / (len(cur) + k)
    return w * np.mean(cur) + (1 - w) * np.mean(pri)

recs = []
for pid, g in by_player.items():
    for row in g[g.season >= 2024].itertuples():
        cur = g[(g.season == row.season) & (g.week < row.week)].tail(pte.CURRENT_SEASON_MAX_GAMES)
        pri = g[g.season == row.season - 1].tail(pte.PRIOR_SEASON_MAX_GAMES)
        cur_s = cur.snap.dropna().tolist()
        pri_s = pri.snap.dropna().tolist()
        if not cur_s or not pri_s or pd.isna(row.snap):
            continue  # only where the two sources can disagree
        rec = {'pid': pid, 'name': row.player_name, 'pos': row.position, 'season': row.season, 'week': row.week,
               'n_cur': len(cur_s), 'actual_snap': row.snap, 'dk': row.dk}
        for k in KS:
            s = blend(cur_s, pri_s, k)
            t = blend(cur.touches.tolist(), pri.touches.tolist(), k)
            rec[f'snap_{k}'] = s
            if row.position == 'RB':
                rec[f'excl_{k}'] = bool(s < pte.RB_MIN_SNAP_PCT and t < pte.RB_MIN_TOUCHES)
            else:
                rec[f'excl_{k}'] = bool(s < floor[row.position])
        recs.append(rec)
r = pd.DataFrame(recs)
out = {'n_player_weeks': len(r), 'threshold': MEANINGFUL_SCORE_THRESHOLD, 'by_k': {}}
print('player-weeks', len(r), 'threshold', MEANINGFUL_SCORE_THRESHOLD)
for k in KS:
    ex = r[r[f'excl_{k}']]
    kept = r[~r[f'excl_{k}']]
    d = {'snap_mae_all': r[f'snap_{k}'].sub(r.actual_snap).abs().mean(),
         'snap_mae_early(n<4)': r[r.n_cur < 4][f'snap_{k}'].sub(r[r.n_cur < 4].actual_snap).abs().mean(),
         'excluded': len(ex), 'excluded_avg_dk': ex.dk.mean(), 'kept_avg_dk': kept.dk.mean(),
         'misses': int((ex.dk >= MEANINGFUL_SCORE_THRESHOLD).sum()),
         'miss_rate': float((ex.dk >= MEANINGFUL_SCORE_THRESHOLD).mean()),
         'dud_catches': int((ex.dk < 5).sum())}
    out['by_k'][k] = d
    print(f"K={k}: snapMAE {d['snap_mae_all']:.4f} (early {d['snap_mae_early(n<4)']:.4f}) | excluded {d['excluded']} avg {d['excluded_avg_dk']:.2f} vs kept {d['kept_avg_dk']:.2f} | misses {d['misses']} ({d['miss_rate']:.1%}) | duds caught (<5 pts) {d['dud_catches']}")

# Paired: players the live K=4 excludes but K=0 keeps, and the reverse.
only4 = r[r.excl_4 & ~r.excl_0]
only0 = r[r.excl_0 & ~r.excl_4]
print('\nexcluded by K=4 only (K=0 would keep):', len(only4), 'avg dk', round(only4.dk.mean(), 2), 'hits>=thr', int((only4.dk >= MEANINGFUL_SCORE_THRESHOLD).sum()))
print('excluded by K=0 only (K=4 would keep):', len(only0), 'avg dk', round(only0.dk.mean(), 2), 'hits>=thr', int((only0.dk >= MEANINGFUL_SCORE_THRESHOLD).sum()))
# week-cluster bootstrap of the snap MAE difference K=4 minus K=0
r['d'] = r.snap_4.sub(r.actual_snap).abs() - r.snap_0.sub(r.actual_snap).abs()
wk = r.groupby(['season', 'week']).d.agg(['sum', 'count'])
rng = np.random.default_rng(0)
bs = []
for _ in range(4000):
    s = wk.sample(len(wk), replace=True, random_state=rng.integers(1e9))
    bs.append(s['sum'].sum() / s['count'].sum())
print('snap MAE diff K4-K0', round(r.d.mean(), 4), 'CI95', np.round(np.percentile(bs, [2.5, 97.5]), 4))
print('\nK=4-only exclusions that scored well:')
print(only4[only4.dk >= MEANINGFUL_SCORE_THRESHOLD].sort_values('dk', ascending=False)[['name', 'pos', 'season', 'week', 'n_cur', 'snap_4', 'snap_0', 'actual_snap', 'dk']].head(15).round(3).to_string())
print('\nK=0-only exclusions that scored well:')
print(only0[only0.dk >= MEANINGFUL_SCORE_THRESHOLD].sort_values('dk', ascending=False)[['name', 'pos', 'season', 'week', 'n_cur', 'snap_4', 'snap_0', 'actual_snap', 'dk']].head(10).round(3).to_string())
out['only_k4'] = {'n': len(only4), 'avg_dk': only4.dk.mean(), 'hits': int((only4.dk >= MEANINGFUL_SCORE_THRESHOLD).sum())}
out['only_k0'] = {'n': len(only0), 'avg_dk': only0.dk.mean(), 'hits': int((only0.dk >= MEANINGFUL_SCORE_THRESHOLD).sum())}
out['snap_mae_diff_k4_minus_k0'] = {'mean': r.d.mean(), 'ci95': list(np.percentile(bs, [2.5, 97.5]))}
json.dump(out, open("shrinkage_k_study.json", "w"), indent=1, default=float)


def ci(a, b):
    d = r[f'snap_{a}'].sub(r.actual_snap).abs() - r[f'snap_{b}'].sub(r.actual_snap).abs()
    tmp = r.assign(d=d).groupby(['season', 'week']).d.agg(['sum', 'count'])
    rng = np.random.default_rng(1)
    bs = []
    for _ in range(4000):
        s = tmp.sample(len(tmp), replace=True, random_state=rng.integers(1e9))
        bs.append(s['sum'].sum() / s['count'].sum())
    return round(d.mean(), 4), np.round(np.percentile(bs, [2.5, 97.5]), 4)


for a, b in ((1, 4), (1, 0), (1, 2)):
    print(f'snap MAE K{a}-K{b}', *ci(a, b))
# held-out check: pick K on 2024, evaluate on 2025-2026
for part, mask in (('2024 (fit)', r.season == 2024), ('2025-26 (held out)', r.season >= 2025)):
    print(part, {k: round(r[mask][f'snap_{k}'].sub(r[mask].actual_snap).abs().mean(), 4) for k in KS})
x = r[(r.name == 'Kalif Raymond') & (r.season == 2026)]
print(x[['week', 'n_cur'] + [f'snap_{k}' for k in KS] + [f'excl_{k}' for k in KS]].round(3).to_string())
