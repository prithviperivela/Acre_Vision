"""How well does the PREVIOUS deliverable (vacancy_scores_v2.csv) match ground truth?

v2 was the last session's output: a U-Net trained on v4 OSM labels, single seed,
32 px, no TTA.  Scoring it against the blind gold set gives an honest 'before'
number for the report -- and separates 'the model was bad' from 'the answer key
was bad', which the OSM-vs-OSM metrics could not do.
"""
import numpy as np, pandas as pd, json
from pathlib import Path
D = Path('/home/prithvi/AcreVision_v2/data'); O = Path('/home/prithvi/AcreVision_v2/outputs')
G = O/'gold'
gid = np.load(D/'gid.npy')
gold = {int(k): v for k, v in json.load(open(G/'gold_claude.json')).items() if not k.startswith('_')}
gold = {g: v for g, v in gold.items() if v != 'Mixed'}

df = pd.read_csv(O/'vacancy_scores_v2.csv').set_index('original_chip_id')
rows = []
for g, gl in gold.items():
    cid = int(gid[g])
    if cid not in df.index: continue
    r = df.loc[cid]
    pct = {'Vacant': r.vacancy_score, 'Urban': r.urban_percentage,
           'Water': r.water_percentage, 'Other': r.other_percentage}
    rows.append(dict(gid=g, gold=gl, pred=max(pct, key=pct.get), vac=r.vacancy_score))
print(f'{len(rows)} gold chips matched into vacancy_scores_v2.csv\n')

CLS = ['Vacant', 'Urban', 'Water', 'Other']
acc = 100*np.mean([r['gold'] == r['pred'] for r in rows])
print(f'v2 chip-level accuracy vs gold : {acc:.1f}%')
print('\nper-class F1 (v2 vs gold)')
for c in CLS:
    tp = sum(1 for r in rows if r['pred'] == c and r['gold'] == c)
    fp = sum(1 for r in rows if r['pred'] == c and r['gold'] != c)
    fn = sum(1 for r in rows if r['pred'] != c and r['gold'] == c)
    print(f'  {c:7s} F1={100*2*tp/max(2*tp+fp+fn,1):5.1f}  (tp={tp} fp={fp} fn={fn})')

print('\nconfusion  gold(rows) vs v2 prediction(cols)')
print(f'{"":9s}'+''.join(f'{c:>8s}' for c in CLS))
for gc in CLS:
    print(f'{gc:9s}'+''.join(f'{sum(1 for r in rows if r["gold"]==gc and r["pred"]==pc):8d}' for pc in CLS))

# ranking quality: does a higher vacancy score mean more likely truly vacant?
y = np.array([r['gold'] == 'Vacant' for r in rows]); s = np.array([r['vac'] for r in rows])
o = np.argsort(-s); yy = y[o]
pos, neg = y.sum(), (~y).sum()
# cumsum(~yy)[yy] counts negatives ranked ABOVE each positive = discordant pairs
disc = (np.cumsum(~yy)[yy].sum())/(pos*neg) if pos and neg else float('nan')
auc = 1.0 - disc
print(f'\nranking AUC (vacancy_score separates truly-Vacant chips): {auc:.3f}')
print(f'  mean vacancy_score  gold=Vacant {s[y].mean():.1f}%   gold=other {s[~y].mean():.1f}%')
json.dump(dict(acc=acc, auc=float(auc), n=len(rows)), open(O/'v2_vs_gold.json', 'w'), indent=2)
