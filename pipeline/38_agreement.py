"""Turn the three human label sets into the result that validates this project.

Run once all three of you have finished:  python 38_agreement.py

It answers two questions.

1. DO THE THREE OF YOU AGREE WITH EACH OTHER?
   Reported as Fleiss' kappa, the standard measure for >2 annotators. It
   corrects for agreement that would happen by chance, so a class that is 80%
   Vacant cannot score well just by everyone guessing Vacant.
       kappa > 0.80  excellent -- the class definitions are unambiguous
       0.60 - 0.80   substantial -- normal for land-cover work, publishable
       0.40 - 0.60   moderate -- the definitions need tightening
       < 0.40        poor -- the task as posed is not well defined
   If kappa is low, the per-class and per-stratum breakdown below says WHERE,
   and that is a finding in itself, not a failure.

2. WAS THE MODEL-GENERATED GROUND TRUTH RIGHT?
   The existing 256 labels were produced by Claude. This scores them against the
   human majority vote. If they agree closely, every number in the report stands
   on human-verified ground. If they do not, the report's numbers get recomputed
   against the human labels instead -- which is the correct outcome either way.
"""
import json
import numpy as np
from pathlib import Path
from collections import Counter
from itertools import combinations

O = Path('/home/prithvi/AcreVision_v2/outputs')
CLS = ['Vacant', 'Urban', 'Water', 'Other']
ALL = CLS + ['Mixed']

order = json.load(open(O/'label_images'/'order.json'))
strat = {str(s['gid']): s['stratum'] for s in order}

humans = {}
for f in sorted(O.glob('gold_human_*.json')):
    name = f.stem.replace('gold_human_', '')
    humans[name] = json.load(open(f))
    print(f'  {name:12s} {len(humans[name])} labels')
if len(humans) < 2:
    raise SystemExit('\nNeed at least two annotators. Each runs:\n'
                     '    python scripts/37_label_gold.py <name>')

claude = {}
for p in [O/'gold'/'gold_claude.json', O/'gold2'/'gold_claude2.json',
          O/'gold3'/'gold_claude3.json']:
    if p.exists():
        claude.update({k: v for k, v in json.load(open(p)).items()
                       if not k.startswith('_')})

common = sorted(set.intersection(*[set(h) for h in humans.values()]))
print(f'\n{len(common)} parcels labelled by all {len(humans)} annotators')
if not common:
    raise SystemExit('No overlap yet -- everyone must label the SAME parcels.')

# ---------- 1. agreement between humans ----------
print('\n=== 1. DO THE ANNOTATORS AGREE WITH EACH OTHER? ===')
names = sorted(humans)
print('\npairwise raw agreement:')
for a, b in combinations(names, 2):
    ag = np.mean([humans[a][g] == humans[b][g] for g in common])
    print(f'  {a:10s} vs {b:10s}  {100*ag:5.1f}%')

n_rat = len(names)
counts = np.array([[sum(1 for a in names if humans[a][g] == c) for c in ALL]
                   for g in common], float)
P_i = ((counts**2).sum(1) - n_rat) / (n_rat*(n_rat-1))
P_bar = P_i.mean()
p_j = counts.sum(0) / (len(common)*n_rat)
P_e = (p_j**2).sum()
kappa = (P_bar - P_e) / (1 - P_e) if P_e < 1 else float('nan')
verdict = ('excellent - definitions are unambiguous' if kappa > .8 else
           'substantial - normal for land-cover work' if kappa > .6 else
           'moderate - definitions need tightening' if kappa > .4 else
           'poor - the task as posed is not well defined')
print(f'\nFleiss kappa = {kappa:.3f}   ({verdict})')
print(f'  mean raw agreement {100*P_bar:.1f}%, chance level {100*P_e:.1f}%')

unan = [g for g in common if len({humans[a][g] for a in names}) == 1]
print(f'  unanimous on {len(unan)}/{len(common)} parcels ({100*len(unan)/len(common):.0f}%)')

print('\nagreement by class (how often the majority class was unanimous):')
for c in ALL:
    sub = [g for g in common if Counter(humans[a][g] for a in names).most_common(1)[0][0] == c]
    if not sub:
        continue
    u = sum(1 for g in sub if len({humans[a][g] for a in sub and names}) == 1) if False else \
        sum(1 for g in sub if len({humans[a][g] for a in names}) == 1)
    print(f'  {c:8s} n={len(sub):3d}   unanimous {100*u/len(sub):5.0f}%')

print('\nagreement by sampling stratum (where the hard cases live):')
for st in sorted({strat.get(g, '?') for g in common}):
    sub = [g for g in common if strat.get(g) == st]
    if not sub:
        continue
    u = sum(1 for g in sub if len({humans[a][g] for a in names}) == 1)
    print(f'  {st:14s} n={len(sub):3d}   unanimous {100*u/len(sub):5.0f}%')

# ---------- 2. was the model-generated truth right? ----------
def majority(g):
    c = Counter(humans[a][g] for a in names).most_common()
    return c[0][0] if (len(c) == 1 or c[0][1] > c[1][1]) else None   # None = tie

print('\n=== 2. WAS THE MODEL-GENERATED GROUND TRUTH RIGHT? ===')
scor = [g for g in common if majority(g) not in (None, 'Mixed') and g in claude
        and claude[g] != 'Mixed']
ties = sum(1 for g in common if majority(g) is None)
if not scor:
    raise SystemExit('nothing scorable yet')
agree = [g for g in scor if claude[g] == majority(g)]
print(f'\nClaude vs human majority: {len(agree)}/{len(scor)} = '
      f'{100*len(agree)/len(scor):.1f}% agreement   ({ties} ties excluded)')

print('\nwhere they differ  (human majority -> Claude said):')
diff = Counter((majority(g), claude[g]) for g in scor if claude[g] != majority(g))
for (h, c), n in diff.most_common():
    print(f'  {h:8s} -> {c:8s}  {n:3d}')

print('\nper-class recall of the Claude labels against human majority:')
for c in CLS:
    tot = [g for g in scor if majority(g) == c]
    if not tot:
        continue
    hit = sum(1 for g in tot if claude[g] == c)
    print(f'  {c:8s} n={len(tot):3d}   {100*hit/len(tot):5.1f}%')

human_truth = {g: majority(g) for g in scor}
json.dump({'kappa': float(kappa), 'n_common': len(common), 'annotators': names,
           'unanimous_frac': len(unan)/len(common),
           'claude_vs_human': len(agree)/len(scor),
           'human_majority': human_truth},
          open(O/'human_agreement.json', 'w'), indent=2)
print(f'\nwrote {O}/human_agreement.json')
print('\nNEXT: if Claude-vs-human agreement is high, the report stands as published.')
print('If it is not, rerun scripts 32 and 35 against human_majority to get the')
print('corrected numbers -- the human labels win, always.')
