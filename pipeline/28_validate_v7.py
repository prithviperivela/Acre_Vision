"""The honest test of v7: score every label version on gold batch 2.

Batch 2 was fetched, shuffled and classified without reference to the v7 road
gate, and the gate's threshold was fitted on batch 1 alone. So batch 2 is the
first measurement of v7 that is not contaminated by its own tuning.

It also carries an 'unlabelled' stratum, which batch 1 lacked -- the first
measurement of anything on the 17,921 parcels that dominate the city-wide mean.
"""
import numpy as np, json
from pathlib import Path
from collections import Counter
D=Path('/home/prithvi/AcreVision_v2/data'); O=Path('/home/prithvi/AcreVision_v2/outputs')
CLS=['Vacant','Urban','Water','Other']
g1={int(k):v for k,v in json.load(open(O/'gold'/'gold_claude.json')).items() if not k.startswith('_')}
g2={int(k):v for k,v in json.load(open(O/'gold2'/'gold_claude2.json')).items() if not k.startswith('_')}
s2={s['gid']:s for s in json.load(open(O/'gold2'/'sample.json'))}
L={v:np.load(D/f'osm_labels{v}.npy',mmap_mode='r') for v in [4,5,6,7]}
def maj(a):
    a=np.asarray(a); a=a[a!=255]
    return CLS[int(np.bincount(a,minlength=4).argmax())] if a.size else None

def report(gold,title,restrict=None):
    rows=[(g,v) for g,v in gold.items() if v!='Mixed' and (restrict is None or restrict(g))]
    print(f'\n=== {title}  (n={len(rows)}) ===')
    print(f'{"ver":5s} {"Vacant prec":>12s} {"Vacant rec":>11s} {"Vacant F1":>10s} {"accuracy":>9s}   errors on Vacant')
    out={}
    for v in [4,5,6,7]:
        pred={g:maj(L[v][g]) for g,_ in rows}
        vp=[gl for g,gl in rows if pred[g]=='Vacant']
        tot=[gl for g,gl in rows if gl=='Vacant']
        tp=sum(1 for x in vp if x=='Vacant')
        prec=100*tp/max(len(vp),1); rec=100*tp/max(len(tot),1)
        f1=100*2*tp/max(2*tp+ (len(vp)-tp) + (len(tot)-tp),1)
        acc=100*sum(1 for g,gl in rows if pred[g]==gl)/max(len(rows),1)
        err=Counter(x for x in vp if x!='Vacant')
        out[v]=dict(prec=prec,rec=rec,f1=f1,acc=acc)
        print(f'v{v:<4} {prec:11.1f}% {rec:10.1f}% {f1:9.1f} {acc:8.1f}%   {dict(err) or "none"}')
    return out

r1=report(g1,'GOLD BATCH 1 -- the v7 gate was FITTED here, so this is optimistic')
r2=report(g2,'GOLD BATCH 2 -- never used for fitting: THE HONEST TEST')
report(g2,'BATCH 2, interstitial stratum only (what the gate targets)',
       restrict=lambda g: s2[g]['stratum']=='interstitial')

# the previously unmeasured 60%
unl=[g for g,v in g2.items() if s2[g]['stratum']=='unlabelled']
print(f'\n=== UNLABELLED parcels -- never measured before (n={len(unl)}) ===')
print('OSM assigns these no label at all, so no label rule can be scored on them.')
print('What is actually there, by verified imagery:')
for k,c in Counter(g2[g] for g in unl).most_common():
    print(f'  {k:8s} {c:3d}  ({100*c/len(unl):.0f}%)')

print(f'\n=== v6 -> v7 on batch 2 ===')
d=r2[7]['f1']-r2[6]['f1']
print(f'  Vacant F1 {r2[6]["f1"]:.1f} -> {r2[7]["f1"]:.1f}  ({d:+.1f})')
print(f'  Vacant precision {r2[6]["prec"]:.1f}% -> {r2[7]["prec"]:.1f}%')
print(f'  accuracy {r2[6]["acc"]:.1f}% -> {r2[7]["acc"]:.1f}%')
json.dump({'batch1':r1,'batch2':r2},open(O/'v7_validation.json','w'),indent=2)
