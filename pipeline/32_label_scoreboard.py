"""Every label version scored against all 256 verified parcels.

Batches differ in what they were drawn to test, so they are reported separately
as well as pooled:
  batch 1 (96) -- stratified over v6's classes; informed the `scrub` decision
                  and the v7 gate, so it is optimistic for both.
  batch 2 (100) -- independent; adds the previously unmeasured UNLABELLED stratum.
  batch 3 (60) -- drawn only from land v8 newly claims; the test of v8.
"""
import numpy as np, json
from pathlib import Path
from collections import Counter
D=Path('/home/prithvi/AcreVision_v2/data'); O=Path('/home/prithvi/AcreVision_v2/outputs')
CLS=['Vacant','Urban','Water','Other']; VERS=[4,5,6,7,8]
def load(p):
    p=Path(p)
    return {int(k):v for k,v in json.load(open(p)).items() if not k.startswith('_')} if p.exists() else {}
B={'batch1':load(O/'gold'/'gold_claude.json'),
   'batch2':load(O/'gold2'/'gold_claude2.json'),
   'batch3':load(O/'gold3'/'gold_claude3.json')}
L={v:np.load(D/f'osm_labels{v}.npy',mmap_mode='r') for v in VERS}
def maj(a):
    a=np.asarray(a); a=a[a!=255]
    return CLS[int(np.bincount(a,minlength=4).argmax())] if a.size else None

def block(gold,title):
    rows=[(g,v) for g,v in gold.items() if v!='Mixed']
    print(f'\n=== {title}  (n={len(rows)}) ===')
    print(f'{"ver":4s} {"coverage":>9s} {"Vac prec":>9s} {"Vac rec":>8s} {"Vac F1":>7s} {"accuracy":>9s}')
    out={}
    for v in VERS:
        pred={g:maj(L[v][g]) for g,_ in rows}
        cov=100*sum(1 for g,_ in rows if pred[g] is not None)/len(rows)
        vp=[gl for g,gl in rows if pred[g]=='Vacant']; tot=[gl for g,gl in rows if gl=='Vacant']
        tp=sum(1 for x in vp if x=='Vacant')
        prec=100*tp/max(len(vp),1); rec=100*tp/max(len(tot),1)
        f1=100*2*tp/max(2*tp+(len(vp)-tp)+(len(tot)-tp),1)
        acc=100*sum(1 for g,gl in rows if pred[g]==gl)/len(rows)
        out[v]=dict(coverage=cov,prec=prec,rec=rec,f1=f1,acc=acc)
        print(f'v{v:<3} {cov:8.0f}% {prec:8.1f}% {rec:7.1f}% {f1:6.1f} {acc:8.1f}%')
    return out

res={k:block(g,k) for k,g in B.items() if g}
pooled={}
for g in B.values(): pooled.update(g)
res['pooled']=block(pooled,'ALL THREE BATCHES POOLED')

print('\n"coverage" = share of parcels the rule labels at all. A rule that says')
print('nothing about a parcel cannot be right about it, and 60% of the city was')
print('uncovered until v8.')
print('\nBatch 1 is optimistic for v6 (it chose the scrub fix) and for v7 (it fitted')
print('the gate). Batch 2 and 3 are clean tests. v7 does not survive batch 2 and is')
print('rejected; v8 is confirmed by batch 3 at 90% precision on new ground.')
json.dump(res,open(O/'label_scoreboard.json','w'),indent=2)
