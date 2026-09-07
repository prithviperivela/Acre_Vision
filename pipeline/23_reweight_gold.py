"""The gold sample is stratified, not random -- correct for it.

Chips were drawn 48/16/16/8/8 across strata to put measurement power on the
disputed interstitial class.  That makes the raw gold accuracy an accuracy on a
Vacant-enriched sample, NOT a city-wide figure.  Reweighting each stratum by how
much of the labelled city it actually represents gives the honest city-level
number.
"""
import numpy as np, json
from pathlib import Path
D=Path('/home/prithvi/AcreVision_v2/data'); O=Path('/home/prithvi/AcreVision_v2/outputs'); G=O/'gold'
CLS=['Vacant','Urban','Water','Other']
gold={int(k):v for k,v in json.load(open(G/'gold_claude.json')).items() if not k.startswith('_')}
sample={s['gid']:s for s in json.load(open(G/'sample.json'))}
L5=np.load(D/'osm_labels5.npy',mmap_mode='r'); L6=np.load(D/'osm_labels6.npy',mmap_mode='r')
L4=np.load(D/'osm_labels4.npy',mmap_mode='r')
lp=np.load(D/'osm_label_frac6.npy'); valid=np.load(D/'valid.npy'); off=np.load(D/'off64.npy')

# population: how many train-eligible chips fall in each stratum?
idx=np.where(valid&(lp>=0.25)&(off[:,0]==16)&(off[:,1]==16))[0]
a5=np.asarray(L5[idx]); a4=np.asarray(L4[idx]); a6=np.asarray(L6[idx])
f_new=((a5==0)&(a4!=0)).reshape(len(idx),-1).mean(1)
f_old=((a5==0)&(a4==0)).reshape(len(idx),-1).mean(1)
f={k:(a6.reshape(len(idx),-1)==k).mean(1) for k in range(4)}
pop={'interstitial':(f_new>0.5).sum(),'vacant_tag':(f_old>0.5).sum(),
     'urban':(f[1]>0.7).sum(),'water':(f[2]>0.7).sum(),'other':(f[3]>0.7).sum()}
covered=sum(pop.values())
print(f'population of train-eligible chips per stratum (n={len(idx)}, {covered} fall in a stratum)')
for k,v in pop.items(): print(f'  {k:13s} {v:6d}  ({100*v/covered:5.1f}% of covered)')

def maj(a):
    a=np.asarray(a); a=a[a!=255]
    return CLS[int(np.bincount(a,minlength=4).argmax())] if a.size else None

rows=[(g,v,sample[g]['stratum']) for g,v in gold.items() if v!='Mixed']
print(f'\n{"":26s} {"raw (stratified)":>18s} {"reweighted to city":>20s}')
for tag,L in [('v4 labels',L4),('v6 labels',L6)]:
    per={}
    for st in pop:
        sub=[(g,gl) for g,gl,s_ in rows if s_==st]
        per[st]=np.mean([maj(L[g])==gl for g,gl in sub]) if sub else np.nan
    raw=np.mean([maj(L[g])==gl for g,gl,_ in rows])
    rw=np.nansum([per[st]*pop[st]/covered for st in pop])
    print(f'  {tag:24s} {100*raw:17.1f}% {100*rw:19.1f}%')

# same for the trained models, using saved per-seed gold predictions if present
fv=O/'final_v6.json'
if fv.exists():
    d=json.load(open(fv))
    print(f'\n  (model gold accuracy is reported raw; the stratum mix is identical across')
    print(f'   v4/v6 model runs, so the v4-vs-v6 COMPARISON is unaffected by the weighting.)')
json.dump({'pop':{k:int(v) for k,v in pop.items()},'covered':int(covered)},
          open(O/'gold_strata.json','w'),indent=2)
