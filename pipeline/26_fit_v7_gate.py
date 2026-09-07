"""Fit the v7 minor-road gate on gold batch 1 ONLY.

Batch 1 is spent here: once a threshold is chosen against it, batch 1 can no
longer measure the gate honestly.  Batch 2 (scripts/25) is the held-out test and
is not touched by this script.

Criterion is F1 on the Vacant class, not precision.  Maximising precision alone
picked a threshold that halved recall, which trades one kind of error for
another rather than reducing error.
"""
import numpy as np, geopandas as gpd, rasterio, json
from rasterio.features import rasterize
from pathlib import Path
TIF='/home/prithvi/Downloads/Hyderabad_Sentinel2_12Bands_JanMar2025.tif'
OSM=Path('/home/prithvi/osm_integration /hyderabad_osm_data')
D=Path('/home/prithvi/AcreVision_v2/data'); O=Path('/home/prithvi/AcreVision_v2/outputs')
G=O/'gold'; CS=32
MINOR={'residential','service','living_street','unclassified','footway','path','track'}

with rasterio.open(TIF) as s: tr,W,H,crs=s.transform,s.width,s.height,s.crs
def burn(g):
    if len(g)==0: return np.zeros((H,W),bool)
    return rasterize(((x,1) for x in g.geometry),out_shape=(H,W),transform=tr,fill=0,
                     dtype=np.uint8,all_touched=True).astype(bool)
def boxmean(a,k):
    p=np.pad(a.astype(np.float32),k,mode='edge'); c=p.cumsum(0).cumsum(1)
    c=np.pad(c,((1,0),(1,0)))
    return (c[2*k+1:,2*k+1:]-c[:-2*k-1,2*k+1:]-c[2*k+1:,:-2*k-1]+c[:-2*k-1,:-2*k-1])/((2*k+1)**2)

mf=D/'minor_road_density.npy'
if mf.exists():
    minor=np.load(mf); print('reused minor_road_density.npy',flush=True)
else:
    rd=gpd.read_file(OSM/'gis_osm_roads_free_1_hyderabad.shp').to_crs(crs)
    minor=boxmean(burn(rd[rd.fclass.isin(MINOR)]),5).astype(np.float32)   # 110 m window
    np.save(mf,minor); print(f'built minor_road_density from {len(rd)} roads',flush=True)

rc=np.load(D/'rowcol.npy')
gold={int(k):v for k,v in json.load(open(G/'gold_claude.json')).items() if not k.startswith('_')}
sample={s['gid']:s for s in json.load(open(G/'sample.json'))}
rows=[(g,gl,float(minor[rc[g,0]:rc[g,0]+CS,rc[g,1]:rc[g,1]+CS].mean()))
      for g,gl in gold.items() if gl!='Mixed' and sample[g]['stratum']=='interstitial']
y=np.array([gl=='Vacant' for _,gl,_ in rows]); s_=np.array([m for _,_,m in rows])
print(f'\nfitting on {len(rows)} batch-1 interstitial parcels '
      f'({y.sum()} truly Vacant, {(~y).sum()} not)')

print(f'\n{"threshold":>10s} {"kept":>5s} {"precision":>10s} {"recall":>8s} {"F1":>7s}')
best=None
for t in np.unique(np.round(np.percentile(s_,np.arange(10,100,5)),4)):
    keep=s_<=t
    if keep.sum()==0: continue
    tp=int((keep&y).sum()); fp=int((keep&~y).sum()); fn=int((~keep&y).sum())
    prec=100*tp/max(tp+fp,1); rec=100*tp/max(tp+fn,1)
    f1=100*2*tp/max(2*tp+fp+fn,1)
    star=''
    if best is None or f1>best[3]: best=(t,prec,rec,f1); star=' *'
    print(f'{t:10.4f} {keep.sum():5d} {prec:9.1f}% {rec:7.1f}% {f1:6.1f}{star}')
print(f'\nno gate (baseline): precision {100*y.mean():.1f}%  recall 100%  '
      f'F1 {100*2*y.sum()/(2*y.sum()+(~y).sum()):.1f}')
t,prec,rec,f1=best
print(f'\nCHOSEN threshold {t:.4f}  ->  precision {prec:.1f}%  recall {rec:.1f}%  F1 {f1:.1f}')
print('  (fitted on batch 1; batch 2 is the honest test)')
json.dump({'threshold':float(t),'batch1_precision':prec,'batch1_recall':rec,'batch1_f1':f1,
           'window_px':5,'minor_classes':sorted(MINOR)},
          open(O/'v7_gate.json','w'),indent=2)
