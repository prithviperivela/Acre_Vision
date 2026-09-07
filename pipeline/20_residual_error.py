"""Can an OSM-only signal separate the interstitial rule's 12 false-Vacant chips?

Those are places where OSM has mapped no BUILDINGS but a settlement exists.
Settlements usually still have their STREETS mapped, so residential/service road
density is an independent signal the building layer does not carry.

This is diagnostic only -- it proposes a v7 rule and reports how well it would
separate the errors.  It is NOT applied, because the threshold would then be
fitted to the gold set and gold would stop being an independent test.
"""
import numpy as np, geopandas as gpd, rasterio, json
from rasterio.features import rasterize
from pathlib import Path
TIF='/home/prithvi/Downloads/Hyderabad_Sentinel2_12Bands_JanMar2025.tif'
OSM=Path('/home/prithvi/osm_integration /hyderabad_osm_data')
D=Path('/home/prithvi/AcreVision_v2/data'); G=Path('/home/prithvi/AcreVision_v2/outputs/gold')
CS=32
MINOR={'residential','service','living_street','unclassified','footway','path','track'}
with rasterio.open(TIF) as s: tr,W,H,crs=s.transform,s.width,s.height,s.crs
rd=gpd.read_file(OSM/'gis_osm_roads_free_1_hyderabad.shp').to_crs(crs)
po=gpd.read_file(OSM/'gis_osm_pois_free_1_hyderabad.shp').to_crs(crs)
print(f'{len(rd)} roads, {len(po)} POIs',flush=True)
def burn(g,shape=(H,W)):
    if len(g)==0: return np.zeros(shape,bool)
    return rasterize(((x,1) for x in g.geometry),out_shape=shape,transform=tr,fill=0,
                     dtype=np.uint8,all_touched=True).astype(bool)
def boxmean(a,k):
    p=np.pad(a.astype(np.float32),k,mode='edge'); c=p.cumsum(0).cumsum(1)
    c=np.pad(c,((1,0),(1,0)))
    return (c[2*k+1:,2*k+1:]-c[:-2*k-1,2*k+1:]-c[2*k+1:,:-2*k-1]+c[:-2*k-1,:-2*k-1])/((2*k+1)**2)
minor=boxmean(burn(rd[rd.fclass.isin(MINOR)]),5)     # 110 m window
poi  =boxmean(burn(po),10)                           # 210 m window

rc=np.load(D/'rowcol.npy')
gold={int(k):v for k,v in json.load(open(G/'gold_claude.json')).items() if not k.startswith('_')}
sample={s['gid']:s for s in json.load(open(G/'sample.json'))}
rows=[]
for g,gl in gold.items():
    if gl=='Mixed' or sample[g]['stratum']!='interstitial': continue
    r,c=rc[g]
    rows.append((gl,float(minor[r:r+CS,c:c+CS].mean()),float(poi[r:r+CS,c:c+CS].mean())))
print(f'\n{len(rows)} interstitial-stratum gold chips')
print(f'{"gold":8s} {"n":>3s}  {"minor-road density":>20s}  {"POI density":>14s}')
for gl in ['Vacant','Urban','Other']:
    sub=[r for r in rows if r[0]==gl]
    if not sub: continue
    m=np.array([x[1] for x in sub]); p=np.array([x[2] for x in sub])
    print(f'{gl:8s} {len(sub):3d}  {m.mean():9.4f}+-{m.std():7.4f}  {p.mean():7.5f}+-{p.std():6.5f}')

y=np.array([r[0]=='Urban' for r in rows]); s_=np.array([r[1] for r in rows])
o=np.argsort(-s_); yy=y[o]; pos,neg=y.sum(),(~y).sum()
auc=1.0-(np.cumsum(~yy)[yy].sum())/(pos*neg)
print(f'\nminor-road density as an "actually a settlement" detector: AUC={auc:.3f}')
print('  (1.0 = perfectly separates the false-Vacant Urban chips from the true ones)')
best=None
for t in np.percentile(s_,np.arange(5,96,5)):
    keep=s_<=t
    if keep.sum()==0: continue
    prec=100*np.mean([rows[i][0]=='Vacant' for i in range(len(rows)) if keep[i]])
    rec=100*keep[[r[0]=='Vacant' for r in rows]].sum()/max(sum(1 for r in rows if r[0]=='Vacant'),1)
    if best is None or prec>best[1]: best=(t,prec,rec,int(keep.sum()))
print(f'\nIF a v7 gate "minor-road density <= {best[0]:.4f}" were applied to interstitial land:')
print(f'  Vacant precision {best[1]:.1f}% (from 67.4%), recall of true-Vacant {best[2]:.0f}%, n={best[3]}')
print('  NOT APPLIED -- the threshold is fitted to gold; validate on a fresh gold batch first.')
json.dump(dict(auc=float(auc),proposed_threshold=float(best[0]),
               would_be_precision=float(best[1])),open(Path('/home/prithvi/AcreVision_v2/outputs')/'v7_proposal.json','w'),indent=2)
