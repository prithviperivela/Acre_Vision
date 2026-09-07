"""Which OSM landuse tag is poisoning the Vacant class?

The gold audit says the v4 'landuse-tag Vacant' rule is only 35% precise, with
every error being gold=Other.  VACANT is a set of 8 fclasses; this attributes
the errors to specific tags so the fix is evidence-led rather than a guess.
"""
import numpy as np, geopandas as gpd, rasterio, json
from rasterio.features import rasterize
from pathlib import Path
from collections import Counter, defaultdict

TIF='/home/prithvi/Downloads/Hyderabad_Sentinel2_12Bands_JanMar2025.tif'
OSM=Path('/home/prithvi/osm_integration /hyderabad_osm_data')
D=Path('/home/prithvi/AcreVision_v2/data'); G=Path('/home/prithvi/AcreVision_v2/outputs/gold')
VACANT=['farmland','scrub','grass','meadow','orchard','vineyard','allotments','heath']
OTHER =['forest','nature_reserve','park','military','cemetery','quarry','recreation_ground']
CS=32
with rasterio.open(TIF) as s: tr,W,H,crs=s.transform,s.width,s.height,s.crs
lu=gpd.read_file(OSM/'gis_osm_landuse_a_free_1_hyderabad.shp').to_crs(crs)

# one raster per fclass of interest
masks={}
for fc in VACANT+OTHER:
    g=lu[lu.fclass==fc]
    masks[fc]=(rasterize(((x,1) for x in g.geometry),out_shape=(H,W),transform=tr,fill=0,
                         dtype=np.uint8,all_touched=True).astype(bool) if len(g) else
               np.zeros((H,W),bool))
rc=np.load(D/'rowcol.npy')
gold={int(k):v for k,v in json.load(open(G/'gold_claude.json')).items() if not k.startswith('_')}
sample={s['gid']:s for s in json.load(open(G/'sample.json'))}

print('=== gold verdict on chips dominated by each VACANT-set fclass ===')
print(f'{"fclass":16s} {"n_gold_chips":>12s}   gold breakdown')
tally=defaultdict(Counter)
for g,gl in gold.items():
    if gl=='Mixed': continue
    r,c=rc[g]
    for fc in VACANT+OTHER:
        frac=masks[fc][r:r+CS,c:c+CS].mean()
        if frac>0.5: tally[fc][gl]+=1
for fc in VACANT:
    t=tally[fc]
    if sum(t.values())==0: continue
    good=100*t['Vacant']/sum(t.values())
    print(f'{fc:16s} {sum(t.values()):12d}   {dict(t)}  -> {good:.0f}% truly Vacant')
print('\n--- for reference, the OTHER-set tags ---')
for fc in OTHER:
    t=tally[fc]
    if sum(t.values())==0: continue
    print(f'{fc:16s} {sum(t.values()):12d}   {dict(t)}')

# what would moving scrub -> Other do to the v4 rule's precision on gold?
print('\n=== simulated fix: move scrub from VACANT to OTHER ===')
bdens=np.load(D/'building_density.npy')
vac_old=np.zeros((H,W),bool); vac_new=np.zeros((H,W),bool)
for fc in VACANT:
    vac_old|=masks[fc]
    if fc!='scrub': vac_new|=masks[fc]
for tag,m in [('v4 VACANT set (with scrub)',vac_old),('VACANT minus scrub',vac_new)]:
    hits=[]
    for g,gl in gold.items():
        if gl=='Mixed': continue
        r,c=rc[g]
        if (m[r:r+CS,c:c+CS]&(bdens[r:r+CS,c:c+CS]<0.02)).mean()>0.5: hits.append(gl)
    if hits:
        print(f'  {tag:28s} n={len(hits):3d}  precision={100*hits.count("Vacant")/len(hits):5.1f}%  {dict(Counter(hits))}')
