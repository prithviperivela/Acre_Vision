"""Quantify the two failure modes found in manual review of the ranker.

Manual review of the top-ranked parcels found the vacancy call itself sound --
no water bodies surfaced, no obvious false positives -- but flagged two kinds of
land that ARE open and ARE correctly classified as vacant, and yet are not
acquirable:

  A. RIVER / STREAM CORRIDOR.  Open ground lying between the braids of the Musi
     and its drains. Genuinely bare, genuinely vacant, and legally undevelopable:
     it is floodplain, usually government land, and under HMDA rules carries a
     buffer restriction.
  B. INSTITUTIONAL CURTILAGE.  The open grounds of colleges, hospitals, military
     compounds and government campuses. Open, unbuilt, and not for sale.

Neither is a model error. The model was asked "is this land open?" and answered
correctly. Nobody ever asked "can it be bought and built on?" -- there is no
constraint layer in the pipeline. This script measures how much of the ranked
output each failure mode actually touches, so the upgrade can be scoped rather
than guessed at.

It also checks the reviewer's second point: that both failures are partly an
artefact of ground sample distance, because the 10 m pixel forces a 320 m parcel.
"""
import json
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from pathlib import Path

TIF = '/home/prithvi/Downloads/Hyderabad_Sentinel2_12Bands_JanMar2025.tif'
OSM = Path('/home/prithvi/osm_integration /hyderabad_osm_data')
D = Path('/home/prithvi/AcreVision_v2/data')
O = Path('/home/prithvi/AcreVision_v2/outputs')
CS = 32
PX = 10.0                                   # ground sample distance, metres

INSTITUTIONAL = {'school', 'college', 'university', 'hospital', 'clinic',
                 'community_centre', 'graveyard', 'sports_centre', 'stadium',
                 'police', 'fire_station', 'prison', 'town_hall', 'courthouse'}
WATERWAY_BUFFER_M = 100                     # HMDA-style corridor, both banks
INSTITUTION_BUFFER_M = 120                  # campus curtilage around a point POI

with rasterio.open(TIF) as s:
    tr, W, H, crs = s.transform, s.width, s.height, s.crs
shape = (H, W)
px_km2 = (PX*PX)/1e6

print('=== parcel geometry, and why it is 25 acres ===')
side = CS*PX
print(f'  chip {CS} px x {PX:.0f} m GSD = {side:.0f} m square')
print(f'  area = {side*side/10000:.2f} ha = {side*side/4046.86:.1f} acres')
print('  The parcel size is not a modelling choice -- it is what one 32 px chip')
print('  covers at Sentinel-2 resolution. A finer sensor gives a finer parcel.\n')


def burn(gdf, buf=0):
    if len(gdf) == 0:
        return np.zeros(shape, bool)
    g = gdf.buffer(buf) if buf else gdf.geometry
    return rasterize(((x, 1) for x in g), out_shape=shape, transform=tr,
                     fill=0, dtype=np.uint8, all_touched=True).astype(bool)


print('building constraint layers...', flush=True)
ww = gpd.read_file(OSM/'gis_osm_waterways_free_1_hyderabad.shp').to_crs(crs)
wa = gpd.read_file(OSM/'gis_osm_water_a_free_1_hyderabad.shp').to_crs(crs)
poi = gpd.read_file(OSM/'gis_osm_pois_free_1_hyderabad.shp').to_crs(crs)
poia = gpd.read_file(OSM/'gis_osm_pois_a_free_1_hyderabad.shp').to_crs(crs)
lu = gpd.read_file(OSM/'gis_osm_landuse_a_free_1_hyderabad.shp').to_crs(crs)

river = burn(ww, WATERWAY_BUFFER_M) | burn(wa, WATERWAY_BUFFER_M)
inst = (burn(poi[poi.fclass.isin(INSTITUTIONAL)], INSTITUTION_BUFFER_M)
        | burn(poia[poia.fclass.isin(INSTITUTIONAL)], 40)
        | burn(lu[lu.fclass.isin({'military', 'cemetery', 'recreation_ground'})]))
constrained = river | inst
print(f'  river/stream corridor ({WATERWAY_BUFFER_M} m)  {river.mean()*100:5.2f}% of AOI'
      f'  = {river.sum()*px_km2:7.1f} km2')
print(f'  institutional curtilage             {inst.mean()*100:5.2f}% of AOI'
      f'  = {inst.sum()*px_km2:7.1f} km2')
print(f'  either                              {constrained.mean()*100:5.2f}% of AOI'
      f'  = {constrained.sum()*px_km2:7.1f} km2\n')

rc = np.load(D/'rowcol.npy')
gid = np.load(D/'gid.npy')
data = json.load(open(O/'dashboard'/'data.json'))
g2i = {int(g): i for i, g in enumerate(gid)}

frac_r, frac_i = [], []
for g in data['gid']:
    i = g2i.get(int(g))
    if i is None:
        frac_r.append(0.0); frac_i.append(0.0); continue
    r, c = rc[i]
    frac_r.append(float(river[r:r+CS, c:c+CS].mean()))
    frac_i.append(float(inst[r:r+CS, c:c+CS].mean()))
df = pd.DataFrame(dict(gid=data['gid'], vac8=data['vac8'], road=data['road'],
                       heal=data['heal'], educ=data['educ'], life=data['life'],
                       esse=data['esse'], river=frac_r, inst=frac_i))

DEF = dict(vac8=35, road=20, heal=15, educ=10, life=12, esse=8)
tot = sum(DEF.values())
df['composite'] = sum(df[k]*(w/tot) for k, w in DEF.items())
df['blocked'] = (df.river > 0.25) | (df.inst > 0.25)

print('=== how much of the RANKED OUTPUT the two failure modes touch ===')
print('(default weights: vacancy 35, road 20, healthcare 15, education 10, lifestyle 12, essential 8)\n')
print(f'{"cohort":>14s} {"n":>6s} {"river>25%":>10s} {"institutional>25%":>18s} {"either":>8s}')
for label, sub in [('top 25', df.nlargest(25, 'composite')),
                   ('top 100', df.nlargest(100, 'composite')),
                   ('top 500', df.nlargest(500, 'composite')),
                   ('all parcels', df)]:
    print(f'{label:>14s} {len(sub):6d} {100*(sub.river>.25).mean():9.1f}% '
          f'{100*(sub.inst>.25).mean():17.1f}% {100*sub.blocked.mean():7.1f}%')

print('\n=== the ranked list after a developability filter ===')
keep = df[~df.blocked]
print(f'  {len(keep)} of {len(df)} parcels survive ({100*len(keep)/len(df):.1f}%)')
old_top = set(df.nlargest(100, 'composite').gid)
new_top = set(keep.nlargest(100, 'composite').gid)
print(f'  top-100 overlap before/after filter: {len(old_top & new_top)}/100')
print(f'  {100-len(old_top & new_top)} parcels would be replaced by the next-best unconstrained land')

print('\n=== what a finer sensor would buy ===')
for gsd, name in [(10, 'Sentinel-2  (current, free)'),
                  (3, 'Planet SuperDove (commercial)'),
                  (0.5, 'Pleiades / Maxar (commercial)')]:
    s_ = CS*gsd
    print(f'  {name:32s} {CS}px = {s_:6.0f} m = {s_*s_/4046.86:8.2f} acres per parcel')
print('  At 0.5 m a parcel is 0.06 acres, so a college ground and the vacant plot')
print('  beside it stop sharing a chip -- which is exactly the confusion reported.')

df.to_csv(O/'developability_audit.csv', index=False)
json.dump(dict(river_pct_aoi=float(river.mean()*100), inst_pct_aoi=float(inst.mean()*100),
               constrained_pct_aoi=float(constrained.mean()*100),
               top100_blocked=float(100*df.nlargest(100, 'composite').blocked.mean()),
               top25_blocked=float(100*df.nlargest(25, 'composite').blocked.mean()),
               all_blocked=float(100*df.blocked.mean()),
               parcel_side_m=side, parcel_acres=side*side/4046.86),
          open(O/'developability_audit.json', 'w'), indent=2)
print(f'\nwrote {O}/developability_audit.csv and .json')
