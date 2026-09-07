"""Gold-corrected INDEPENDENT weak labels from OSM vector data -- v6.

v6 change, driven by the 96-chip blind gold audit (scripts/14, scripts/15)
------------------------------------------------------------------------
The gold set showed the incumbent v4 Vacant rule was only 35% precise, and
attributed almost all of the error to ONE tag: OSM `scrub`.  Of 12 gold chips
dominated by scrub, 10 are dense woody vegetation (gold=Other) and just 2 are
open developable land -- 17% precision.  scrub is also the largest single
contributor to the old Vacant footprint (47.8 of 74.9 km2), so it dominated the
class it was corrupting.  Moving scrub to Other lifts the landuse-tag rule from
31% to 75% precision on gold.

So v6 = v5 (keep the interstitial rule, 67% precise) with scrub reclassified
from VACANT to OTHER.  This is the only change; the interstitial thresholds are
untouched and remain the a-priori values from v5.

Motivation (carried over from v5)
---------------------------------
v4's Vacant class covered only 2.1% of AOI pixels and scored IoU ~35.  A profile
of the OSM extract shows the obvious fix -- adding more landuse fclasses -- has
no headroom: the whole VACANT-fclass footprint is 74.9 km2 of a 3090 km2 AOI,
there is no greenfield/brownfield tag in this extract, and gis_osm_natural_a
contains exactly one beach.  Relaxing the building-density gate from 0.02 to
"no gate at all" moves Vacant by only 47.5 -> 53.3 km2.

So v5 adds a genuinely new Vacant source instead: MAPPED-BUT-EMPTY interstitial
land.  A pixel qualifies when
    (a) no OSM polygon of any kind claims it,
    (b) its 810 m neighbourhood is demonstrably well mapped (>=40% of it lies
        within 150 m of a mapped building) -- this is what rules out the
        under-mapped-rural failure mode, where "no buildings" means "nobody
        surveyed it" rather than "empty",
    (c) its own 150 m neighbourhood holds essentially no building (<0.5%),
    (d) it is not the carriageway of a major road or a railway.
That is open, developable land inside the built fabric -- exactly the target
concept -- and it is derived purely from OSM vectors, so it stays independent
of the Sentinel-2 bands the model sees.

Thresholds were fixed a priori from the geometry of the definition; they were
NOT tuned against spectral indices or model score, which would reintroduce the
circularity v2 was built to remove.

0=Vacant(developable) 1=Urban 2=Water 3=Other(non-developable) 255=ignore
"""
import numpy as np, geopandas as gpd, rasterio, json
from rasterio.features import rasterize
from pathlib import Path

TIF = '/home/prithvi/Downloads/Hyderabad_Sentinel2_12Bands_JanMar2025.tif'
OSM = Path('/home/prithvi/osm_integration /hyderabad_osm_data')
OUT = Path('/home/prithvi/AcreVision_v2/data')
CS = 32

VACANT = {'farmland','grass','meadow','orchard','vineyard','allotments','heath'}
URBAN  = {'residential','industrial','commercial','retail'}
WATER  = {'water','reservoir','riverbank'}
OTHER  = {'forest','nature_reserve','park','military','cemetery','quarry','recreation_ground','scrub'}
BIGROAD = {'motorway','trunk','primary','motorway_link','trunk_link','primary_link'}

RING_K, RING_T = 40, 0.40      # 810 m window, >=40% built-around  -> "mapped"
LOCAL_T        = 0.005         # 150 m window, <0.5% building      -> "empty"

with rasterio.open(TIF) as s:
    tr, W, H, crs = s.transform, s.width, s.height, s.crs
shape = (H, W); px_km2 = 1e-4
print(f'raster {W}x{H} {crs}', flush=True)


def burn(gdf):
    if len(gdf) == 0:
        return np.zeros(shape, bool)
    return rasterize(((g, 1) for g in gdf.geometry), out_shape=shape, transform=tr,
                     fill=0, dtype=np.uint8, all_touched=True).astype(bool)


def boxmean(a, k):
    """Dependency-free uniform filter via integral image."""
    p = np.pad(a.astype(np.float32), k, mode='edge')
    c = p.cumsum(0).cumsum(1)
    c = np.pad(c, ((1, 0), (1, 0)))
    s = c[2*k+1:, 2*k+1:] - c[:-2*k-1, 2*k+1:] - c[2*k+1:, :-2*k-1] + c[:-2*k-1, :-2*k-1]
    return s / ((2*k+1)**2)


# ---- building density: reuse v4's raster if present, else rebuild ----
bd_f = OUT/'building_density.npy'
if bd_f.exists():
    bdens = np.load(bd_f)
    print(f'reused {bd_f.name} {bdens.shape}', flush=True)
else:
    b = gpd.read_file(OSM/'gis_osm_buildings_a_free_1_hyderabad.shp').to_crs(crs)
    bdens = boxmean(burn(b), 7)
    np.save(bd_f, bdens.astype(np.float32))
    print(f'built building_density from {len(b)} polys', flush=True)

lu = gpd.read_file(OSM/'gis_osm_landuse_a_free_1_hyderabad.shp').to_crs(crs)
wa = gpd.read_file(OSM/'gis_osm_water_a_free_1_hyderabad.shp').to_crs(crs)
lu_vac = burn(lu[lu.fclass.isin(VACANT)])
lu_urb = burn(lu[lu.fclass.isin(URBAN)])
lu_oth = burn(lu[lu.fclass.isin(OTHER)])
wa_msk = burn(wa[wa.fclass.isin(WATER)])
print(f'  landuse vacant {lu_vac.mean()*100:.2f}%  urban {lu_urb.mean()*100:.2f}%  '
      f'other {lu_oth.mean()*100:.2f}%  water {wa_msk.mean()*100:.2f}%', flush=True)

# ---- (d) major-road carriageway + railway exclusion ----
rd = gpd.read_file(OSM/'gis_osm_roads_free_1_hyderabad.shp').to_crs(crs)
rl = gpd.read_file(OSM/'gis_osm_railways_free_1_hyderabad.shp').to_crs(crs)
corridor = burn(rd[rd.fclass.isin(BIGROAD)]) | burn(rl)
print(f'  major-road+rail corridor {corridor.mean()*100:.2f}% of pixels', flush=True)

# ---- (a)(b)(c) mapped-but-empty interstitial land ----
unclaimed = ~lu_vac & ~lu_urb & ~lu_oth & ~wa_msk
ring = boxmean(bdens > 0, RING_K)
inter = unclaimed & (ring > RING_T) & (bdens < LOCAL_T) & ~corridor
print(f'  interstitial Vacant  +{inter.sum()*px_km2:.1f} km2 '
      f'({inter.mean()*100:.2f}% of AOI)', flush=True)

L = np.full(shape, 255, np.uint8)
L[lu_oth] = 3                                   # non-developable green
L[lu_vac & (bdens < 0.02)] = 0                  # v4 rule: open land by landuse tag
L[inter] = 0                                    # v5 rule: mapped-but-empty land
L[lu_urb | (bdens > 0.15)] = 1                  # zoned urban OR dense built-up
L[wa_msk] = 2                                   # water polygons win outright

# provenance of the final Vacant pixels, for the write-up
vac_final = (L == 0)
src_lu = int((vac_final & lu_vac).sum()); src_in = int((vac_final & inter & ~lu_vac).sum())
print(f'\nVacant provenance: landuse-tag {src_lu*px_km2:.1f} km2 | '
      f'interstitial {src_in*px_km2:.1f} km2', flush=True)

stats = {}
for v, n in [(0,'Vacant'), (1,'Urban'), (2,'Water'), (3,'Other'), (255,'ignore')]:
    stats[n] = round(100*float(np.mean(L == v)), 2)
    print(f'  {n:8s} {stats[n]:6.2f}% of AOI pixels', flush=True)

rc = np.load(OUT/'rowcol.npy')
N = len(rc)
lab = np.lib.format.open_memmap(OUT/'osm_labels6.npy', mode='w+', dtype=np.uint8, shape=(N, CS, CS))
for i, (r, c) in enumerate(rc):
    lab[i] = L[r:r+CS, c:c+CS]
lab.flush()
lp = (lab != 255).reshape(N, -1).mean(1)
np.save(OUT/'osm_label_frac6.npy', lp)
print(f'\nchips with >=25% labelled: {(lp>=.25).sum()}  >=50%: {(lp>=.5).sum()}', flush=True)

old = np.load(OUT/'osm_labels4.npy', mmap_mode='r')
lp4 = np.load(OUT/'osm_label_frac4.npy')
v4 = np.asarray(old[lp4>=.25]); v5 = np.asarray(lab[lp>=.25])
print(f'train-eligible chips  v4 {(lp4>=.25).sum()} -> v5 {(lp>=.25).sum()}', flush=True)
print(f'Vacant pixels in those chips  v4 {(v4==0).sum()} -> v5 {(v5==0).sum()}', flush=True)
json.dump({'aoi_pct':stats,'vacant_km2_landuse':src_lu*px_km2,
           'vacant_km2_interstitial':src_in*px_km2,
           'chips_ge25_v4':int((lp4>=.25).sum()),'chips_ge25_v5':int((lp>=.25).sum()),
           'vacant_px_v4':int((v4==0).sum()),'vacant_px_v5':int((v5==0).sum()),
           'ring_k':RING_K,'ring_t':RING_T,'local_t':LOCAL_T},
          open('/home/prithvi/AcreVision_v2/outputs/labels6.json','w'), indent=2)
print('DONE', flush=True)
