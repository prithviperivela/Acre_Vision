"""INDEPENDENT weak labels from OSM vector data -- 4-class version.

Why 4 classes: the 3-class version had no bucket for non-developable green space,
so at inference every rural pixel that was not urban/water was forced into
'Vacant', giving an absurd 51% city-wide vacancy. Forest/park/military now get
their own class and stop absorbing into Vacant.

0=Vacant(developable) 1=Urban 2=Water 3=Other(non-developable) 255=ignore
"""
import numpy as np, geopandas as gpd, rasterio
from rasterio.features import rasterize
from pathlib import Path

TIF = '/home/prithvi/Downloads/Hyderabad_Sentinel2_12Bands_JanMar2025.tif'
OSM = Path('/home/prithvi/osm_integration /hyderabad_osm_data')
OUT = Path('/home/prithvi/AcreVision_v2/data')
CS = 32

VACANT = {'farmland','scrub','grass','meadow','orchard','vineyard','allotments','heath'}
URBAN  = {'residential','industrial','commercial','retail'}
WATER  = {'water','reservoir','riverbank'}
OTHER  = {'forest','nature_reserve','park','military','cemetery','quarry','recreation_ground'}

with rasterio.open(TIF) as s:
    tr, W, H, crs = s.transform, s.width, s.height, s.crs
shape = (H, W)
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


print('rasterising buildings...', flush=True)
b = gpd.read_file(OSM/'gis_osm_buildings_a_free_1_hyderabad.shp').to_crs(crs)
bmask = burn(b)
print(f'  {len(b)} polys, {bmask.mean()*100:.2f}% of pixels', flush=True)
bdens = boxmean(bmask, 7)                       # 15x15 px = 150 m neighbourhood
np.save(OUT/'building_density.npy', bdens.astype(np.float32))

lu = gpd.read_file(OSM/'gis_osm_landuse_a_free_1_hyderabad.shp').to_crs(crs)
wa = gpd.read_file(OSM/'gis_osm_water_a_free_1_hyderabad.shp').to_crs(crs)
lu_vac = burn(lu[lu.fclass.isin(VACANT)])
lu_urb = burn(lu[lu.fclass.isin(URBAN)])
lu_oth = burn(lu[lu.fclass.isin(OTHER)])
wa_msk = burn(wa[wa.fclass.isin(WATER)])
print(f'  landuse vacant {lu_vac.mean()*100:.2f}%  urban {lu_urb.mean()*100:.2f}%  '
      f'other {lu_oth.mean()*100:.2f}%  water {wa_msk.mean()*100:.2f}%', flush=True)

L = np.full(shape, 255, np.uint8)
L[lu_oth] = 3                                   # non-developable green
L[lu_vac & (bdens < 0.02)] = 0                  # developable open land
L[lu_urb | (bdens > 0.15)] = 1                  # zoned urban OR dense built-up
L[wa_msk] = 2                                   # water polygons win outright

for v, n in [(0, 'Vacant'), (1, 'Urban'), (2, 'Water'), (3, 'Other'), (255, 'ignore')]:
    print(f'  {n:8s} {100*np.mean(L == v):6.2f}% of AOI pixels', flush=True)

rc = np.load(OUT/'rowcol.npy')
N = len(rc)
lab = np.lib.format.open_memmap(OUT/'osm_labels4.npy', mode='w+', dtype=np.uint8, shape=(N, CS, CS))
bd = np.zeros(N, np.float32)
for i, (r, c) in enumerate(rc):
    lab[i] = L[r:r+CS, c:c+CS]
    bd[i] = bdens[r:r+CS, c:c+CS].mean()
lab.flush()
lp = (lab != 255).reshape(N, -1).mean(1)
np.save(OUT/'osm_label_frac4.npy', lp)
np.save(OUT/'parcel_building_density.npy', bd)
print(f'\nchips with >=25% labelled: {(lp>=.25).sum()}  >=50%: {(lp>=.5).sum()}', flush=True)
print('DONE', flush=True)
