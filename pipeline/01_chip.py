"""Re-chip the Sentinel-2 GeoTIFF into a single consolidated array.
Replaces the 30,992-loose-file approach with one memmap (faster, no fs thrash)."""
import numpy as np, rasterio, geopandas as gpd, json, time
from pathlib import Path

TIF  = '/home/prithvi/Downloads/Hyderabad_Sentinel2_12Bands_JanMar2025.tif'
GRID = '/home/prithvi/osm_integration /hyderabad_grid_25acre.geojson'
OUT  = Path('/home/prithvi/AcreVision_v2/data')
CS   = 32   # chip size px (317m / 10m = 31.7 -> 32)

t0=time.time()
grid = gpd.read_file(GRID).sort_values('gid').reset_index(drop=True)
N = len(grid); print(f'grid cells: {N}', flush=True)

with rasterio.open(TIF) as src:
    assert src.count == 12
    tr, W, H = src.transform, src.width, src.height
    print(f'raster {W}x{H} x{src.count}  CRS={src.crs}', flush=True)
    print('loading raster into RAM (~1.5 GB)...', flush=True)
    full = src.read().astype(np.float32)          # (12,H,W)
print(f'  loaded in {time.time()-t0:.0f}s', flush=True)

# top-left pixel index for every cell, clamped so a full 32x32 window fits
bx = grid.geometry.bounds
col = np.floor((bx.minx.values - tr.c) / tr.a).astype(int)
row = np.floor((bx.maxy.values - tr.f) / tr.e).astype(int)
col = np.clip(col, 0, W - CS); row = np.clip(row, 0, H - CS)

chips = np.lib.format.open_memmap(OUT/'chips.npy', mode='w+',
                                  dtype=np.float32, shape=(N, 12, CS, CS))
valid = np.zeros(N, bool)
for i in range(N):
    w = full[:, row[i]:row[i]+CS, col[i]:col[i]+CS]
    chips[i] = w
    valid[i] = np.isfinite(w).mean() > 0.5 and np.nanstd(w) > 0
    if i % 5000 == 0: print(f'  {i}/{N}', flush=True)
chips.flush()

np.save(OUT/'gid.npy', grid.gid.values)
np.save(OUT/'valid.npy', valid)
np.save(OUT/'rowcol.npy', np.stack([row, col], 1))
json.dump({'n_cells': int(N), 'n_valid': int(valid.sum()), 'chip_size': CS,
           'tif': TIF, 'transform': list(tr)[:6]}, open(OUT/'chip_meta.json','w'), indent=2)
print(f'\nDONE  valid={valid.sum()}/{N}  ({time.time()-t0:.0f}s)', flush=True)
