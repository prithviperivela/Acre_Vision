"""Re-chip at 64x64 CENTRED on each parcel: the parcel is the middle 32x32,
with a 16px (160 m) context border. Evaluation still happens on the centre crop
so numbers stay directly comparable to the 32x32 runs."""
import numpy as np, rasterio, geopandas as gpd, time
from pathlib import Path

TIF='/home/prithvi/Downloads/Hyderabad_Sentinel2_12Bands_JanMar2025.tif'
OUT=Path('/home/prithvi/AcreVision_v2/data'); CS=64; PAD=16
t0=time.time()
rc=np.load(OUT/'rowcol.npy'); N=len(rc)
with rasterio.open(TIF) as src:
    W,H=src.width,src.height
    print('loading raster...',flush=True)
    full=src.read().astype(np.float32)
r=np.clip(rc[:,0]-PAD,0,H-CS); c=np.clip(rc[:,1]-PAD,0,W-CS)
# offset of the true parcel inside the padded window (usually PAD, less at edges)
off=np.stack([rc[:,0]-r, rc[:,1]-c],1).astype(np.int16)
ch=np.lib.format.open_memmap(OUT/'chips64.npy',mode='w+',dtype=np.float32,shape=(N,12,CS,CS))
for i in range(N):
    ch[i]=full[:,r[i]:r[i]+CS, c[i]:c[i]+CS]
    if i%8000==0: print(f'  {i}/{N}',flush=True)
ch.flush(); np.save(OUT/'off64.npy',off)
print(f'DONE {time.time()-t0:.0f}s',flush=True)
