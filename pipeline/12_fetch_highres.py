"""Build a gold-label workbench: high-resolution imagery for a stratified chip sample.

The v5 'mapped-but-empty' interstitial rule doubled the Vacant class, but the
post-hoc audit (11_validate_labels) showed those pixels sit spectrally closer to
Urban than to real Vacant, and the contact sheet shows visible street grids
inside them -- i.e. the rule is partly picking up settlements OSM has not mapped.
Deciding this from 10 m Sentinel-2 is unreliable, so we pull Esri World Imagery
at zoom 18 (~0.57 m/px) for a stratified sample and label those by eye.

Output: outputs/gold/chip_<gid>.png  (high-res crop | Sentinel RGB | v5 mask)
        outputs/gold/sample.json     (gid -> stratum, lat, lon, v5 composition)
"""
import numpy as np, json, math, io, time, urllib.request, concurrent.futures as cf
from pathlib import Path
import rasterio
from rasterio.transform import xy
from pyproj import Transformer
from PIL import Image
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

D = Path('/home/prithvi/AcreVision_v2/data'); O = Path('/home/prithvi/AcreVision_v2/outputs')
G = O/'gold'; G.mkdir(exist_ok=True); TC = G/'_tiles'; TC.mkdir(exist_ok=True)
TIF = '/home/prithvi/Downloads/Hyderabad_Sentinel2_12Bands_JanMar2025.tif'
URL = ('https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery'
       '/MapServer/tile/{z}/{y}/{x}')
Z = 18; CS = 32; N_TOTAL = 96

with rasterio.open(TIF) as s: tr, crs = s.transform, s.crs
to_wgs = Transformer.from_crs(crs, 'EPSG:4326', always_xy=True)

L4 = np.load(D/'osm_labels4.npy', mmap_mode='r'); L5 = np.load(D/'osm_labels5.npy', mmap_mode='r')
lp5 = np.load(D/'osm_label_frac5.npy'); valid = np.load(D/'valid.npy')
rc = np.load(D/'rowcol.npy'); CH = np.load(D/'chips.npy', mmap_mode='r')
idx = np.where(valid & (lp5 >= 0.25))[0]

# ---- stratify ----
rng = np.random.default_rng(7)
a5 = np.asarray(L5[idx]); a4 = np.asarray(L4[idx])
flat = a5.reshape(len(idx), -1)
f_new = ((a5 == 0) & (a4 != 0)).reshape(len(idx), -1).mean(1)
f_old = ((a5 == 0) & (a4 == 0)).reshape(len(idx), -1).mean(1)
f = {k: (flat == k).mean(1) for k in range(4)}
strata = {
    'interstitial': (np.where(f_new > 0.5)[0], 48),
    'vacant_tag':   (np.where(f_old > 0.5)[0], 16),
    'urban':        (np.where(f[1] > 0.7)[0], 16),
    'water':        (np.where(f[2] > 0.7)[0],  8),
    'other':        (np.where(f[3] > 0.7)[0],  8),
}
sample = []
for name, (pool, n) in strata.items():
    take = rng.choice(pool, min(n, len(pool)), replace=False)
    print(f'  {name:13s} pool={len(pool):5d} take={len(take)}', flush=True)
    for j in take:
        g = int(idx[j]); r, c = int(rc[g, 0]), int(rc[g, 1])
        X, Y = xy(tr, r + CS/2, c + CS/2); lon, lat = to_wgs.transform(X, Y)
        sample.append(dict(gid=g, stratum=name, lat=float(lat), lon=float(lon),
                           row=r, col=c, f_new=float(f_new[j]), f_old=float(f_old[j]),
                           v5=[round(float(f[k][j]), 3) for k in range(4)]))
print(f'total {len(sample)} chips', flush=True)

# ---- tile maths ----
def deg2num(lat, lon, z):
    n = 2.0**z
    return ((lon+180.0)/360.0*n,
            (1.0-math.asinh(math.tan(math.radians(lat)))/math.pi)/2.0*n)

def fetch(z, x, y):
    p = TC/f'{z}_{y}_{x}.jpg'
    if p.exists(): return p
    req = urllib.request.Request(URL.format(z=z, x=x, y=y),
                                 headers={'User-Agent': 'AcreVision-research/1.0'})
    for a in range(3):
        try:
            with urllib.request.urlopen(req, timeout=25) as r: p.write_bytes(r.read())
            return p
        except Exception:
            time.sleep(1.5*(a+1))
    return None

def highres(lat, lon, span_m=320.0):
    """Mosaic enough z18 tiles to cover span_m around (lat,lon), return cropped array."""
    dlat = span_m/2/111320.0; dlon = span_m/2/(111320.0*math.cos(math.radians(lat)))
    x0f, y0f = deg2num(lat+dlat, lon-dlon, Z)     # NW corner
    x1f, y1f = deg2num(lat-dlat, lon+dlon, Z)     # SE corner
    X0, Y0, X1, Y1 = int(x0f), int(y0f), int(x1f), int(y1f)
    W, H = (X1-X0+1)*256, (Y1-Y0+1)*256
    mos = Image.new('RGB', (W, H))
    jobs = [(X0+i, Y0+j) for i in range(X1-X0+1) for j in range(Y1-Y0+1)]
    with cf.ThreadPoolExecutor(8) as ex:
        for (x, y), p in zip(jobs, ex.map(lambda t: fetch(Z, t[0], t[1]), jobs)):
            if p: mos.paste(Image.open(p).convert('RGB'), ((x-X0)*256, (y-Y0)*256))
    l = int((x0f-X0)*256); t = int((y0f-Y0)*256)
    r = int((x1f-X0)*256); b = int((y1f-Y0)*256)
    return np.asarray(mos.crop((l, t, max(r, l+16), max(b, t+16))))

def sent_rgb(g):
    x = np.asarray(CH[g], np.float32)
    rgb = np.stack([x[3], x[2], x[1]], -1)
    return np.nan_to_num(np.clip(rgb/np.nanpercentile(rgb, 98), 0, 1))

ok = 0
for k, s in enumerate(sample):
    out = G/f'chip_{s["gid"]}.png'
    if out.exists(): ok += 1; continue
    try: hr = highres(s['lat'], s['lon'])
    except Exception as e: print(f'  !! {s["gid"]}: {e}', flush=True); continue
    if hr.size == 0: continue
    m5 = np.asarray(L5[s['gid']]); m4 = np.asarray(L4[s['gid']])
    ov = np.zeros((CS, CS, 3)); ov[..., 0] = (m5 == 0) & (m4 != 0)
    ov[..., 1] = (m5 == 0) & (m4 == 0); ov[..., 2] = (m5 == 1)
    fig, ax = plt.subplots(1, 3, figsize=(12, 4.3))
    ax[0].imshow(hr); ax[0].set_title(f'Esri z{Z} ~0.57m/px  ({hr.shape[1]}x{hr.shape[0]})', fontsize=9)
    ax[1].imshow(sent_rgb(s['gid']), interpolation='nearest'); ax[1].set_title('Sentinel-2 10m RGB', fontsize=9)
    ax[2].imshow(sent_rgb(s['gid'])); ax[2].imshow(ov, alpha=0.5)
    ax[2].set_title('v5 mask  R=new-interstitial G=vacant-tag B=urban', fontsize=8)
    for a in ax: a.axis('off')
    fig.suptitle(f'gid {s["gid"]}   stratum={s["stratum"]}   {s["lat"]:.5f},{s["lon"]:.5f}', fontsize=10)
    plt.tight_layout(); plt.savefig(out, dpi=85); plt.close(fig)
    ok += 1
    if (k+1) % 12 == 0: print(f'  rendered {k+1}/{len(sample)}', flush=True)

json.dump(sample, open(G/'sample.json', 'w'), indent=1)
print(f'\nwrote {ok} chip sheets -> {G}', flush=True)
